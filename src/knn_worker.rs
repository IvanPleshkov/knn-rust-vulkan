use std::sync::Arc;

use crate::{
    context::Context,
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    gpu_buffer::{GpuBuffer, GpuBufferType},
    gpu_device::GpuDevice,
    pipeline::Pipeline,
    pipeline_builder::PipelineBuilder,
    shader::Shader,
};

const BLOCK_SIZE: usize = 128;
const SORT_BLOCK_SIZE: usize = 128;

pub struct KnnWorker {
    dim: usize,
    size: usize,
    capacity: usize,
    batch_size: usize,
    batched_count: usize,
    context: Context,
    vector_data_buffer: Arc<GpuBuffer>,
    vector_data_upload_buffer: Arc<GpuBuffer>,
    scores_buffer_odd: Arc<GpuBuffer>,
    scores_buffer_even: Arc<GpuBuffer>,
    scores_download_buffer: Arc<GpuBuffer>,
    knn_uniform_buffer: Arc<GpuBuffer>,
    knn_uniform_buffer_uploader: Arc<GpuBuffer>,
    query_buffer: Arc<GpuBuffer>,
    query_buffer_uploader: Arc<GpuBuffer>,
    scores_pipeline: Arc<Pipeline>,
    take_best_pipelines_odd: Arc<Pipeline>,
    take_best_pipelines_even: Arc<Pipeline>,
}

#[repr(C)]
struct KnnUniformBuffer {
    dim: u32,
    capacity: u32,
    block_size: u32,
    k: u32,
}

impl KnnWorker {
    pub fn new(device: Arc<GpuDevice>, dim: usize, capacity: usize) -> Self {
        let batch_size = 128;
        let vector_data_buffer = Arc::new(GpuBuffer::new(
            device.clone(),
            GpuBufferType::Storage,
            capacity * dim * std::mem::size_of::<f32>(),
        ));

        let vector_data_upload_buffer = Arc::new(GpuBuffer::new(
            device.clone(),
            GpuBufferType::CpuToGpu,
            batch_size * dim * std::mem::size_of::<f32>(),
        ));

        let scores_buffer_odd = Arc::new(GpuBuffer::new(
            device.clone(),
            GpuBufferType::Storage,
            capacity * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>()),
        ));

        let scores_buffer_even = Arc::new(GpuBuffer::new(
            device.clone(),
            GpuBufferType::Storage,
            capacity * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>()),
        ));

        let scores_download_buffer = Arc::new(GpuBuffer::new(
            device.clone(),
            GpuBufferType::GpuToCpu,
            capacity * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>()),
        ));

        let knn_uniform_buffer = Arc::new(GpuBuffer::new(
            device.clone(),
            GpuBufferType::Uniform,
            std::mem::size_of::<KnnUniformBuffer>(),
        ));

        let knn_uniform_buffer_uploader = Arc::new(GpuBuffer::new(
            device.clone(),
            GpuBufferType::CpuToGpu,
            std::mem::size_of::<KnnUniformBuffer>(),
        ));

        let query_buffer = Arc::new(GpuBuffer::new(
            device.clone(),
            GpuBufferType::Storage,
            dim * std::mem::size_of::<f32>(),
        ));

        let query_buffer_uploader = Arc::new(GpuBuffer::new(
            device.clone(),
            GpuBufferType::CpuToGpu,
            dim * std::mem::size_of::<f32>(),
        ));

        let context = Context::new(device.clone());
        let scores_pipeline = Self::create_scores_pipeline(
            device.clone(),
            vector_data_buffer.clone(),
            scores_buffer_odd.clone(),
            knn_uniform_buffer.clone(),
            query_buffer.clone(),
        );
        let (take_best_pipelines_odd, take_best_pipelines_even) = Self::create_best_scores_pipeline(
            device.clone(),
            scores_buffer_odd.clone(),
            scores_buffer_even.clone(),
            knn_uniform_buffer.clone(),
        );
        Self {
            dim,
            size: 0,
            capacity,
            batch_size,
            batched_count: 0,
            context,
            vector_data_buffer,
            vector_data_upload_buffer,
            scores_buffer_odd,
            scores_buffer_even,
            scores_download_buffer,
            knn_uniform_buffer,
            knn_uniform_buffer_uploader,
            query_buffer,
            query_buffer_uploader,
            scores_pipeline,
            take_best_pipelines_odd,
            take_best_pipelines_even,
        }
    }

    pub fn add_vector(&mut self, vector: &[f32], idx: usize) {
        if self.batched_count == self.batch_size {
            self.flush();
        }
        self.size = std::cmp::max(self.size, idx);
        let offset = self.batched_count * self.dim * std::mem::size_of::<f32>();
        self.vector_data_upload_buffer.upload_slice(vector, offset);
        self.context.copy_gpu_buffer(
            self.vector_data_upload_buffer.clone(),
            self.vector_data_buffer.clone(),
            offset,
            idx * self.dim * std::mem::size_of::<f32>(),
            self.dim * std::mem::size_of::<f32>(),
        );
        self.batched_count += 1;
    }

    pub fn remove_vector(&mut self, _vector: &[f32], _idx: u32) {
        unimplemented!()
    }

    pub fn flush(&mut self) {
        self.context.run();
        self.context.wait_finish();
        self.batched_count = 0;
    }

    pub fn knn(&mut self, query: &[f32], count: usize) -> Vec<(u32, f32)> {
        self.update_query_buffer(query);
        self.update_uniform_buffer(count + 1);
        self.flush();
        self.score_all();
        self.take_best(count + 1);
    
        let mut scores = self.download_best_scores(self.scores_buffer_even.clone(), SORT_BLOCK_SIZE * (count + 1));
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scores.iter().take(count).cloned().collect()
    }

    pub fn upload_vector_data(&mut self, data: &[f32]) {
        let vector_data_upload_buffer = Arc::new(GpuBuffer::new(
            self.vector_data_buffer.device.clone(),
            GpuBufferType::CpuToGpu,
            self.capacity * self.dim * std::mem::size_of::<f32>(),
        ));
        vector_data_upload_buffer.upload_slice(data, 0);

        self.context.copy_gpu_buffer(
            vector_data_upload_buffer,
            self.vector_data_buffer.clone(),
            0,
            0,
            self.vector_data_buffer.size,
        );
        self.flush();
    }

    pub fn download_vector_data(&mut self) -> Vec<f32> {
        let vector_data_download_buffer = Arc::new(GpuBuffer::new(
            self.vector_data_buffer.device.clone(),
            GpuBufferType::GpuToCpu,
            self.capacity * self.dim * std::mem::size_of::<f32>(),
        ));

        self.context.copy_gpu_buffer(
            self.vector_data_buffer.clone(),
            vector_data_download_buffer.clone(),
            0,
            0,
            self.vector_data_buffer.size,
        );
        self.flush();
        let mut result: Vec<f32> = vec![0.; self.dim * self.capacity];
        vector_data_download_buffer
            .download_slice(result.as_mut_slice(), 0);
        result
    }

    fn score_all(&mut self) {
        self.context.bind_pipeline(self.scores_pipeline.clone());
        self.context.dispatch(self.size / BLOCK_SIZE + 1, 1, 1);
    }

    fn take_best(&mut self, _count: usize) {
        self.context.bind_pipeline(self.take_best_pipelines_odd.clone());
        self.context.dispatch(self.size / SORT_BLOCK_SIZE + 1, 1, 1);
    }

    fn download_best_scores(
        &mut self,
        scores_buffer: Arc<GpuBuffer>,
        count: usize,
    ) -> Vec<(u32, f32)> {
        self.context.copy_gpu_buffer(
            scores_buffer,
            self.scores_download_buffer.clone(),
            0,
            0,
            count * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>()),
        );
        self.flush();
        let mut result: Vec<(u32, f32)> = vec![(0, 0.); count];
        self.scores_download_buffer
            .download_slice(result.as_mut_slice(), 0);
        result
    }

    fn update_uniform_buffer(&mut self, count: usize) {
        let uniform_buffer = KnnUniformBuffer {
            dim: self.dim as u32,
            capacity: self.capacity as u32,
            block_size: BLOCK_SIZE as u32,
            k: count as u32,
        };
        self.knn_uniform_buffer_uploader.upload(&uniform_buffer, 0);
        self.context.copy_gpu_buffer(
            self.knn_uniform_buffer_uploader.clone(),
            self.knn_uniform_buffer.clone(),
            0,
            0,
            std::mem::size_of::<KnnUniformBuffer>(),
        );
    }

    fn update_query_buffer(&mut self, query: &[f32]) {
        self.query_buffer_uploader.upload_slice(query, 0);
        self.context.copy_gpu_buffer(
            self.query_buffer_uploader.clone(),
            self.query_buffer.clone(),
            0,
            0,
            self.dim * std::mem::size_of::<f32>(),
        );
    }

    fn create_scores_pipeline(
        device: Arc<GpuDevice>,
        vector_data_buffer: Arc<GpuBuffer>,
        scores_buffer: Arc<GpuBuffer>,
        knn_uniform_buffer: Arc<GpuBuffer>,
        query_buffer: Arc<GpuBuffer>,
    ) -> Arc<Pipeline> {
        let shader = Arc::new(Shader::new(
            device.clone(),
            include_bytes!("../shaders/compute_dot_scores.spv"),
        ));

        let descriptor_set_layout = DescriptorSetLayout::builder()
            .add_uniform_buffer(0)
            .add_storage_buffer(1)
            .add_storage_buffer(2)
            .add_storage_buffer(3)
            .build(device.clone());
        let descriptor_set = DescriptorSet::builder(descriptor_set_layout.clone())
            .add_uniform_buffer(0, knn_uniform_buffer)
            .add_storage_buffer(1, vector_data_buffer)
            .add_storage_buffer(2, query_buffer)
            .add_storage_buffer(3, scores_buffer)
            .build();
        PipelineBuilder::new()
            .add_descriptor_set_layout(0, descriptor_set_layout.clone())
            .add_descriptor_set(0, descriptor_set.clone())
            .add_shader(shader)
            .build(device.clone())
    }

    fn create_best_scores_pipeline(
        device: Arc<GpuDevice>,
        scores_buffer_odd: Arc<GpuBuffer>,
        scores_buffer_even: Arc<GpuBuffer>,
        knn_uniform_buffer: Arc<GpuBuffer>,
    ) -> (Arc<Pipeline>, Arc<Pipeline>) {
        let shader = Arc::new(Shader::new(
            device.clone(),
            include_bytes!("../shaders/take_best.spv"),
        ));

        let descriptor_set_layout = DescriptorSetLayout::builder()
            .add_uniform_buffer(0)
            .add_storage_buffer(1)
            .add_storage_buffer(2)
            .build(device.clone());
        let descriptor_set = DescriptorSet::builder(descriptor_set_layout.clone())
            .add_uniform_buffer(0, knn_uniform_buffer.clone())
            .add_storage_buffer(1, scores_buffer_odd.clone())
            .add_storage_buffer(2, scores_buffer_even.clone())
            .build();
        let pipeline_odd = PipelineBuilder::new()
            .add_descriptor_set_layout(0, descriptor_set_layout.clone())
            .add_descriptor_set(0, descriptor_set.clone())
            .add_shader(shader.clone())
            .build(device.clone());

        let descriptor_set_layout = DescriptorSetLayout::builder()
            .add_uniform_buffer(0)
            .add_storage_buffer(1)
            .add_storage_buffer(2)
            .build(device.clone());
        let descriptor_set = DescriptorSet::builder(descriptor_set_layout.clone())
            .add_uniform_buffer(0, knn_uniform_buffer)
            .add_storage_buffer(1, scores_buffer_even.clone())
            .add_storage_buffer(2, scores_buffer_odd.clone())
            .build();
        let pipeline_even = PipelineBuilder::new()
            .add_descriptor_set_layout(0, descriptor_set_layout.clone())
            .add_descriptor_set(0, descriptor_set.clone())
            .add_shader(shader)
            .build(device.clone());

        (pipeline_odd, pipeline_even)
    }
}
