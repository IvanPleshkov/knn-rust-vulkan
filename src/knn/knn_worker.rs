use std::{collections::BinaryHeap, sync::Arc};

use crate::gpu::{
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

#[derive(PartialEq, Clone, Debug, Default)]
#[repr(C)]
pub struct Score {
    pub index: u32,
    pub score: f32,
}

impl Eq for Score {}

impl std::cmp::PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

pub struct KnnWorker {
    dim: usize,
    size: usize,
    capacity: usize,
    batch_size: usize,
    batched_count: usize,
    device: Arc<GpuDevice>,
    context: Context,
    vector_data_buffer: Arc<GpuBuffer>,
    vector_data_upload_buffer: Arc<GpuBuffer>,
    scores_buffer: Arc<GpuBuffer>,
    scores_download_buffer: Arc<GpuBuffer>,
    knn_uniform_buffer: Arc<GpuBuffer>,
    knn_uniform_buffer_uploader: Arc<GpuBuffer>,
    query_buffer: Arc<GpuBuffer>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    descriptor_set: Arc<DescriptorSet>,
    pipeline: Arc<Pipeline>,
}

#[repr(C)]
struct KnnUniformBuffer {
    dim: u32,
    capacity: u32,
    block_size: u32,
    k: u32,
}

impl KnnWorker {
    pub fn new(device: Arc<GpuDevice>, dim: usize) -> Self {
        let capacity = 4096;
        let batch_size = 128;
        let query_buffer = Arc::new(GpuBuffer::new(
            device.clone(),
            GpuBufferType::Storage,
            dim * std::mem::size_of::<f32>(),
        ));

        let vector_data_upload_buffer = Arc::new(GpuBuffer::new(
            device.clone(),
            GpuBufferType::CpuToGpu,
            batch_size * dim * std::mem::size_of::<f32>(),
        ));

        let vector_data_buffer = Arc::new(GpuBuffer::new(
            device.clone(),
            GpuBufferType::Storage,
            capacity * dim * std::mem::size_of::<f32>(),
        ));

        let scores_buffer = Arc::new(GpuBuffer::new(
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

        let descriptor_set_layout = DescriptorSetLayout::builder()
            .add_uniform_buffer(0)
            .add_storage_buffer(1)
            .add_storage_buffer(2)
            .add_storage_buffer(3)
            .build(device.clone());

        let descriptor_set = DescriptorSet::builder(descriptor_set_layout.clone())
            .add_uniform_buffer(0, knn_uniform_buffer.clone())
            .add_storage_buffer(1, vector_data_buffer.clone())
            .add_storage_buffer(2, query_buffer.clone())
            .add_storage_buffer(3, scores_buffer.clone())
            .build();

        let shader = Arc::new(Shader::new(
            device.clone(),
            include_bytes!("../../shaders/compute_dot_scores.spv"),
        ));

        let pipeline = PipelineBuilder::new()
            .add_descriptor_set_layout(0, descriptor_set_layout.clone())
            .add_shader(shader.clone())
            .build(device.clone());

        let mut context = Context::new(device.clone());
        Self::fill_vector_data(
            capacity,
            dim,
            batch_size,
            vector_data_buffer.clone(),
            vector_data_upload_buffer.clone(),
            &mut context,
        );

        Self {
            dim,
            size: 0,
            capacity,
            batch_size,
            batched_count: 0,
            device,
            context,
            vector_data_buffer,
            vector_data_upload_buffer,
            scores_buffer,
            scores_download_buffer,
            knn_uniform_buffer,
            knn_uniform_buffer_uploader,
            query_buffer,
            descriptor_set_layout,
            descriptor_set,
            pipeline,
        }
    }

    pub fn add_vector(&mut self, vector: &[f32], idx: usize) {
        while idx >= self.capacity {
            self.realloc();
        }
        if self.batched_count == self.batch_size {
            self.flush();
        }
        self.size = std::cmp::max(self.size, idx + 1);
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

    pub fn remove_vector(&mut self, idx: usize) {
        if idx < self.size {
            let nan_vec = vec![f32::NAN; self.dim];
            self.add_vector(&nan_vec, idx);
        }
    }

    pub fn flush(&mut self) {
        self.context.run();
        self.context.wait_finish();
        self.batched_count = 0;
    }

    pub fn knn(&mut self, query: &[f32], count: usize) -> Vec<Score> {
        self.update_query_buffer(query);
        self.update_uniform_buffer(count + 1);
        self.flush();
        self.score_all();

        let scores = self.download_best_scores(
            self.scores_buffer.clone(),
            (count + 1) * self.capacity / BLOCK_SIZE,
        );

        let mut heap: BinaryHeap<Score> = BinaryHeap::new();
        for score in scores {
            if heap.len() == count {
                let mut top = heap.peek_mut().unwrap();
                if top.score > score.score {
                    *top = score;
                }
            } else {
                heap.push(score);
            }
        }
        heap.into_sorted_vec()
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
        vector_data_download_buffer.download_slice(result.as_mut_slice(), 0);
        result
    }

    fn score_all(&mut self) {
        self.context
            .bind_pipeline(self.pipeline.clone(), &[self.descriptor_set.clone()]);
        self.context.dispatch(self.capacity / BLOCK_SIZE, 1, 1);
        self.flush();
    }

    fn download_best_scores(&mut self, scores_buffer: Arc<GpuBuffer>, count: usize) -> Vec<Score> {
        self.context.copy_gpu_buffer(
            scores_buffer,
            self.scores_download_buffer.clone(),
            0,
            0,
            count * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>()),
        );
        self.flush();
        let mut result: Vec<Score> = vec![Default::default(); count];
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
        self.vector_data_upload_buffer.upload_slice(query, 0);
        self.context.copy_gpu_buffer(
            self.vector_data_upload_buffer.clone(),
            self.query_buffer.clone(),
            0,
            0,
            self.dim * std::mem::size_of::<f32>(),
        );
    }

    fn fill_vector_data(
        capacity: usize,
        dim: usize,
        batch_size: usize,
        vector_data_buffer: Arc<GpuBuffer>,
        vector_data_upload_buffer: Arc<GpuBuffer>,
        context: &mut Context,
    ) {
        let batch_data = vec![f32::NAN; dim * batch_size];
        vector_data_upload_buffer.upload_slice(batch_data.as_slice(), 0);
        for i in 0..capacity / batch_size {
            context.copy_gpu_buffer(
                vector_data_upload_buffer.clone(),
                vector_data_buffer.clone(),
                0,
                i * batch_size * dim * std::mem::size_of::<f32>(),
                batch_size * dim * std::mem::size_of::<f32>(),
            );
        }
        context.run();
        context.wait_finish();
    }

    fn realloc(&mut self) {
        self.flush();

        self.capacity *= 2;
        let old_vector_data_buffer = self.vector_data_buffer.clone();
        self.vector_data_buffer = Arc::new(GpuBuffer::new(
            self.device.clone(),
            GpuBufferType::Storage,
            self.capacity * self.dim * std::mem::size_of::<f32>(),
        ));

        self.scores_buffer = Arc::new(GpuBuffer::new(
            self.device.clone(),
            GpuBufferType::Storage,
            self.capacity * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>()),
        ));

        self.scores_download_buffer = Arc::new(GpuBuffer::new(
            self.device.clone(),
            GpuBufferType::GpuToCpu,
            self.capacity * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>()),
        ));

        self.descriptor_set = DescriptorSet::builder(self.descriptor_set_layout.clone())
            .add_uniform_buffer(0, self.knn_uniform_buffer.clone())
            .add_storage_buffer(1, self.vector_data_buffer.clone())
            .add_storage_buffer(2, self.query_buffer.clone())
            .add_storage_buffer(3, self.scores_buffer.clone())
            .build();

        let batch_data = vec![f32::NAN; self.dim * self.batch_size];
        self.vector_data_upload_buffer
            .upload_slice(batch_data.as_slice(), 0);
        for i in self.capacity / (2 * self.batch_size)..self.capacity / self.batch_size {
            self.context.copy_gpu_buffer(
                self.vector_data_upload_buffer.clone(),
                self.vector_data_buffer.clone(),
                0,
                i * self.batch_size * self.dim * std::mem::size_of::<f32>(),
                self.batch_size * self.dim * std::mem::size_of::<f32>(),
            );
        }
        self.context.copy_gpu_buffer(
            old_vector_data_buffer,
            self.vector_data_buffer.clone(),
            0,
            0,
            self.vector_data_buffer.size / 2,
        );

        self.context.run();
        self.context.wait_finish();
    }
}
