use crate::descriptor_set::DescriptorSet;
use crate::gpu_buffer::GpuBuffer;
use crate::gpu_device::GpuDevice;
use crate::pipeline::Pipeline;
use crate::GpuResource;
use ash::vk;
use std::sync::Arc;

pub struct Context {
    pub device: Arc<GpuDevice>,
    pub vk_queue: vk::Queue,
    pub vk_queue_family_index: usize,
    pub vk_command_pool: vk::CommandPool,
    pub vk_command_buffer: vk::CommandBuffer,
    pub vk_fence: vk::Fence,
    pub gpu_resources: Vec<Arc<dyn GpuResource>>,
}

impl Drop for Context {
    fn drop(&mut self) {
        self.wait_finish();
        if self.vk_fence != vk::Fence::null() {
            unsafe {
                self.device
                    .vk_device
                    .destroy_fence(self.vk_fence, self.device.alloc());
            }
            self.vk_fence = vk::Fence::null();
        }
        if self.vk_command_pool != vk::CommandPool::null() {
            unsafe {
                self.device
                    .vk_device
                    .destroy_command_pool(self.vk_command_pool, self.device.alloc());
            }
            self.vk_command_pool = vk::CommandPool::null();
        }
    }
}

impl Context {
    pub fn new(device: Arc<GpuDevice>) -> Self {
        let queue;
        let vk_command_pool;
        let vk_fence;
        {
            queue = device.compute_queues[0].clone();

            let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(queue.vk_queue_family_index as u32)
                .flags(vk::CommandPoolCreateFlags::default());
            vk_command_pool = unsafe {
                device
                    .vk_device
                    .create_command_pool(&command_pool_create_info, device.alloc())
                    .unwrap()
            };

            let fence_create_info =
                vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::default());
            vk_fence = unsafe {
                device
                    .vk_device
                    .create_fence(&fence_create_info, device.alloc())
                    .unwrap()
            };
        }

        let mut context = Self {
            device,
            vk_queue: queue.vk_queue,
            vk_queue_family_index: queue.vk_queue_family_index,
            vk_command_pool,
            vk_command_buffer: vk::CommandBuffer::null(),
            vk_fence,
            gpu_resources: Vec::new(),
        };
        context.init_command_buffer();
        context
    }

    pub fn draw_empty(&mut self, verts_count: usize) {
        if self.vk_command_buffer == vk::CommandBuffer::null() {
            self.init_command_buffer();
        }

        unsafe {
            self.device
                .vk_device
                .cmd_draw(self.vk_command_buffer, verts_count as u32, 1, 0, 0);
        }
    }

    pub fn dispatch(&mut self, x: usize, y: usize, z: usize) {
        if self.vk_command_buffer == vk::CommandBuffer::null() {
            self.init_command_buffer();
        }

        unsafe {
            self.device.vk_device.cmd_dispatch(
                self.vk_command_buffer,
                x as u32,
                y as u32,
                z as u32,
            );
        }
    }

    pub fn bind_pipeline(
        &mut self,
        pipeline: Arc<Pipeline>,
        descriptor_sets: &[Arc<DescriptorSet>],
    ) {
        if self.vk_command_buffer == vk::CommandBuffer::null() {
            self.init_command_buffer();
        }

        unsafe {
            self.device.vk_device.cmd_bind_pipeline(
                self.vk_command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.vk_pipeline,
            );
        }

        unsafe {
            if !descriptor_sets.is_empty() {
                let vk_descriptor_sets: Vec<_> = descriptor_sets
                    .iter()
                    .map(|set| set.as_ref().vk_descriptor_set)
                    .collect();
                self.device.vk_device.cmd_bind_descriptor_sets(
                    self.vk_command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.vk_pipeline_layout,
                    0,
                    &vk_descriptor_sets,
                    &[],
                );
            }
        }

        self.gpu_resources.extend(
            descriptor_sets
                .iter()
                .map(|r| r.clone() as Arc<dyn GpuResource>),
        );
        self.gpu_resources.push(pipeline)
    }

    pub fn copy_gpu_buffer(
        &mut self,
        src: Arc<GpuBuffer>,
        dst: Arc<GpuBuffer>,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) {
        if self.vk_command_buffer == vk::CommandBuffer::null() {
            self.init_command_buffer();
        }

        let buffer_copy = vk::BufferCopy::builder()
            .src_offset(src_offset as vk::DeviceSize)
            .dst_offset(dst_offset as vk::DeviceSize)
            .size(size as vk::DeviceSize)
            .build();
        unsafe {
            self.device.vk_device.cmd_copy_buffer(
                self.vk_command_buffer,
                src.vk_buffer,
                dst.vk_buffer,
                &[buffer_copy],
            );
        }

        self.gpu_resources.push(src);
        self.gpu_resources.push(dst);
    }

    pub fn run(&mut self) {
        if self.vk_command_buffer == vk::CommandBuffer::null() {
            return;
        }
        unsafe {
            self.device
                .vk_device
                .end_command_buffer(self.vk_command_buffer)
                .unwrap();
        }

        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&[self.vk_command_buffer])
            .build();
        unsafe {
            self.device
                .vk_device
                .queue_submit(self.vk_queue, &[submit_info], self.vk_fence)
                .unwrap();
        }
    }

    pub fn wait_finish(&mut self) {
        if self.vk_command_buffer == vk::CommandBuffer::null() {
            return;
        }

        unsafe {
            self.device
                .vk_device
                .wait_for_fences(&[self.vk_fence], true, u64::MAX)
                .unwrap();
            self.device
                .vk_device
                .reset_fences(&[self.vk_fence])
                .unwrap();
        }
        self.destroy_command_buffer();
    }

    fn init_command_buffer(&mut self) {
        if self.vk_command_buffer != vk::CommandBuffer::null() {
            panic!("vk command buffer was already created");
        }

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.vk_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let vk_command_buffer = unsafe {
            self.device
                .vk_device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap()[0]
        };
        self.vk_command_buffer = vk_command_buffer;

        let command_buffer_begin_info =
            vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::default());
        //.inheritance_info(..);
        unsafe {
            self.device
                .vk_device
                .begin_command_buffer(self.vk_command_buffer, &command_buffer_begin_info)
                .unwrap();
        }
    }

    fn destroy_command_buffer(&mut self) {
        if self.vk_command_buffer != vk::CommandBuffer::null() {
            unsafe {
                self.device
                    .vk_device
                    .free_command_buffers(self.vk_command_pool, &[self.vk_command_buffer]);
            }
            self.vk_command_buffer = vk::CommandBuffer::null();
        }
        self.gpu_resources.clear();
    }
}
