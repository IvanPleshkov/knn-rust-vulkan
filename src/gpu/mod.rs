pub mod context;
pub mod debug_messenger;
pub mod descriptor_set;
pub mod descriptor_set_layout;
pub mod gpu_buffer;
pub mod gpu_device;
pub mod gpu_instance;
pub mod pipeline;
pub mod pipeline_builder;
pub mod shader;

pub trait GpuResource: Send + Sync {}
