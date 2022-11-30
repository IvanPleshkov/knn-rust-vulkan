use crate::gpu_device::GpuDevice;
use crate::GpuResource;
use ash::vk;
use std::ffi::CStr;
use std::sync::Arc;

pub struct Shader {
    pub device: Arc<GpuDevice>,
    pub name: String,
    pub vk_shader_module: vk::ShaderModule,
    pub vk_pipeline_shader_stage_create_info: vk::PipelineShaderStageCreateInfo,
}

impl GpuResource for Shader {}

impl Drop for Shader {
    fn drop(&mut self) {
        if self.vk_shader_module != vk::ShaderModule::null() {
            unsafe {
                self.device
                    .vk_device
                    .destroy_shader_module(self.vk_shader_module, self.device.alloc());
            }
            self.vk_shader_module = vk::ShaderModule::null();
        }
    }
}

impl Shader {
    pub fn new(device: Arc<GpuDevice>, name: &str, shader_code: &[u8]) -> Self {
        let mut spv_file = std::io::Cursor::new(shader_code);
        let shader_code = ash::util::read_spv(&mut spv_file).unwrap();

        let shader_module_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(&shader_code)
            .build();
        let shader_module = unsafe {
            device
                .vk_device
                .create_shader_module(&shader_module_create_info, device.alloc())
                .unwrap()
        };
        Self {
            device,
            name: name.to_owned(),
            vk_shader_module: shader_module,
            vk_pipeline_shader_stage_create_info: vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(CStr::from_bytes_with_nul(b"main\0").unwrap())
                .build(),
        }
    }
}
