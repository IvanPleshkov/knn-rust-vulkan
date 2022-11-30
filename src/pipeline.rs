use crate::descriptor_set::DescriptorSet;
use crate::descriptor_set_layout::DescriptorSetLayout;
use crate::gpu_device::GpuDevice;
use crate::pipeline_builder::PipelineBuilder;
use crate::shader::Shader;
use crate::GpuResource;
use ash::vk;
use std::sync::Arc;

#[allow(dead_code)]
pub struct Pipeline {
    pub(crate) device: Arc<GpuDevice>,
    pub(crate) name: String,
    pub(crate) shader: Arc<Shader>,
    pub(crate) descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>>,
    pub(crate) descriptor_sets: Vec<Arc<DescriptorSet>>,
    pub(crate) vk_pipeline_layout: vk::PipelineLayout,
    pub(crate) vk_pipeline: vk::Pipeline,
}

impl GpuResource for Pipeline {}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if self.vk_pipeline != vk::Pipeline::null() {
            unsafe {
                self.device
                    .vk_device
                    .destroy_pipeline(self.vk_pipeline, self.device.alloc());
            }
            self.vk_pipeline = vk::Pipeline::null();
        }
        if self.vk_pipeline_layout != vk::PipelineLayout::null() {
            unsafe {
                self.device
                    .vk_device
                    .destroy_pipeline_layout(self.vk_pipeline_layout, self.device.alloc());
            }
            self.vk_pipeline_layout = vk::PipelineLayout::null();
        }
        self.descriptor_set_layouts.clear();
    }
}

impl Pipeline {
    pub(crate) fn new(device: Arc<GpuDevice>, name: &str, builder: &PipelineBuilder) -> Self {
        let descriptor_set_layouts: Vec<_> =
            builder.descriptor_set_layouts.values().cloned().collect();
        let vk_descriptor_set_layouts: Vec<_> = descriptor_set_layouts
            .iter()
            .map(|set| set.vk_descriptor_set_layout)
            .collect();
        let descriptor_sets: Vec<_> = builder.descriptor_sets.values().cloned().collect();
        let vk_pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&vk_descriptor_set_layouts)
            .push_constant_ranges(&[])
            .build();
        let vk_pipeline_layout = unsafe {
            device
                .vk_device
                .create_pipeline_layout(&vk_pipeline_layout_create_info, device.alloc())
                .unwrap()
        };
        let vk_pipeline_shader_stage_create_info = builder
            .shader
            .clone()
            .unwrap()
            .vk_pipeline_shader_stage_create_info;

        let vk_compute_pipeline_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(vk_pipeline_shader_stage_create_info)
            .layout(vk_pipeline_layout)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1)
            .build();
        let vk_pipelines = unsafe {
            device
                .vk_device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk_compute_pipeline_create_info],
                    device.alloc(),
                )
                .unwrap()
        };

        Self {
            device,
            name: name.to_owned(),
            shader: builder.shader.clone().unwrap(),
            vk_pipeline_layout,
            vk_pipeline: vk_pipelines[0],
            descriptor_set_layouts,
            descriptor_sets,
        }
    }
}