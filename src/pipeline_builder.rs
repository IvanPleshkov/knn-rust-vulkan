use std::{collections::BTreeMap, sync::Arc};

use crate::{
    descriptor_set::DescriptorSet, descriptor_set_layout::DescriptorSetLayout,
    gpu_device::GpuDevice, pipeline::Pipeline, shader::Shader,
};

pub struct PipelineBuilder {
    pub(crate) shader: Option<Arc<Shader>>,
    pub(crate) descriptor_set_layouts: BTreeMap<usize, Arc<DescriptorSetLayout>>,
    pub(crate) descriptor_sets: BTreeMap<usize, Arc<DescriptorSet>>,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            shader: None,
            descriptor_set_layouts: BTreeMap::new(),
            descriptor_sets: BTreeMap::new(),
        }
    }

    pub fn add_shader(mut self, shader: Arc<Shader>) -> Self {
        self.shader = Some(shader);
        self
    }

    pub fn add_descriptor_set_layout(
        mut self,
        set: usize,
        descriptor_set_layout: Arc<DescriptorSetLayout>,
    ) -> Self {
        self.descriptor_set_layouts
            .insert(set, descriptor_set_layout);
        self
    }

    pub fn add_descriptor_set(mut self, set: usize, descriptor_set: Arc<DescriptorSet>) -> Self {
        self.descriptor_sets.insert(set, descriptor_set);
        self
    }

    pub fn build(&self, device: Arc<GpuDevice>, name: &str) -> Arc<Pipeline> {
        Arc::new(Pipeline::new(device, name, self))
    }
}
