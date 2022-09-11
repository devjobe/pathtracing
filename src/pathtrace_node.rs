use std::borrow::Cow;

use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        texture::FallbackImage,
        RenderApp, RenderStage,
    },
};

const WORKGROUP_SIZE: u32 = 8;

pub struct PathTraceNodePlugin;

impl Plugin for PathTraceNodePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractResourcePlugin::<PathTracingImage>::default());
        app.add_plugin(ExtractResourcePlugin::<PathTracingUniform>::default());
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<PathTracingPipeline>()
            .add_system_to_stage(RenderStage::Queue, queue_bind_group);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("path_tracing", PathTracingNode::default());
        render_graph
            .add_node_edge(
                "path_tracing",
                bevy::render::main_graph::node::CAMERA_DRIVER,
            )
            .unwrap();
    }
}

pub struct PathTracingPipeline {
    uniforms_bind_group_layout: BindGroupLayout,
    texture_bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

#[derive(Clone, ShaderType, ExtractResource, AsBindGroup)]
pub struct PathTracingUniform {
    #[uniform(0)]
    pub size: Vec2,
    #[uniform(0)]
    pub frame: f32,
    #[uniform(0)]
    pub time: f32,
    #[uniform(0)]
    pub origin: Vec3,
}

#[derive(Clone, ExtractResource)]
pub struct PathTracingImage(pub Handle<Image>, pub Handle<Image>);

impl FromWorld for PathTracingPipeline {
    fn from_world(world: &mut World) -> Self {
        let uniforms_bind_group_layout =
            PathTracingUniform::bind_group_layout(world.resource::<RenderDevice>());

        let texture_bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::ReadWrite,
                                format: TextureFormat::Rgba32Float,
                                view_dimension: TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::ReadWrite,
                                format: TextureFormat::Rgba32Float,
                                view_dimension: TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

        let shader = world.resource::<AssetServer>().load("pathtracing.wgsl");
        let mut pipeline_cache = world.resource_mut::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![
                texture_bind_group_layout.clone(),
                uniforms_bind_group_layout.clone(),
            ]),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("init"),
        });

        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![
                texture_bind_group_layout.clone(),
                uniforms_bind_group_layout.clone(),
            ]),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("update"),
        });

        PathTracingPipeline {
            uniforms_bind_group_layout,
            texture_bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

struct PathTracingImageBindGroup(BindGroup);

fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<PathTracingPipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    images: Res<PathTracingImage>,
    uniforms: Res<PathTracingUniform>,
    render_device: Res<RenderDevice>,
    fallback_image: Res<FallbackImage>,
) {
    let view = &gpu_images[&images.0];
    let view2 = &gpu_images[&images.1];
    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.texture_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view.texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&view2.texture_view),
            },
        ],
    });

    let prepared = uniforms.as_bind_group(
        &pipeline.uniforms_bind_group_layout,
        &render_device,
        &gpu_images,
        &fallback_image,
    );
    if let Ok(uniform_group) = prepared {
        commands.insert_resource(uniform_group);
    }
    commands.insert_resource(PathTracingImageBindGroup(bind_group));
}

enum PathTracingState {
    Loading,
    Init,
    Update,
}

struct PathTracingNode {
    state: PathTracingState,
}

impl Default for PathTracingNode {
    fn default() -> Self {
        Self {
            state: PathTracingState::Loading,
        }
    }
}

impl render_graph::Node for PathTracingNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<PathTracingPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            PathTracingState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline)
                {
                    self.state = PathTracingState::Init;
                }
            }
            PathTracingState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = PathTracingState::Update;
                }
            }
            PathTracingState::Update => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        if matches!(self.state, PathTracingState::Loading) {
            return Ok(());
        }

        let uniform_bind_group = &world
            .resource::<PreparedBindGroup<PathTracingUniform>>()
            .bind_group;
        let texture_bind_group = &world.resource::<PathTracingImageBindGroup>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<PathTracingPipeline>();

        let uniform = world.resource::<PathTracingUniform>();

        let workgroups = (
            (uniform.size.x as u32) / WORKGROUP_SIZE,
            (uniform.size.y as u32) / WORKGROUP_SIZE,
            1u32,
        );

        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, texture_bind_group, &[]);
        pass.set_bind_group(1, uniform_bind_group, &[]);

        if uniform.frame == 0.0 {
            let init_pipeline = pipeline_cache
                .get_compute_pipeline(pipeline.init_pipeline)
                .unwrap();
            pass.set_pipeline(init_pipeline);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        } else if matches!(self.state, PathTracingState::Update) {
            let update_pipeline = pipeline_cache
                .get_compute_pipeline(pipeline.update_pipeline)
                .unwrap();
            pass.set_pipeline(update_pipeline);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        Ok(())
    }
}
