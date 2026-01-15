//! GPU-Driven Vegetation System
//!
//! Implements massive-scale vegetation using compute shaders for placement
//! and indirect drawing for zero-CPU-overhead rendering.
//!
//! Features:
//! - Deterministic placement based on world position + seed (multiplayer-safe)
//! - Atomic instance append in compute shader
//! - Indirect draw call (GPU sets instance count)
//! - Wind animation in vertex shader
//!
//! Reference: `docs/04_TASKS/task_12_vegetation.md`

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of vegetation instances.
pub const MAX_VEGETATION_INSTANCES: u32 = 100_000;

/// Spacing between vegetation samples in world units.
pub const VEGETATION_SPACING: f32 = 2.0;

/// Grid size for vegetation compute dispatch (per chunk).
pub const VEGETATION_GRID_SIZE: u32 = 64;

// ============================================================================
// Data Structures
// ============================================================================

/// GPU instance data for a single vegetation object.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GrassInstance {
    /// World position (x, y, z).
    pub position: [f32; 3],
    /// Scale factor for height variation.
    pub scale: f32,
    /// Y-axis rotation in radians.
    pub rotation: f32,
    /// Vegetation type (0=grass, 1=flower, 2=bush).
    pub type_id: u32,
    /// Padding for 32-byte alignment.
    pub _padding: [f32; 2],
}

// Compile-time size verification (must be 32 bytes)
const _: () = assert!(std::mem::size_of::<GrassInstance>() == 32);

/// Arguments for draw_indexed_indirect call.
/// Matches wgpu's DrawIndexedIndirectArgs layout.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DrawIndexedIndirectArgs {
    /// Number of indices per instance (6 for cross-quad).
    pub index_count: u32,
    /// Number of instances to draw (set by GPU!).
    pub instance_count: u32,
    /// First index in the index buffer.
    pub first_index: u32,
    /// Base vertex offset.
    pub base_vertex: i32,
    /// First instance ID.
    pub first_instance: u32,
}

/// Uniform buffer for vegetation compute shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct VegetationUniforms {
    /// Camera position for culling.
    pub camera_x: f32,
    pub camera_y: f32,
    pub camera_z: f32,
    /// World seed for deterministic placement.
    pub seed: f32,
    /// Current time for wind animation.
    pub time: f32,
    /// Vegetation density (0.0-1.0).
    pub density: f32,
    /// Grid spacing.
    pub spacing: f32,
    /// Padding.
    pub _padding: f32,
}

// ============================================================================
// Cross-Quad Mesh
// ============================================================================

/// Vertex for vegetation mesh.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct VegetationVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

/// Generates a cross-quad mesh (2 intersecting quads for 3D grass appearance).
pub fn create_cross_quad_mesh() -> (Vec<VegetationVertex>, Vec<u16>) {
    let half_width = 0.3;
    let height = 1.0;

    // Quad 1: XY plane
    let vertices = vec![
        // First quad (aligned to X axis)
        VegetationVertex { position: [-half_width, 0.0, 0.0], uv: [0.0, 1.0] },
        VegetationVertex { position: [half_width, 0.0, 0.0], uv: [1.0, 1.0] },
        VegetationVertex { position: [half_width, height, 0.0], uv: [1.0, 0.0] },
        VegetationVertex { position: [-half_width, height, 0.0], uv: [0.0, 0.0] },
        // Second quad (aligned to Z axis, rotated 90 degrees)
        VegetationVertex { position: [0.0, 0.0, -half_width], uv: [0.0, 1.0] },
        VegetationVertex { position: [0.0, 0.0, half_width], uv: [1.0, 1.0] },
        VegetationVertex { position: [0.0, height, half_width], uv: [1.0, 0.0] },
        VegetationVertex { position: [0.0, height, -half_width], uv: [0.0, 0.0] },
    ];

    let indices: Vec<u16> = vec![
        // First quad (2 triangles)
        0, 1, 2, 0, 2, 3,
        // Second quad (2 triangles)
        4, 5, 6, 4, 6, 7,
    ];

    (vertices, indices)
}

// ============================================================================
// Vegetation System
// ============================================================================

/// GPU-driven vegetation system with indirect rendering.
pub struct VegetationSystem {
    /// Instance buffer for vegetation positions.
    pub instance_buffer: wgpu::Buffer,
    /// Indirect draw arguments buffer.
    pub indirect_buffer: wgpu::Buffer,
    /// Atomic counter for instance append.
    pub counter_buffer: wgpu::Buffer,
    /// Uniform buffer for compute shader.
    pub uniform_buffer: wgpu::Buffer,
    /// Cross-quad vertex buffer.
    pub vertex_buffer: wgpu::Buffer,
    /// Cross-quad index buffer.
    pub index_buffer: wgpu::Buffer,
    /// Number of indices in the mesh.
    pub index_count: u32,

    /// Compute pipeline for placement.
    pub compute_pipeline: wgpu::ComputePipeline,
    /// Render pipeline for drawing.
    pub render_pipeline: wgpu::RenderPipeline,

    /// Bind group for compute shader.
    pub compute_bind_group: wgpu::BindGroup,
    /// Bind group for render shader.
    pub render_bind_group: wgpu::BindGroup,

    /// Current uniforms.
    pub uniforms: VegetationUniforms,

    /// Whether the system is enabled.
    pub enabled: bool,
}

impl VegetationSystem {
    /// Creates a new vegetation system.
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        terrain_texture_view: &wgpu::TextureView,
        terrain_sampler: &wgpu::Sampler,
    ) -> Self {
        // =====================================================================
        // Buffers
        // =====================================================================

        // Instance buffer (100k instances × 32 bytes = 3.2 MB)
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vegetation Instance Buffer"),
            size: (MAX_VEGETATION_INSTANCES as u64) * std::mem::size_of::<GrassInstance>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Indirect buffer (5 × 4 bytes = 20 bytes)
        let initial_indirect = DrawIndexedIndirectArgs {
            index_count: 12, // Cross-quad has 12 indices
            instance_count: 0,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        };
        let indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vegetation Indirect Buffer"),
            contents: bytemuck::cast_slice(&[initial_indirect]),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Atomic counter buffer (needs COPY_SRC for GPU-to-GPU transfer)
        let counter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vegetation Counter Buffer"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        // Uniform buffer
        let uniforms = VegetationUniforms {
            camera_x: 0.0,
            camera_y: 50.0,
            camera_z: 0.0,
            seed: 12345.0,
            time: 0.0,
            density: 0.4,
            spacing: VEGETATION_SPACING,
            _padding: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vegetation Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // =====================================================================
        // Mesh
        // =====================================================================

        let (vertices, indices) = create_cross_quad_mesh();
        let index_count = indices.len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vegetation Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vegetation Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // =====================================================================
        // Bind Group Layouts
        // =====================================================================

        // Compute bind group layout
        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Vegetation Compute Bind Group Layout"),
            entries: &[
                // Instance buffer (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Indirect buffer (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Counter buffer (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Render bind group layout
        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Vegetation Render Bind Group Layout"),
            entries: &[
                // Instance buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Texture array
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // =====================================================================
        // Bind Groups
        // =====================================================================

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vegetation Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indirect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vegetation Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(terrain_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(terrain_sampler),
                },
            ],
        });

        // =====================================================================
        // Shaders & Pipelines
        // =====================================================================

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vegetation Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/vegetation_compute.wgsl").into(),
            ),
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vegetation Render Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/vegetation_render.wgsl").into(),
            ),
        });

        // Compute pipeline
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vegetation Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Vegetation Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("cs_place_vegetation"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Render pipeline
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vegetation Render Pipeline Layout"),
            bind_group_layouts: &[&render_bind_group_layout, camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Vegetation Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_vegetation"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<VegetationVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_vegetation"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Double-sided for grass
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        log::info!(
            "Vegetation system initialized: max {} instances, {} indices per instance",
            MAX_VEGETATION_INSTANCES,
            index_count
        );

        Self {
            instance_buffer,
            indirect_buffer,
            counter_buffer,
            uniform_buffer,
            vertex_buffer,
            index_buffer,
            index_count,
            compute_pipeline,
            render_pipeline,
            compute_bind_group,
            render_bind_group,
            uniforms,
            enabled: true,
        }
    }

    /// Updates uniforms (call each frame).
    pub fn update(&mut self, queue: &wgpu::Queue, time: f32, camera_pos: (f32, f32, f32)) {
        self.uniforms.time = time;
        self.uniforms.camera_x = camera_pos.0;
        self.uniforms.camera_y = camera_pos.1;
        self.uniforms.camera_z = camera_pos.2;
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
    }

    /// Resets the indirect buffer for a new frame.
    pub fn reset_buffers(&self, queue: &wgpu::Queue) {
        // Reset instance count to 0
        queue.write_buffer(&self.indirect_buffer, 4, bytemuck::cast_slice(&[0u32]));
        // Reset atomic counter to 0
        queue.write_buffer(&self.counter_buffer, 0, bytemuck::cast_slice(&[0u32]));
    }

    /// Dispatches compute shader for vegetation placement.
    pub fn dispatch_compute<'a>(&'a self, compute_pass: &mut wgpu::ComputePass<'a>) {
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
        // Dispatch 64x64 workgroups, each 8x8 threads = 512x512 samples
        compute_pass.dispatch_workgroups(
            VEGETATION_GRID_SIZE / 8,
            VEGETATION_GRID_SIZE / 8,
            1,
        );
    }

    /// Copies atomic counter to indirect buffer's instance_count (GPU-to-GPU).
    /// Call this after compute pass ends but before render pass.
    pub fn copy_counter_to_indirect(&self, encoder: &mut wgpu::CommandEncoder) {
        // Copy 4 bytes from counter_buffer to indirect.instance_count (offset 4)
        encoder.copy_buffer_to_buffer(
            &self.counter_buffer, 0,
            &self.indirect_buffer, 4,  // instance_count is at offset 4
            4,  // sizeof(u32)
        );
    }

    /// Renders vegetation using indirect draw.
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        if !self.enabled {
            return;
        }

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.render_bind_group, &[]);
        render_pass.set_bind_group(1, camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed_indirect(&self.indirect_buffer, 0);
    }
}
