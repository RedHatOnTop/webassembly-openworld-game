//! Terrain System Module
//!
//! GPU-driven terrain mesh system using compute shaders.
//! Implements the "Compute -> Storage -> Render" pipeline pattern
//! for procedural terrain generation with proper normals and lighting.
//!
//! Features:
//! - Index generation compute shader (one-time)
//! - Position/Normal update compute shader (per-frame)
//! - Solid mesh rendering with TriangleList topology
//! - Directional lighting in fragment shader
//!
//! Reference: `docs/04_TASKS/task_06_terrain_mesh.md`

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ============================================================================
// Constants
// ============================================================================

/// Grid dimension (GRID_SIZE x GRID_SIZE vertices).
pub const GRID_SIZE: u32 = 100;

/// Total number of vertices in the terrain mesh.
pub const VERTEX_COUNT: u32 = GRID_SIZE * GRID_SIZE; // 10,000

/// Total number of indices for the terrain mesh.
/// Each quad (GRID_SIZE-1)^2 produces 2 triangles = 6 indices.
pub const INDEX_COUNT: u32 = (GRID_SIZE - 1) * (GRID_SIZE - 1) * 6; // 58,806

/// Workgroup size matching the compute shader.
const WORKGROUP_SIZE: u32 = 8;

// ============================================================================
// Terrain Vertex Structure
// ============================================================================

/// GPU-side terrain vertex data.
///
/// Memory Layout (32 bytes, 16-byte aligned):
/// - `position`: vec4<f32> (16 bytes) - xyz = world position, w = 1.0
/// - `normal`: vec4<f32> (16 bytes) - xyz = normal vector, w = 0.0
///
/// Uses vec4 for proper storage buffer alignment (16-byte boundaries).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TerrainVertex {
    pub position: [f32; 4],
    pub normal: [f32; 4],
}

impl TerrainVertex {
    /// Creates a zeroed vertex (will be initialized by compute shader).
    pub fn zeroed() -> Self {
        Self {
            position: [0.0, 0.0, 0.0, 1.0],
            normal: [0.0, 1.0, 0.0, 0.0],
        }
    }
}

// Compile-time size verification (must be 32 bytes for proper alignment)
const _: () = assert!(std::mem::size_of::<TerrainVertex>() == 32);

// ============================================================================
// Terrain Uniforms
// ============================================================================

/// Uniforms for terrain compute shader.
///
/// Memory Layout (16 bytes, 16-byte aligned):
/// - `time`: f32 (4 bytes) - Current simulation time
/// - `grid_size`: u32 (4 bytes) - Grid dimension
/// - `_pad`: [u32; 2] (8 bytes) - Padding for 16-byte alignment
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TerrainUniforms {
    pub time: f32,
    pub grid_size: u32,
    pub _pad: [u32; 2],
}

impl TerrainUniforms {
    pub fn new() -> Self {
        Self {
            time: 0.0,
            grid_size: GRID_SIZE,
            _pad: [0; 2],
        }
    }
}

impl Default for TerrainUniforms {
    fn default() -> Self {
        Self::new()
    }
}

// Compile-time size verification
const _: () = assert!(std::mem::size_of::<TerrainUniforms>() == 16);

// ============================================================================
// Terrain System
// ============================================================================

/// Complete terrain system with compute and render pipelines.
///
/// Pipeline flow:
/// 1. Index Generation Compute Pass (one-time): Generate mesh indices
/// 2. Terrain Update Compute Pass (per-frame): Update positions and normals
/// 3. Render Pass: Draw solid terrain mesh with lighting
pub struct TerrainSystem {
    // === Vertex Storage Buffer ===
    /// Storage buffer containing all terrain vertices (position + normal).
    /// Used as read_write in compute shader, read in vertex shader.
    pub vertex_buffer: wgpu::Buffer,

    // === Index Buffer ===
    /// Index buffer for terrain mesh triangles.
    /// Written by compute shader, read by draw_indexed.
    pub index_buffer: wgpu::Buffer,

    // === Uniform Buffer ===
    /// Uniform buffer for compute shader parameters.
    pub uniform_buffer: wgpu::Buffer,
    /// Current uniform values.
    pub uniforms: TerrainUniforms,

    // === Compute Pipelines ===
    /// Compute pipeline for one-time index generation.
    pub index_gen_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for per-frame terrain updates.
    pub terrain_update_pipeline: wgpu::ComputePipeline,

    // === Compute Bind Groups ===
    /// Bind group 0 for compute: vertex storage + index storage (read_write).
    pub compute_bind_group_0: wgpu::BindGroup,
    /// Bind group 1 for compute: uniforms.
    pub compute_bind_group_1: wgpu::BindGroup,

    // === Render Pipeline ===
    /// Render pipeline for solid mesh drawing.
    pub render_pipeline: wgpu::RenderPipeline,
    /// Bind group 0 for render: vertex storage (read).
    pub render_bind_group_0: wgpu::BindGroup,

    // === State Flags ===
    /// Whether indices have been generated (only done once).
    pub indices_generated: bool,
}

impl TerrainSystem {
    /// Creates a new TerrainSystem with all GPU resources initialized.
    ///
    /// # Arguments
    /// * `device` - WebGPU device for resource creation.
    /// * `surface_format` - Surface texture format for render pipeline.
    /// * `camera_bind_group_layout` - Shared camera bind group layout.
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create uniforms
        let uniforms = TerrainUniforms::new();

        // =====================================================================
        // Buffer Creation
        // =====================================================================

        // Vertex storage buffer (32 bytes per vertex * 10,000 vertices = 320,000 bytes)
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Vertex Storage Buffer"),
            size: (std::mem::size_of::<TerrainVertex>() * VERTEX_COUNT as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Index buffer (4 bytes per index * 58,806 indices = 235,224 bytes)
        // Must have STORAGE usage for compute shader write access
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Index Buffer"),
            size: (std::mem::size_of::<u32>() * INDEX_COUNT as usize) as u64,
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uniform buffer
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terrain Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // =====================================================================
        // Bind Group Layouts
        // =====================================================================

        // Compute bind group layout 0: Vertex storage + Index storage (read_write)
        let compute_storage_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Terrain Compute Storage Bind Group Layout"),
                entries: &[
                    // Vertex buffer (read_write)
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
                    // Index buffer (read_write)
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
                ],
            });

        // Compute bind group layout 1: Uniforms
        let compute_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Terrain Compute Uniform Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Render bind group layout 0: Vertex storage (read-only)
        let render_storage_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Terrain Render Storage Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // =====================================================================
        // Bind Groups
        // =====================================================================

        // Compute bind group 0: Storage buffers
        let compute_bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Compute Storage Bind Group"),
            layout: &compute_storage_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index_buffer.as_entire_binding(),
                },
            ],
        });

        // Compute bind group 1: Uniforms
        let compute_bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Compute Uniform Bind Group"),
            layout: &compute_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Render bind group 0: Vertex storage (read-only)
        let render_bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Render Storage Bind Group"),
            layout: &render_storage_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: vertex_buffer.as_entire_binding(),
            }],
        });

        // =====================================================================
        // Shader Module
        // =====================================================================

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/terrain.wgsl").into(),
            ),
        });

        // =====================================================================
        // Compute Pipelines
        // =====================================================================

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Terrain Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_storage_layout, &compute_uniform_layout],
                push_constant_ranges: &[],
            });

        // Index generation pipeline (one-time)
        let index_gen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Terrain Index Generation Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: Some("cs_generate_indices"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Terrain update pipeline (per-frame)
        let terrain_update_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Terrain Update Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: Some("cs_update_terrain"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // =====================================================================
        // Render Pipeline
        // =====================================================================

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Terrain Render Pipeline Layout"),
                // Group 0: Vertex storage buffer
                // Group 1: Camera uniform
                bind_group_layouts: &[&render_storage_layout, camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terrain Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // No vertex buffers - reading from storage
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // Solid mesh!
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // CCW winding order
                cull_mode: Some(wgpu::Face::Back), // Back-face culling enabled
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

        Self {
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            uniforms,
            index_gen_pipeline,
            terrain_update_pipeline,
            compute_bind_group_0,
            compute_bind_group_1,
            render_pipeline,
            render_bind_group_0,
            indices_generated: false,
        }
    }

    /// Updates the time uniform for animation.
    pub fn update_time(&mut self, time: f32) {
        self.uniforms.time = time;
    }

    /// Calculates the number of workgroups needed for vertex dispatch.
    /// ceil(GRID_SIZE / WORKGROUP_SIZE) = ceil(100 / 8) = 13
    pub fn vertex_workgroup_count() -> u32 {
        (GRID_SIZE + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }

    /// Calculates the number of workgroups needed for index dispatch.
    /// ceil((GRID_SIZE - 1) / WORKGROUP_SIZE) = ceil(99 / 8) = 13
    pub fn index_workgroup_count() -> u32 {
        (GRID_SIZE - 1 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}
