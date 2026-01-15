//! Terrain System Module
//!
//! GPU-driven multi-chunk terrain system using compute shaders.
//! Implements the "Chunk Pool" pattern for infinite terrain streaming.
//!
//! Features:
//! - 49-chunk pool (7x7 grid) with zero runtime allocation
//! - Chunk recycling when camera moves
//! - Shared index buffer across all chunks
//! - Per-chunk compute dispatch for terrain generation
//! - 5-Channel Climate Model for procedural terrain
//!
//! Reference: `docs/04_TASKS/task_10_multi_chunk.md`

use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use wgpu::util::DeviceExt;

use super::chunk::{
    Chunk, ChunkUniforms, CHUNK_POOL_SIZE, CHUNK_SIZE, GRID_SIZE, RENDER_DISTANCE,
    VERTICES_PER_CHUNK, get_required_chunks, is_chunk_in_range, world_to_chunk_coord,
};
use super::texture::{
    TextureArrayBuilder, TextureData,
    GRASS_COLOR_1, GRASS_COLOR_2,
    ROCK_COLOR_1, ROCK_COLOR_2,
    SNOW_COLOR_1, SNOW_COLOR_2,
    create_terrain_sampler, DEFAULT_TEXTURE_SIZE,
};

// ============================================================================
// Constants
// ============================================================================

/// Total number of indices for one chunk mesh.
pub const INDEX_COUNT: u32 = (GRID_SIZE - 1) * (GRID_SIZE - 1) * 6;

/// Workgroup size matching the compute shader.
const WORKGROUP_SIZE: u32 = 8;

/// Default world seed.
const DEFAULT_SEED: u32 = 12345;

// ============================================================================
// Debug Mode Constants
// ============================================================================

pub const DEBUG_MODE_NORMAL: u32 = 0;
pub const DEBUG_MODE_CONTINENTALNESS: u32 = 1;
pub const DEBUG_MODE_EROSION: u32 = 2;
pub const DEBUG_MODE_PEAKS: u32 = 3;

// ============================================================================
// Terrain Uniforms (Global, not per-chunk)
// ============================================================================

/// Global uniforms for terrain system.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TerrainUniforms {
    pub time: f32,
    pub seed: f32,
    pub debug_mode: u32,
    pub _padding0: u32,
    // Camera position for fog calculation
    pub camera_x: f32,
    pub camera_y: f32,
    pub camera_z: f32,
    pub _padding1: u32,
}

impl TerrainUniforms {
    pub fn new() -> Self {
        Self::with_seed(DEFAULT_SEED)
    }

    pub fn with_seed(seed: u32) -> Self {
        Self {
            time: 0.0,
            seed: (seed as f32) * 0.0001,
            debug_mode: DEBUG_MODE_NORMAL,
            _padding0: 0,
            camera_x: 0.0,
            camera_y: 50.0,
            camera_z: 0.0,
            _padding1: 0,
        }
    }
}

impl Default for TerrainUniforms {
    fn default() -> Self {
        Self::new()
    }
}

// 32 bytes
const _: () = assert!(std::mem::size_of::<TerrainUniforms>() == 32);

// ============================================================================
// Terrain System
// ============================================================================

/// Multi-chunk terrain system with pooled GPU resources.
pub struct TerrainSystem {
    // === Chunk Pool ===
    /// Pool of pre-allocated chunks (49 for 7x7 grid).
    pub chunks: Vec<Chunk>,
    
    /// Current center chunk coordinate (camera position).
    center_chunk: (i32, i32),

    // === Shared Resources ===
    /// Shared index buffer (all chunks use same grid topology).
    pub index_buffer: wgpu::Buffer,
    
    /// Global uniform buffer.
    pub uniform_buffer: wgpu::Buffer,
    
    /// Current global uniform values.
    pub uniforms: TerrainUniforms,

    // === Bind Group Layouts ===
    /// Layout for compute shader bind group 0 (per-chunk: vertex + uniforms).
    compute_chunk_layout: wgpu::BindGroupLayout,
    
    /// Layout for compute shader bind group 1 (global uniforms + index buffer).
    compute_global_layout: wgpu::BindGroupLayout,
    
    /// Global compute bind group.
    pub compute_global_bind_group: wgpu::BindGroup,
    
    /// Layout for render shader bind group 0 (per-chunk: vertex storage).
    render_storage_layout: wgpu::BindGroupLayout,

    // === Pipelines ===
    /// Compute pipeline for index generation.
    pub index_gen_pipeline: wgpu::ComputePipeline,
    
    /// Compute pipeline for terrain generation.
    pub terrain_update_pipeline: wgpu::ComputePipeline,
    
    /// Render pipeline.
    pub render_pipeline: wgpu::RenderPipeline,

    // === Texture Resources ===
    /// Render bind group for textures.
    pub texture_bind_group: wgpu::BindGroup,
    
    /// Render bind group for global uniforms (debug mode, fog).
    pub render_uniform_bind_group: wgpu::BindGroup,

    // === State ===
    /// Whether indices have been generated.
    pub indices_generated: bool,
}

impl TerrainSystem {
    /// Creates a new multi-chunk terrain system.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let uniforms = TerrainUniforms::new();

        // =====================================================================
        // Shared Index Buffer (all chunks use same topology)
        // =====================================================================
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Shared Index Buffer"),
            size: (std::mem::size_of::<u32>() * INDEX_COUNT as usize) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // =====================================================================
        // Global Uniform Buffer
        // =====================================================================
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terrain Global Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // =====================================================================
        // Bind Group Layouts
        // =====================================================================

        // Compute chunk layout (per-chunk: vertex buffer + chunk uniforms)
        let compute_chunk_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Compute Chunk Layout"),
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
                // Chunk uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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

        // Compute global layout (global uniforms + index buffer)
        let compute_global_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Compute Global Layout"),
            entries: &[
                // Index buffer (read_write for generation)
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
                // Global uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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

        // Render storage layout (per-chunk vertex buffer)
        let render_storage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Render Storage Layout"),
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

        // Render uniform layout (for fragment shader)
        let render_uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Render Uniform Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // =====================================================================
        // Bind Groups
        // =====================================================================

        let compute_global_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Compute Global Bind Group"),
            layout: &compute_global_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let render_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Render Uniform Bind Group"),
            layout: &render_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // =====================================================================
        // Textures
        // =====================================================================

        let grass_texture = TextureData::load_or_fallback("assets/textures/grass.png", GRASS_COLOR_1, GRASS_COLOR_2);
        let rock_texture = TextureData::load_or_fallback("assets/textures/rock.png", ROCK_COLOR_1, ROCK_COLOR_2);
        let snow_texture = TextureData::load_or_fallback("assets/textures/snow.png", SNOW_COLOR_1, SNOW_COLOR_2);

        let (texture_array, texture_view) = TextureArrayBuilder::new(DEFAULT_TEXTURE_SIZE, DEFAULT_TEXTURE_SIZE)
            .add_layer(grass_texture)
            .add_layer(rock_texture)
            .add_layer(snow_texture)
            .build(device, queue, "Terrain Texture Array");

        let sampler = create_terrain_sampler(device);

        let render_texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Render Texture Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Texture Bind Group"),
            layout: &render_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // =====================================================================
        // Shader & Pipelines
        // =====================================================================

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../assets/shaders/terrain.wgsl").into()),
        });

        // Compute pipeline layout
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terrain Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_chunk_layout, &compute_global_layout],
            push_constant_ranges: &[],
        });

        let index_gen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Terrain Index Generation Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: Some("cs_generate_indices"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let terrain_update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Terrain Update Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: Some("cs_update_terrain"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Render pipeline layout
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terrain Render Pipeline Layout"),
            bind_group_layouts: &[
                &render_storage_layout,      // Group 0: Per-chunk vertex storage
                camera_bind_group_layout,    // Group 1: Camera
                &render_texture_layout,      // Group 2: Textures
                &render_uniform_layout,      // Group 3: Global uniforms
            ],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terrain Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
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
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
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

        // =====================================================================
        // Initialize Chunk Pool
        // =====================================================================

        let mut chunks = Vec::with_capacity(CHUNK_POOL_SIZE);
        let initial_coords = get_required_chunks((0, 0));
        
        for (i, coord) in initial_coords.iter().enumerate() {
            chunks.push(Chunk::new(
                device,
                *coord,
                &compute_chunk_layout,
                &render_storage_layout,
            ));
        }

        log::info!(
            "Chunk pool initialized: {} chunks ({}x{} grid), {} vertices each",
            CHUNK_POOL_SIZE,
            RENDER_DISTANCE * 2 + 1,
            RENDER_DISTANCE * 2 + 1,
            VERTICES_PER_CHUNK
        );

        Self {
            chunks,
            center_chunk: (0, 0),
            index_buffer,
            uniform_buffer,
            uniforms,
            compute_chunk_layout,
            compute_global_layout,
            compute_global_bind_group,
            render_storage_layout,
            index_gen_pipeline,
            terrain_update_pipeline,
            render_pipeline,
            texture_bind_group,
            render_uniform_bind_group,
            indices_generated: false,
        }
    }

    /// Updates chunk visibility based on camera position.
    /// Recycles out-of-range chunks to new positions.
    pub fn update_chunks(&mut self, queue: &wgpu::Queue, camera_x: f32, camera_z: f32) {
        let camera_chunk = world_to_chunk_coord(camera_x, camera_z);

        // Skip if camera hasn't moved to a new chunk
        if camera_chunk == self.center_chunk {
            return;
        }

        self.center_chunk = camera_chunk;

        // Get required chunk coordinates
        let required_coords: HashSet<(i32, i32)> = get_required_chunks(camera_chunk)
            .into_iter()
            .collect();

        // Get currently loaded coordinates
        let loaded_coords: HashSet<(i32, i32)> = self.chunks.iter()
            .map(|c| c.coord)
            .collect();

        // Find chunks to unload (out of range)
        let to_unload: Vec<usize> = self.chunks.iter()
            .enumerate()
            .filter(|(_, c)| !required_coords.contains(&c.coord))
            .map(|(i, _)| i)
            .collect();

        // Find new positions needed
        let new_positions: Vec<(i32, i32)> = required_coords.difference(&loaded_coords)
            .cloned()
            .collect();

        // Recycle chunks
        for (new_coord, chunk_idx) in new_positions.iter().zip(to_unload.iter()) {
            self.chunks[*chunk_idx].set_coord(queue, *new_coord);
        }

        if !new_positions.is_empty() {
            log::debug!("Recycled {} chunks around camera chunk {:?}", new_positions.len(), camera_chunk);
        }
    }

    /// Updates global uniforms (time, camera position).
    pub fn update_uniforms(&mut self, queue: &wgpu::Queue, time: f32, camera_pos: (f32, f32, f32)) {
        self.uniforms.time = time;
        self.uniforms.camera_x = camera_pos.0;
        self.uniforms.camera_y = camera_pos.1;
        self.uniforms.camera_z = camera_pos.2;
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
    }

    /// Sets debug visualization mode.
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.uniforms.debug_mode = mode.min(3);
        log::info!(
            "Terrain debug mode: {}",
            match self.uniforms.debug_mode {
                0 => "Normal",
                1 => "Continentalness",
                2 => "Erosion",
                3 => "Peaks & Valleys",
                _ => "Unknown",
            }
        );
    }

    /// Gets chunks that need compute update.
    pub fn get_dirty_chunks(&self) -> Vec<usize> {
        self.chunks.iter()
            .enumerate()
            .filter(|(_, c)| c.dirty)
            .map(|(i, _)| i)
            .collect()
    }

    /// Marks all chunks as clean (call after compute pass).
    pub fn clear_dirty_flags(&mut self) {
        for chunk in &mut self.chunks {
            chunk.dirty = false;
        }
    }

    /// Gets active chunks for rendering.
    pub fn get_active_chunks(&self) -> impl Iterator<Item = &Chunk> {
        self.chunks.iter().filter(|c| c.active)
    }

    /// Workgroup count for vertex dispatch.
    pub fn vertex_workgroup_count() -> u32 {
        (GRID_SIZE + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }

    /// Workgroup count for index dispatch.
    pub fn index_workgroup_count() -> u32 {
        (GRID_SIZE - 1 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}
