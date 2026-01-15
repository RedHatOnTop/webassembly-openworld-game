//! Structure Placement System
//!
//! CPU-driven structure placement for physics-ready objects (trees, rocks, ruins).
//! Uses instanced rendering for efficient batch drawing.
//!
//! Reference: docs/04_TASKS/task_13_structures.md

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::core::assets::{LoadedMesh, ModelManager, StructureType, MeshVertex};

// ============================================================================
// Constants
// ============================================================================

/// Grid spacing for structure placement (much sparser than grass).
pub const STRUCTURE_SPACING: f32 = 12.0;

/// Grid size per chunk for structure sampling.
pub const STRUCTURE_GRID_SIZE: u32 = 8;

/// Maximum structures per chunk.
pub const MAX_STRUCTURES_PER_CHUNK: usize = 64;

// ============================================================================
// Instance Data
// ============================================================================

/// GPU instance data for a structure.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct StructureInstance {
    /// World transform matrix (position, rotation, scale).
    pub transform: [[f32; 4]; 4],
}

impl StructureInstance {
    /// Creates a new instance at the given world position.
    pub fn new(position: Vec3, rotation_y: f32, scale: f32) -> Self {
        let transform = Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            glam::Quat::from_rotation_y(rotation_y),
            position,
        );
        Self {
            transform: transform.to_cols_array_2d(),
        }
    }

    /// Vertex buffer layout for instanced rendering.
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<StructureInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // mat4x4 takes 4 vertex attribute slots
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 4,  // transform row 0
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 5,  // transform row 1
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 6,  // transform row 2
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 7,  // transform row 3
                },
            ],
        }
    }
}

// ============================================================================
// Deterministic Hash (matches vegetation shader)
// ============================================================================

fn hash2d(x: f32, z: f32, seed: f32) -> f32 {
    let p = glam::vec3(x, z, seed) * 0.1031;
    let p = glam::vec3(
        p.x.fract(),
        p.y.fract(),
        p.z.fract(),
    );
    let p = p + glam::Vec3::splat(p.x + p.y + p.z + 33.33);
    ((p.x + p.y) * p.z).fract()
}

// ============================================================================
// Simplified Terrain Height (must match terrain shader)
// ============================================================================

fn simplex_noise_2d_simple(x: f32, z: f32) -> f32 {
    // Simplified noise for CPU placement (approximate, not exact match)
    let p = glam::vec2(x, z);
    let i = glam::vec2(p.x.floor(), p.y.floor());
    let f = glam::vec2(p.x.fract(), p.y.fract());
    
    let u = f * f * (glam::Vec2::splat(3.0) - f * 2.0);
    
    let a = hash2d(i.x, i.y, 0.0);
    let b = hash2d(i.x + 1.0, i.y, 0.0);
    let c = hash2d(i.x, i.y + 1.0, 0.0);
    let d = hash2d(i.x + 1.0, i.y + 1.0, 0.0);
    
    let mix_x1 = a + (b - a) * u.x;
    let mix_x2 = c + (d - c) * u.x;
    
    (mix_x1 + (mix_x2 - mix_x1) * u.y) * 2.0 - 1.0
}

/// Approximate terrain height for CPU placement.
/// Note: For precise placement, we'd need to read from GPU terrain buffer.
fn get_terrain_height_cpu(world_x: f32, world_z: f32, seed: f32) -> f32 {
    let scale = 0.005;
    let pos_x = world_x * scale + seed * 100.0;
    let pos_z = world_z * scale;
    
    // Continental noise (simplified)
    let continental = simplex_noise_2d_simple(pos_x * 0.3, pos_z * 0.3) * 0.6
        + simplex_noise_2d_simple(pos_x * 0.7, pos_z * 0.7) * 0.3;
    
    // Height mapping
    let height = if continental < -0.2 {
        -15.0 + (continental + 1.0) / 0.8 * 10.0
    } else if continental < 0.2 {
        let t = (continental + 0.2) / 0.4;
        -5.0 + t * 15.0
    } else {
        let t = (continental - 0.2) / 0.8;
        10.0 + t * 50.0
    };
    
    // Detail noise
    height + simplex_noise_2d_simple(pos_x * 5.0, pos_z * 5.0) * 3.0
}

/// Get biome-like classification for placement.
fn get_biome_type(world_x: f32, world_z: f32, seed: f32) -> BiomeType {
    let height = get_terrain_height_cpu(world_x, world_z, seed);
    let weirdness = hash2d(world_x * 0.01, world_z * 0.01, seed + 100.0);
    
    if height < 2.0 {
        BiomeType::Water
    } else if height > 80.0 {
        BiomeType::Snow
    } else if weirdness > 0.85 {
        BiomeType::Weird
    } else if hash2d(world_x * 0.02, world_z * 0.02, seed + 200.0) > 0.4 {
        BiomeType::Forest
    } else if hash2d(world_x * 0.03, world_z * 0.03, seed + 300.0) > 0.6 {
        BiomeType::Rocky
    } else {
        BiomeType::Plains
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BiomeType {
    Water,
    Plains,
    Forest,
    Rocky,
    Snow,
    Weird,
}

// ============================================================================
// Structure Generation
// ============================================================================

/// Generates structure instances for a world region around the camera.
pub fn generate_structures(
    camera_x: f32,
    camera_z: f32,
    seed: f32,
) -> Vec<(StructureType, StructureInstance)> {
    let mut structures = Vec::new();
    
    let half_range = 64.0;  // Generate within 64 units of camera
    let start_x = ((camera_x - half_range) / STRUCTURE_SPACING).floor() as i32;
    let end_x = ((camera_x + half_range) / STRUCTURE_SPACING).ceil() as i32;
    let start_z = ((camera_z - half_range) / STRUCTURE_SPACING).floor() as i32;
    let end_z = ((camera_z + half_range) / STRUCTURE_SPACING).ceil() as i32;
    
    for gx in start_x..end_x {
        for gz in start_z..end_z {
            let base_x = gx as f32 * STRUCTURE_SPACING;
            let base_z = gz as f32 * STRUCTURE_SPACING;
            
            // Deterministic jitter
            let jitter_x = (hash2d(base_x, base_z, seed) - 0.5) * STRUCTURE_SPACING * 0.8;
            let jitter_z = (hash2d(base_z, base_x, seed + 1.0) - 0.5) * STRUCTURE_SPACING * 0.8;
            let world_x = base_x + jitter_x;
            let world_z = base_z + jitter_z;
            
            // Density check (15% spawn rate for structures)
            let density_hash = hash2d(world_x * 0.5, world_z * 0.5, seed + 2.0);
            if density_hash > 0.15 {
                continue;
            }
            
            // Get biome
            let biome = get_biome_type(world_x, world_z, seed);
            
            // Skip water and snow
            if biome == BiomeType::Water || biome == BiomeType::Snow {
                continue;
            }
            
            // Get height
            let height = get_terrain_height_cpu(world_x, world_z, seed);
            
            // Determine structure type based on biome
            let structure_type = match biome {
                BiomeType::Forest => {
                    if hash2d(world_x + 10.0, world_z, seed) > 0.5 {
                        StructureType::TreePine
                    } else {
                        StructureType::TreeOak
                    }
                }
                BiomeType::Rocky => {
                    if hash2d(world_x, world_z + 10.0, seed) > 0.7 {
                        StructureType::RockLarge
                    } else {
                        StructureType::RockSmall
                    }
                }
                BiomeType::Weird => {
                    if hash2d(world_x + 20.0, world_z, seed) > 0.5 {
                        StructureType::RuinPillar
                    } else {
                        StructureType::RuinWall
                    }
                }
                BiomeType::Plains => {
                    // Sparse trees in plains
                    if hash2d(world_x, world_z + 20.0, seed) > 0.85 {
                        StructureType::TreeOak
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            
            // Random rotation and scale
            let rotation = hash2d(world_x * 2.0, world_z * 2.0, seed) * std::f32::consts::TAU;
            let scale = 0.8 + hash2d(world_x * 3.0, world_z * 3.0, seed) * 0.4;
            
            let position = Vec3::new(world_x, height, world_z);
            structures.push((structure_type, StructureInstance::new(position, rotation, scale)));
        }
    }
    
    structures
}

// ============================================================================
// Structure System
// ============================================================================

/// Manages structure rendering with instanced draw calls.
pub struct StructureSystem {
    model_manager: ModelManager,
    instance_buffer: wgpu::Buffer,
    instance_data: Vec<(StructureType, StructureInstance)>,
    render_pipeline: wgpu::RenderPipeline,
    pub enabled: bool,
}

impl StructureSystem {
    /// Creates a new structure system.
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Load models
        let model_manager = ModelManager::new(device);
        
        // Create instance buffer
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Structure Instance Buffer"),
            size: (std::mem::size_of::<StructureInstance>() * 1024) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Structure Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/structure.wgsl").into(),
            ),
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Structure Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Structure Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_structure"),
                buffers: &[MeshVertex::desc(), StructureInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_structure"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,  // Opaque
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
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        
        log::info!("Structure system initialized");
        
        Self {
            model_manager,
            instance_buffer,
            instance_data: Vec::new(),
            render_pipeline,
            enabled: true,
        }
    }
    
    /// Updates structure instances based on camera position.
    pub fn update(&mut self, queue: &wgpu::Queue, camera_x: f32, camera_z: f32, seed: f32) {
        self.instance_data = generate_structures(camera_x, camera_z, seed);
        
        // Group by type and upload to GPU
        // For now we just store the data; rendering will batch by type
    }
    
    /// Renders all structures using instanced draw calls.
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        if !self.enabled || self.instance_data.is_empty() {
            return;
        }
        
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        
        // Group instances by structure type for batched rendering
        for structure_type in [
            StructureType::TreePine,
            StructureType::TreeOak,
            StructureType::RockSmall,
            StructureType::RockLarge,
            StructureType::RuinPillar,
            StructureType::RuinWall,
        ] {
            let instances: Vec<StructureInstance> = self.instance_data
                .iter()
                .filter(|(t, _)| *t == structure_type)
                .map(|(_, i)| *i)
                .collect();
            
            if instances.is_empty() {
                continue;
            }
            
            // Upload instances
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
            
            // Get model
            let mesh = self.model_manager.get(structure_type);
            
            // Draw
            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..mesh.index_count, 0, 0..instances.len() as u32);
        }
    }
}
