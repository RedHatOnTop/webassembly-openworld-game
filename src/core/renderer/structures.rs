//! Structure Placement System
//!
//! CPU-driven structure placement for physics-ready objects (trees, rocks, ruins).
//! Uses instanced rendering for efficient batch drawing.
//!
//! CRITICAL: Height function MUST match terrain.wgsl exactly!
//!
//! Reference: docs/04_TASKS/task_13_structures.md

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::core::assets::{ModelManager, StructureType, MeshVertex};

// ============================================================================
// Constants
// ============================================================================

/// Grid spacing for structure placement (much sparser than grass).
pub const STRUCTURE_SPACING: f32 = 16.0;

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
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 5,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 6,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 7,
                },
            ],
        }
    }
}

// ============================================================================
// NOISE LIBRARY - EXACT COPY FROM terrain.wgsl
// ============================================================================

const F3: f32 = 0.3333333333;
const G3: f32 = 0.1666666667;

const GRAD3: [[f32; 3]; 12] = [
    [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
];

fn hash_int(x: i32) -> u32 {
    let mut n = x as u32;
    n = (n ^ 61) ^ (n >> 16);
    n = n.wrapping_add(n << 3);
    n = n ^ (n >> 4);
    n = n.wrapping_mul(0x27d4eb2d);
    n = n ^ (n >> 15);
    n
}

fn hash_3d(x: i32, y: i32, z: i32) -> u32 {
    hash_int(x.wrapping_add(hash_int(y.wrapping_add(hash_int(z) as i32)) as i32))
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn simplex_noise_3d(px: f32, py: f32, pz: f32) -> f32 {
    let s = (px + py + pz) * F3;
    let i = (px + s).floor();
    let j = (py + s).floor();
    let k = (pz + s).floor();
    
    let t = (i + j + k) * G3;
    let x0_base = i - t;
    let y0_base = j - t;
    let z0_base = k - t;
    
    let x0 = px - x0_base;
    let y0 = py - y0_base;
    let z0 = pz - z0_base;
    
    let (i1, j1, k1, i2, j2, k2);
    if x0 >= y0 {
        if y0 >= z0 { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
        else if x0 >= z0 { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
        else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
    } else {
        if y0 < z0 { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
        else if x0 < z0 { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
        else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
    }
    
    let x1 = x0 - i1 as f32 + G3;
    let y1 = y0 - j1 as f32 + G3;
    let z1 = z0 - k1 as f32 + G3;
    let x2 = x0 - i2 as f32 + 2.0 * G3;
    let y2 = y0 - j2 as f32 + 2.0 * G3;
    let z2 = z0 - k2 as f32 + 2.0 * G3;
    let x3 = x0 - 1.0 + 3.0 * G3;
    let y3 = y0 - 1.0 + 3.0 * G3;
    let z3 = z0 - 1.0 + 3.0 * G3;
    
    let ii = (i as i32) & 255;
    let jj = (j as i32) & 255;
    let kk = (k as i32) & 255;
    
    let mut n0 = 0.0f32;
    let mut n1 = 0.0f32;
    let mut n2 = 0.0f32;
    let mut n3 = 0.0f32;
    
    let mut t0 = 0.6 - x0*x0 - y0*y0 - z0*z0;
    if t0 > 0.0 {
        t0 *= t0;
        let gi0 = (hash_3d(ii, jj, kk) % 12) as usize;
        n0 = t0 * t0 * dot3(GRAD3[gi0], [x0, y0, z0]);
    }
    
    let mut t1 = 0.6 - x1*x1 - y1*y1 - z1*z1;
    if t1 > 0.0 {
        t1 *= t1;
        let gi1 = (hash_3d(ii + i1, jj + j1, kk + k1) % 12) as usize;
        n1 = t1 * t1 * dot3(GRAD3[gi1], [x1, y1, z1]);
    }
    
    let mut t2 = 0.6 - x2*x2 - y2*y2 - z2*z2;
    if t2 > 0.0 {
        t2 *= t2;
        let gi2 = (hash_3d(ii + i2, jj + j2, kk + k2) % 12) as usize;
        n2 = t2 * t2 * dot3(GRAD3[gi2], [x2, y2, z2]);
    }
    
    let mut t3 = 0.6 - x3*x3 - y3*y3 - z3*z3;
    if t3 > 0.0 {
        t3 *= t3;
        let gi3 = (hash_3d(ii + 1, jj + 1, kk + 1) % 12) as usize;
        n3 = t3 * t3 * dot3(GRAD3[gi3], [x3, y3, z3]);
    }
    
    32.0 * (n0 + n1 + n2 + n3)
}

fn fbm_3d(px: f32, py: f32, pz: f32, octaves: i32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 0.5f32;
    let mut frequency = 1.0f32;
    let mut max_value = 0.0f32;
    
    for _ in 0..octaves {
        value += amplitude * simplex_noise_3d(px * frequency, py * frequency, pz * frequency);
        max_value += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    value / max_value
}

fn ridged_fbm_3d(px: f32, py: f32, pz: f32, octaves: i32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 0.5f32;
    let mut frequency = 1.0f32;
    let mut max_value = 0.0f32;
    let mut weight = 1.0f32;
    
    for _ in 0..octaves {
        let n = (1.0 - simplex_noise_3d(px * frequency, py * frequency, pz * frequency).abs()) * weight;
        value += n * amplitude;
        max_value += amplitude;
        weight = (n * 2.0).clamp(0.0, 1.0);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    value / max_value
}

fn domain_warp_3d(px: f32, py: f32, pz: f32, amplitude: f32, frequency: f32) -> (f32, f32, f32) {
    let wx = simplex_noise_3d(px * frequency, py * frequency, pz * frequency);
    let wy = simplex_noise_3d(px * frequency + 5.2, py * frequency + 1.3, pz * frequency);
    let wz = simplex_noise_3d(px * frequency, py * frequency, pz * frequency + 9.4);
    (px + wx * amplitude, py + wy * amplitude, pz + wz * amplitude)
}

// Continentalness spline - EXACT copy from terrain.wgsl
const CONTINENTALNESS_SPLINE: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [-0.5, -0.8],
    [-0.2, -0.3],
    [0.0, 0.1],
    [0.4, 0.5],
    [1.0, 1.0],
];

fn cubic_spline(t: f32) -> f32 {
    let clamped_t = t.clamp(-1.0, 1.0);
    
    if clamped_t <= CONTINENTALNESS_SPLINE[0][0] { return CONTINENTALNESS_SPLINE[0][1]; }
    if clamped_t >= CONTINENTALNESS_SPLINE[5][0] { return CONTINENTALNESS_SPLINE[5][1]; }
    
    for i in 0..5 {
        if clamped_t < CONTINENTALNESS_SPLINE[i + 1][0] {
            let seg_t = (clamped_t - CONTINENTALNESS_SPLINE[i][0]) 
                      / (CONTINENTALNESS_SPLINE[i + 1][0] - CONTINENTALNESS_SPLINE[i][0]);
            return CONTINENTALNESS_SPLINE[i][1] 
                 + seg_t * (CONTINENTALNESS_SPLINE[i + 1][1] - CONTINENTALNESS_SPLINE[i][1]);
        }
    }
    
    CONTINENTALNESS_SPLINE[5][1]
}

struct ClimateChannels {
    continentalness: f32,
    erosion: f32,
    peaks_valleys: f32,
}

fn get_climate_channels(world_x: f32, world_z: f32, seed: f32) -> ClimateChannels {
    let seed_offset_x = seed * 1000.0;
    let seed_offset_z = seed * 500.0;
    let (warped_x, _, warped_z) = domain_warp_3d(
        world_x + seed_offset_x, 0.0, world_z + seed_offset_z, 20.0, 0.01
    );
    
    let cont_raw = fbm_3d(warped_x * 0.005, 0.0, warped_z * 0.005, 4);
    let continentalness = cubic_spline(cont_raw);
    
    let erosion_x = world_x + seed * 2000.0 + 1000.0;
    let erosion_z = world_z + seed * 1000.0;
    let erosion = fbm_3d(erosion_x * 0.01, 0.0, erosion_z * 0.01, 3);
    
    let peaks_x = world_x + seed * 1500.0;
    let peaks_z = world_z + seed * 750.0;
    let peaks_valleys = ridged_fbm_3d(peaks_x * 0.008, 0.0, peaks_z * 0.008, 4);
    
    ClimateChannels { continentalness, erosion, peaks_valleys }
}

/// Get terrain height - EXACT COPY from terrain.wgsl
fn get_terrain_height(world_x: f32, world_z: f32, seed: f32) -> f32 {
    let climate = get_climate_channels(world_x, world_z, seed);
    let sea_level = 0.0;
    
    // Base Height from Continentalness
    let base_height = if climate.continentalness < -0.2 {
        let ocean_t = (climate.continentalness + 1.0) / 0.8;
        sea_level - 15.0 + ocean_t * 10.0
    } else if climate.continentalness < 0.2 {
        let coast_t = (climate.continentalness + 0.2) / 0.4;
        sea_level - 5.0 + coast_t * 10.0
    } else {
        let land_t = (climate.continentalness - 0.2) / 0.8;
        sea_level + 5.0 + land_t * 35.0
    };
    
    // Ocean Mask
    let ocean_mask = smoothstep(-0.1, 0.1, climate.continentalness);
    
    // Apply Peaks/Valleys (masked by ocean)
    let mountain_strength = smoothstep(0.0, 0.5, climate.continentalness);
    let peak_height = climate.peaks_valleys * 25.0 * mountain_strength * ocean_mask;
    
    // Erosion Effect
    let erosion_factor = 1.0 - (climate.erosion * 0.5 + 0.5) * 0.3;
    
    // Detail Noise (also masked by ocean)
    let detail_strength = smoothstep(-0.1, 0.3, climate.continentalness) * 3.0 * ocean_mask;
    let detail = simplex_noise_3d(world_x * 0.1 + seed * 100.0, 0.0, world_z * 0.1) * detail_strength;
    
    // Combine
    let mut height = base_height + peak_height * erosion_factor + detail;
    
    // Ocean floor ripples
    if climate.continentalness < -0.2 {
        height += simplex_noise_3d(world_x * 0.03, 0.0, world_z * 0.03) * 1.0;
    }
    
    height
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ============================================================================
// Deterministic Hash for jitter
// ============================================================================

fn hash2d(x: f32, z: f32, seed: f32) -> f32 {
    let p = glam::vec3(x, z, seed) * 0.1031;
    let p = glam::vec3(p.x.fract(), p.y.fract(), p.z.fract());
    let p = p + glam::Vec3::splat(p.x + p.y + p.z + 33.33);
    ((p.x + p.y) * p.z).fract()
}

// ============================================================================
// Biome Classification
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
enum BiomeType {
    Ocean,
    Beach,
    Plains,
    Forest,
    Rocky,
    Snow,
    Weird,
}

fn get_biome_type(world_x: f32, world_z: f32, seed: f32) -> BiomeType {
    let height = get_terrain_height(world_x, world_z, seed);
    let climate = get_climate_channels(world_x, world_z, seed);
    let weirdness = hash2d(world_x * 0.01, world_z * 0.01, seed + 100.0);
    
    // Ocean/underwater
    if height < 1.0 || climate.continentalness < -0.1 {
        return BiomeType::Ocean;
    }
    
    // Beach (low coastal areas)
    if height < 5.0 && climate.continentalness < 0.1 {
        return BiomeType::Beach;
    }
    
    // Snow (high altitude)
    if height > 60.0 {
        return BiomeType::Snow;
    }
    
    // Rocky (steep or high areas)
    if height > 40.0 || climate.peaks_valleys > 0.7 {
        return BiomeType::Rocky;
    }
    
    // Weird ruins (rare)
    if weirdness > 0.92 {
        return BiomeType::Weird;
    }
    
    // Forest vs Plains based on noise
    if hash2d(world_x * 0.02, world_z * 0.02, seed + 200.0) > 0.45 {
        BiomeType::Forest
    } else {
        BiomeType::Plains
    }
}

// ============================================================================
// Structure Generation
// ============================================================================

pub fn generate_structures(
    camera_x: f32,
    camera_z: f32,
    seed: f32,
) -> Vec<(StructureType, StructureInstance)> {
    let mut structures = Vec::new();
    
    let half_range = 80.0;
    let start_x = ((camera_x - half_range) / STRUCTURE_SPACING).floor() as i32;
    let end_x = ((camera_x + half_range) / STRUCTURE_SPACING).ceil() as i32;
    let start_z = ((camera_z - half_range) / STRUCTURE_SPACING).floor() as i32;
    let end_z = ((camera_z + half_range) / STRUCTURE_SPACING).ceil() as i32;
    
    for gx in start_x..end_x {
        for gz in start_z..end_z {
            let base_x = gx as f32 * STRUCTURE_SPACING;
            let base_z = gz as f32 * STRUCTURE_SPACING;
            
            // Jitter
            let jitter_x = (hash2d(base_x, base_z, seed) - 0.5) * STRUCTURE_SPACING * 0.7;
            let jitter_z = (hash2d(base_z, base_x, seed + 1.0) - 0.5) * STRUCTURE_SPACING * 0.7;
            let world_x = base_x + jitter_x;
            let world_z = base_z + jitter_z;
            
            // Density check (10% spawn rate)
            let density_hash = hash2d(world_x * 0.5, world_z * 0.5, seed + 2.0);
            if density_hash > 0.10 {
                continue;
            }
            
            // Get biome
            let biome = get_biome_type(world_x, world_z, seed);
            
            // Skip ocean, beach, snow
            if matches!(biome, BiomeType::Ocean | BiomeType::Beach | BiomeType::Snow) {
                continue;
            }
            
            // Get height at JITTERED position
            let height = get_terrain_height(world_x, world_z, seed);
            
            // Extra safety: skip if underwater
            if height < 3.0 {
                continue;
            }
            
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
                    if hash2d(world_x, world_z + 10.0, seed) > 0.6 {
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
                    // Very sparse trees in plains
                    if hash2d(world_x, world_z + 20.0, seed) > 0.92 {
                        StructureType::TreeOak
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            
            // Scale based on structure type
            let base_scale = match structure_type {
                StructureType::TreePine | StructureType::TreeOak => 3.0,
                StructureType::RockSmall => 2.0,
                StructureType::RockLarge => 4.0,
                StructureType::RuinPillar | StructureType::RuinWall => 3.5,
            };
            let scale = base_scale * (0.8 + hash2d(world_x * 3.0, world_z * 3.0, seed) * 0.4);
            
            let rotation = hash2d(world_x * 2.0, world_z * 2.0, seed) * std::f32::consts::TAU;
            let position = Vec3::new(world_x, height, world_z);
            
            structures.push((structure_type, StructureInstance::new(position, rotation, scale)));
        }
    }
    
    structures
}

// ============================================================================
// Structure System
// ============================================================================

pub struct StructureSystem {
    model_manager: ModelManager,
    instance_buffer: wgpu::Buffer,
    instance_data: Vec<(StructureType, StructureInstance)>,
    render_pipeline: wgpu::RenderPipeline,
    pub enabled: bool,
}

impl StructureSystem {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let model_manager = ModelManager::new(device);
        
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Structure Instance Buffer"),
            size: (std::mem::size_of::<StructureInstance>() * 1024) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Structure Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/structure.wgsl").into(),
            ),
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Structure Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout],
            push_constant_ranges: &[],
        });
        
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
                    blend: None,
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
    
    pub fn update(&mut self, _queue: &wgpu::Queue, camera_x: f32, camera_z: f32, seed: f32) {
        self.instance_data = generate_structures(camera_x, camera_z, seed);
    }
    
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
            
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
            let mesh = self.model_manager.get(structure_type);
            
            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..mesh.index_count, 0, 0..instances.len() as u32);
        }
    }
}
