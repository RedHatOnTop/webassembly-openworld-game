// Terrain Shader - Aether Engine
//
// GPU-driven terrain mesh system using Compute -> Storage -> Render pipeline.
// Generates a solid terrain mesh with proper normals and directional lighting.
// Uses triplanar texture mapping for realistic terrain without UV stretching.
//
// Compute Shaders:
//   - cs_generate_indices: One-time index buffer generation (6 indices per quad)
//   - cs_update_terrain: Per-frame position and normal updates
//
// Reference: docs/04_TASKS/task_06_terrain_mesh.md, task_08_triplanar_texture.md

// ============================================================================
// Shared Data Structures
// ============================================================================

/// Terrain vertex data structure.
/// Contains position and normal for lighting calculations.
struct Vertex {
    position: vec4<f32>,  // xyz = world position, w = 1.0
    normal: vec4<f32>,    // xyz = normal vector, w = 0.0
}

/// Uniforms for terrain compute/render shaders.
struct TerrainUniforms {
    time: f32,
    grid_size: u32,
    chunk_offset_x: f32,
    chunk_offset_z: f32,
}

/// Camera uniform for rendering.
struct CameraUniform {
    view_proj: mat4x4<f32>,
}

// ============================================================================
// Compute Shader Bindings (Index Generation)
// ============================================================================

@group(0) @binding(0)
var<storage, read_write> vertices: array<Vertex>;

@group(0) @binding(1)
var<storage, read_write> indices: array<u32>;

@group(1) @binding(0)
var<uniform> uniforms: TerrainUniforms;

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate terrain height at a given world position.
/// Uses sine-cosine wave for dynamic terrain animation.
fn get_height(x: f32, z: f32, time: f32) -> f32 {
    let wave_freq = 0.15;
    let wave_amp = 2.0;
    return sin(x * wave_freq + time) * cos(z * wave_freq + time) * wave_amp;
}

// ============================================================================
// Compute Shader: Index Generation (One-time)
// ============================================================================

/// Generates triangle indices for the terrain mesh.
/// Each grid cell produces 2 triangles (6 indices) in CCW winding order.
/// Dispatch: ceil((GRID_SIZE-1)/8) x ceil((GRID_SIZE-1)/8) workgroups
@compute @workgroup_size(8, 8, 1)
fn cs_generate_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_size = uniforms.grid_size;
    let x = gid.x;
    let z = gid.y;
    
    // Only process cells (grid_size - 1) x (grid_size - 1)
    // Each cell creates a quad from 4 adjacent vertices
    if (x >= grid_size - 1u || z >= grid_size - 1u) {
        return;
    }
    
    // Calculate vertex indices for the four corners of this quad
    // Vertex layout in the grid:
    //   TL---TR
    //   |  /  |
    //   | /   |
    //   BL---BR
    let top_left = z * grid_size + x;
    let top_right = top_left + 1u;
    let bottom_left = (z + 1u) * grid_size + x;
    let bottom_right = bottom_left + 1u;
    
    // Calculate index buffer offset
    // Each quad produces 6 indices (2 triangles)
    let quad_index = z * (grid_size - 1u) + x;
    let base = quad_index * 6u;
    
    // Triangle 1: TL -> BL -> TR (CCW winding, facing up)
    indices[base + 0u] = top_left;
    indices[base + 1u] = bottom_left;
    indices[base + 2u] = top_right;
    
    // Triangle 2: TR -> BL -> BR (CCW winding, facing up)
    indices[base + 3u] = top_right;
    indices[base + 4u] = bottom_left;
    indices[base + 5u] = bottom_right;
}

// ============================================================================
// Compute Shader: Terrain Update (Per-frame)
// ============================================================================

/// Updates vertex positions and calculates normals using finite differences.
/// Dispatch: ceil(GRID_SIZE/8) x ceil(GRID_SIZE/8) workgroups
@compute @workgroup_size(8, 8, 1)
fn cs_update_terrain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_size = uniforms.grid_size;
    let x = gid.x;
    let z = gid.y;
    
    // Bounds check
    if (x >= grid_size || z >= grid_size) {
        return;
    }
    
    let index = z * grid_size + x;
    let time = uniforms.time;
    
    // Calculate world position (centered grid + chunk offset for infinite terrain)
    let half_size = f32(grid_size) * 0.5;
    let spacing = 0.5;  // World units between vertices
    
    // Local position within the grid
    let local_x = (f32(x) - half_size) * spacing;
    let local_z = (f32(z) - half_size) * spacing;
    
    // World position = local + chunk offset (infinite terrain scrolling)
    let world_x = local_x + uniforms.chunk_offset_x;
    let world_z = local_z + uniforms.chunk_offset_z;
    let world_y = get_height(world_x, world_z, time);
    
    // Calculate normal using finite differences
    // Sample neighboring heights for gradient calculation
    let delta = spacing;
    let height_px = get_height(world_x + delta, world_z, time);  // +X neighbor
    let height_pz = get_height(world_x, world_z + delta, time);  // +Z neighbor
    
    // Tangent vectors along X and Z axes
    let tangent_x = vec3<f32>(delta, height_px - world_y, 0.0);
    let tangent_z = vec3<f32>(0.0, height_pz - world_y, delta);
    
    // Normal = cross(tangent_z, tangent_x) for CCW winding order
    // This produces an upward-facing normal for a flat surface
    let normal = normalize(cross(tangent_z, tangent_x));
    
    // Store vertex data (using local position for mesh, world for height/normal)
    vertices[index].position = vec4<f32>(local_x, world_y, local_z, 1.0);
    vertices[index].normal = vec4<f32>(normal, 0.0);
}

// ============================================================================
// Render Shader Bindings
// ============================================================================

@group(0) @binding(0)
var<storage, read> vertices_read: array<Vertex>;

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

// Texture array: Layer 0 = Grass, Layer 1 = Rock, Layer 2 = Snow
@group(2) @binding(0)
var terrain_textures: texture_2d_array<f32>;

@group(2) @binding(1)
var terrain_sampler: sampler;

// ============================================================================
// Texture Layer Indices
// ============================================================================

const LAYER_GRASS: i32 = 0;
const LAYER_ROCK: i32 = 1;
const LAYER_SNOW: i32 = 2;

// Texture scale (world units per texture repeat)
const TEXTURE_SCALE: f32 = 0.2;

// ============================================================================
// Vertex Shader
// ============================================================================

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
}

/// Vertex shader reads from storage buffer using vertex_index.
/// Passes world-space position and normal to fragment shader for lighting.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let vertex = vertices_read[vertex_index];
    
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vertex.position;
    out.world_normal = vertex.normal.xyz;
    out.world_position = vertex.position.xyz;
    return out;
}

// ============================================================================
// Fragment Shader with Triplanar Texture Mapping
// ============================================================================

/// Samples a texture layer at the given UV coordinates.
fn sample_texture_layer(uv: vec2<f32>, layer: i32) -> vec4<f32> {
    return textureSample(terrain_textures, terrain_sampler, uv, layer);
}

/// Triplanar texture mapping function.
/// Projects textures from 3 orthogonal axes and blends based on normal direction.
/// This eliminates UV stretching on steep slopes.
fn triplanar_sample(world_pos: vec3<f32>, normal: vec3<f32>, layer: i32) -> vec4<f32> {
    // Calculate blending weights based on absolute normal components
    let blend_weights = abs(normal);
    // Normalize weights so they sum to 1
    let weights = blend_weights / (blend_weights.x + blend_weights.y + blend_weights.z);
    
    // Scale world position for texture coordinates
    let scaled_pos = world_pos * TEXTURE_SCALE;
    
    // Sample texture from each projection axis:
    // X-axis projection: use YZ plane
    let x_proj = sample_texture_layer(scaled_pos.yz, layer);
    // Y-axis projection: use XZ plane (top-down view)
    let y_proj = sample_texture_layer(scaled_pos.xz, layer);
    // Z-axis projection: use XY plane
    let z_proj = sample_texture_layer(scaled_pos.xy, layer);
    
    // Blend projections based on normal direction
    return x_proj * weights.x + y_proj * weights.y + z_proj * weights.z;
}

/// Samples terrain texture with slope-based blending.
/// Flat surfaces get grass, slopes get rock, high altitude gets snow.
fn sample_terrain_texture(world_pos: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    // Slope factor: 1.0 = flat (facing up), 0.0 = vertical wall
    let slope = normal.y;
    
    // Height for snow blending
    let height = world_pos.y;
    
    // Sample textures using triplanar mapping
    let grass = triplanar_sample(world_pos, normal, LAYER_GRASS);
    let rock = triplanar_sample(world_pos, normal, LAYER_ROCK);
    let snow = triplanar_sample(world_pos, normal, LAYER_SNOW);
    
    // Slope-based grass/rock blend
    // steep_factor: 0 = flat (grass), 1 = steep (rock)
    let steep_threshold = 0.7;
    let steep_factor = 1.0 - smoothstep(steep_threshold - 0.2, steep_threshold + 0.1, slope);
    var base_color = mix(grass, rock, steep_factor);
    
    // Height-based snow blend (above y=1.5)
    let snow_start = 1.0;
    let snow_full = 2.0;
    let snow_factor = smoothstep(snow_start, snow_full, height) * slope;
    base_color = mix(base_color, snow, snow_factor);
    
    return base_color;
}

/// Fragment shader with triplanar textured terrain and Lambertian lighting.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalize interpolated normal (may be denormalized after interpolation)
    let normal = normalize(in.world_normal);
    
    // Sample terrain texture using triplanar mapping
    let texture_color = sample_terrain_texture(in.world_position, normal);
    
    // Directional light (pointing towards light source, upper-right-front)
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    
    // Ambient light (minimum illumination)
    let ambient = 0.3;
    
    // Diffuse lighting (Lambertian reflectance)
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let diffuse = n_dot_l * 0.7;
    
    // Final color with lighting
    let lighting = ambient + diffuse;
    let final_color = texture_color.rgb * lighting;
    
    return vec4<f32>(final_color, 1.0);
}
