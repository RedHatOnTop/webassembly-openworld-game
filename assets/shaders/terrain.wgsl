// Terrain Shader - Aether Engine
//
// GPU-driven terrain mesh system using Compute -> Storage -> Render pipeline.
// Generates a solid terrain mesh with proper normals and directional lighting.
//
// Compute Shaders:
//   - cs_generate_indices: One-time index buffer generation (6 indices per quad)
//   - cs_update_terrain: Per-frame position and normal updates
//
// Reference: docs/04_TASKS/task_06_terrain_mesh.md

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
    _pad0: u32,
    _pad1: u32,
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
    
    // Calculate world position (centered grid)
    let half_size = f32(grid_size) * 0.5;
    let spacing = 0.5;  // World units between vertices
    
    let world_x = (f32(x) - half_size) * spacing;
    let world_z = (f32(z) - half_size) * spacing;
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
    
    // Store vertex data
    vertices[index].position = vec4<f32>(world_x, world_y, world_z, 1.0);
    vertices[index].normal = vec4<f32>(normal, 0.0);
}

// ============================================================================
// Render Shader Bindings
// ============================================================================

@group(0) @binding(0)
var<storage, read> vertices_read: array<Vertex>;

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

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
// Fragment Shader with Directional Lighting
// ============================================================================

/// Fragment shader applies Lambertian diffuse lighting.
/// Creates a terrain color gradient based on height.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalize interpolated normal (may be denormalized after interpolation)
    let normal = normalize(in.world_normal);
    
    // Directional light (pointing towards light source, upper-right-front)
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    
    // Ambient light (minimum illumination)
    let ambient = 0.25;
    
    // Diffuse lighting (Lambertian reflectance)
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let diffuse = n_dot_l * 0.75;
    
    // Height-based terrain color gradient
    let height = in.world_position.y;
    
    // Color palette
    let deep_color = vec3<f32>(0.15, 0.25, 0.35);   // Dark blue-gray (valleys)
    let mid_color = vec3<f32>(0.25, 0.45, 0.20);    // Green (mid-level)
    let high_color = vec3<f32>(0.50, 0.45, 0.35);   // Brown-tan (peaks)
    
    // Blend colors based on height [-2, 2] range
    let t = clamp((height + 2.0) / 4.0, 0.0, 1.0);
    var base_color: vec3<f32>;
    if (t < 0.5) {
        base_color = mix(deep_color, mid_color, t * 2.0);
    } else {
        base_color = mix(mid_color, high_color, (t - 0.5) * 2.0);
    }
    
    // Final color with lighting
    let lighting = ambient + diffuse;
    let final_color = base_color * lighting;
    
    return vec4<f32>(final_color, 1.0);
}
