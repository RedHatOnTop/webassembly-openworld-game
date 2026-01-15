// ============================================================================
// Vegetation Compute Shader
// ============================================================================
//
// GPU-driven vegetation placement using deterministic hashing.
// Placement is based on world position and seed - NOT time/frame/thread ID.
// This ensures multiplayer-safe synchronization.
//
// Reference: docs/04_TASKS/task_12_vegetation.md

// ============================================================================
// Data Structures
// ============================================================================

struct GrassInstance {
    position: vec3<f32>,
    scale: f32,
    rotation: f32,
    type_id: u32,
    _padding: vec2<f32>,
}

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct VegetationUniforms {
    camera_x: f32,
    camera_y: f32,
    camera_z: f32,
    seed: f32,
    time: f32,
    density: f32,
    spacing: f32,
    _padding: f32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> instances: array<GrassInstance>;
@group(0) @binding(1) var<storage, read_write> indirect: DrawIndexedIndirectArgs;
@group(0) @binding(2) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(3) var<uniform> uniforms: VegetationUniforms;

// ============================================================================
// Constants
// ============================================================================

const MAX_INSTANCES: u32 = 100000u;
const PI: f32 = 3.14159265359;

// ============================================================================
// Deterministic Hash Functions (NO TIME DEPENDENCY)
// ============================================================================

/// 2D hash function for deterministic randomness.
/// Input: world position. Output: 0.0-1.0 pseudo-random value.
fn hash2d(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

/// 2D hash returning vec2 for multiple random values.
fn hash2d_vec2(p: vec2<f32>) -> vec2<f32> {
    let p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    let p4 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// ============================================================================
// Noise Functions (Duplicated from terrain for standalone compute)
// ============================================================================

fn permute(x: vec4<f32>) -> vec4<f32> {
    return ((x * 34.0 + 1.0) * x) % 289.0;
}

fn taylor_inv_sqrt(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn simplex_noise_2d(v: vec2<f32>) -> f32 {
    let C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    
    var i = floor(v + dot(v, C.yy));
    let x0 = v - i + dot(i, C.xx);
    
    var i1: vec2<f32>;
    if (x0.x > x0.y) {
        i1 = vec2(1.0, 0.0);
    } else {
        i1 = vec2(0.0, 1.0);
    }
    
    var x12 = x0.xyxy + C.xxzz;
    x12 = vec4(x12.xy - i1, x12.zw);
    
    i = i % 289.0;
    let p = permute(permute(i.y + vec4(0.0, i1.y, 1.0, 0.0)) + i.x + vec4(0.0, i1.x, 1.0, 0.0));
    
    var m = max(0.5 - vec4(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw), 0.0), vec4(0.0));
    m = m * m;
    m = m * m;
    
    let x = 2.0 * fract(p * C.wwww) - 1.0;
    let h = abs(x) - 0.5;
    let ox = floor(x + 0.5);
    let a0 = x - ox;
    
    m = m * (1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h));
    
    let g = vec3(
        a0.x * x0.x + h.x * x0.y,
        a0.yz * x12.xz + h.yz * x12.yw
    );
    
    return 130.0 * dot(m.xyz, g);
}

/// Sample terrain height at world position (simplified version).
fn get_terrain_height(world_x: f32, world_z: f32, seed: f32) -> f32 {
    let scale = 0.005;
    let pos = vec2(world_x * scale + seed * 0.1, world_z * scale);
    
    // Continental noise
    let continental = simplex_noise_2d(pos * 0.3) * 0.6 + 
                      simplex_noise_2d(pos * 0.7) * 0.3 +
                      simplex_noise_2d(pos * 1.5) * 0.1;
    
    // Height mapping
    var height: f32;
    if (continental < -0.2) {
        // Ocean
        height = -15.0 + continental * 10.0;
    } else if (continental < 0.2) {
        // Coast
        let t = (continental + 0.2) / 0.4;
        height = -5.0 + t * 15.0;
    } else {
        // Land
        let t = (continental - 0.2) / 0.8;
        height = 10.0 + t * 50.0;
    }
    
    // Detail noise
    height += simplex_noise_2d(pos * 5.0) * 3.0;
    
    return height;
}

/// Sample terrain normal at world position.
fn get_terrain_normal(world_x: f32, world_z: f32, seed: f32) -> vec3<f32> {
    let delta = 1.0;
    let h0 = get_terrain_height(world_x, world_z, seed);
    let hx = get_terrain_height(world_x + delta, world_z, seed);
    let hz = get_terrain_height(world_x, world_z + delta, seed);
    
    let tangent_x = vec3(delta, hx - h0, 0.0);
    let tangent_z = vec3(0.0, hz - h0, delta);
    
    return normalize(cross(tangent_z, tangent_x));
}

// ============================================================================
// Main Compute Shader
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs_place_vegetation(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_x = gid.x;
    let grid_z = gid.y;
    
    // Calculate world position (centered around camera)
    let camera_pos = vec2(uniforms.camera_x, uniforms.camera_z);
    let grid_offset = vec2(f32(grid_x) - 32.0, f32(grid_z) - 32.0) * uniforms.spacing;
    let world_pos = camera_pos + grid_offset;
    
    let world_x = world_pos.x;
    let world_z = world_pos.y;
    let seed = uniforms.seed;
    
    // === Deterministic placement check ===
    // Hash based on world position (NOT time or thread ID)
    let density_hash = hash2d(world_pos * 0.5 + vec2(seed * 0.01));
    
    // Skip if below density threshold
    if (density_hash > uniforms.density) {
        return;
    }
    
    // === Terrain sampling ===
    let height = get_terrain_height(world_x, world_z, seed);
    let normal = get_terrain_normal(world_x, world_z, seed);
    
    // === Placement rules ===
    
    // Rule 1: No underwater vegetation
    if (height < 1.0) {
        return;
    }
    
    // Rule 2: No vegetation on steep slopes (> ~50 degrees)
    let slope = 1.0 - normal.y;  // 0=flat, 1=vertical
    if (slope > 0.5) {
        return;
    }
    
    // Rule 3: No vegetation in snow zones
    if (height > 90.0) {
        return;
    }
    
    // === Generate instance data (deterministic) ===
    let rand2 = hash2d_vec2(world_pos + vec2(seed));
    
    // Atomic append to instance buffer
    let index = atomicAdd(&counter, 1u);
    
    // Bounds check
    if (index >= MAX_INSTANCES) {
        return;
    }
    
    // Write instance data
    instances[index].position = vec3(world_x, height, world_z);
    instances[index].scale = 0.6 + rand2.x * 0.6;  // 0.6 to 1.2 scale
    instances[index].rotation = rand2.y * PI * 2.0;  // Random Y rotation
    instances[index].type_id = 0u;  // Grass type
    instances[index]._padding = vec2(0.0);
    
    // Note: instance_count updated via atomic counter, read by CPU
    // The indirect buffer's instance_count is set from counter after dispatch
}
