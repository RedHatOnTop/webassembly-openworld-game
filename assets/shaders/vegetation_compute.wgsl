// ============================================================================
// Vegetation Compute Shader
// ============================================================================
//
// GPU-driven vegetation placement using deterministic hashing.
// Placement is based on world position and seed - NOT time/frame/thread ID.
// This ensures multiplayer-safe synchronization.
//
// CRITICAL: Height function MUST match terrain.wgsl exactly!
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
// NOISE LIBRARY (Copied from terrain.wgsl - MUST match exactly!)
// ============================================================================

const F3: f32 = 0.3333333333;  // 1/3
const G3: f32 = 0.1666666667;  // 1/6

// 3D Gradient vectors (12 edges of a cube)
const GRAD3: array<vec3<f32>, 12> = array<vec3<f32>, 12>(
    vec3( 1.0,  1.0,  0.0), vec3(-1.0,  1.0,  0.0), vec3( 1.0, -1.0,  0.0), vec3(-1.0, -1.0,  0.0),
    vec3( 1.0,  0.0,  1.0), vec3(-1.0,  0.0,  1.0), vec3( 1.0,  0.0, -1.0), vec3(-1.0,  0.0, -1.0),
    vec3( 0.0,  1.0,  1.0), vec3( 0.0, -1.0,  1.0), vec3( 0.0,  1.0, -1.0), vec3( 0.0, -1.0, -1.0)
);

/// Integer hash function for gradient indexing.
fn hash_int(x: i32) -> u32 {
    var n = u32(x);
    n = (n ^ 61u) ^ (n >> 16u);
    n = n + (n << 3u);
    n = n ^ (n >> 4u);
    n = n * 0x27d4eb2du;
    n = n ^ (n >> 15u);
    return n;
}

/// 3D hash for simplex noise.
fn hash_3d(x: i32, y: i32, z: i32) -> u32 {
    return hash_int(x + i32(hash_int(y + i32(hash_int(z)))));
}

/// 3D Simplex Noise. Returns value in range [-1.0, 1.0].
fn simplex_noise_3d(p: vec3<f32>) -> f32 {
    let s = (p.x + p.y + p.z) * F3;
    let i = floor(p.x + s);
    let j = floor(p.y + s);
    let k = floor(p.z + s);
    
    let t = (i + j + k) * G3;
    let X0 = i - t;
    let Y0 = j - t;
    let Z0 = k - t;
    
    let x0 = p.x - X0;
    let y0 = p.y - Y0;
    let z0 = p.z - Z0;
    
    var i1: i32; var j1: i32; var k1: i32;
    var i2: i32; var j2: i32; var k2: i32;
    
    if (x0 >= y0) {
        if (y0 >= z0) {
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
        } else if (x0 >= z0) {
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1;
        } else {
            i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1;
        }
    } else {
        if (y0 < z0) {
            i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1;
        } else if (x0 < z0) {
            i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1;
        } else {
            i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
        }
    }
    
    let x1 = x0 - f32(i1) + G3;
    let y1 = y0 - f32(j1) + G3;
    let z1 = z0 - f32(k1) + G3;
    let x2 = x0 - f32(i2) + 2.0 * G3;
    let y2 = y0 - f32(j2) + 2.0 * G3;
    let z2 = z0 - f32(k2) + 2.0 * G3;
    let x3 = x0 - 1.0 + 3.0 * G3;
    let y3 = y0 - 1.0 + 3.0 * G3;
    let z3 = z0 - 1.0 + 3.0 * G3;
    
    let ii = i32(i) & 255;
    let jj = i32(j) & 255;
    let kk = i32(k) & 255;
    
    var n0: f32 = 0.0;
    var n1: f32 = 0.0;
    var n2: f32 = 0.0;
    var n3: f32 = 0.0;
    
    var t0 = 0.6 - x0*x0 - y0*y0 - z0*z0;
    if (t0 > 0.0) {
        t0 *= t0;
        let gi0 = hash_3d(ii, jj, kk) % 12u;
        n0 = t0 * t0 * dot(GRAD3[gi0], vec3(x0, y0, z0));
    }
    
    var t1 = 0.6 - x1*x1 - y1*y1 - z1*z1;
    if (t1 > 0.0) {
        t1 *= t1;
        let gi1 = hash_3d(ii + i1, jj + j1, kk + k1) % 12u;
        n1 = t1 * t1 * dot(GRAD3[gi1], vec3(x1, y1, z1));
    }
    
    var t2 = 0.6 - x2*x2 - y2*y2 - z2*z2;
    if (t2 > 0.0) {
        t2 *= t2;
        let gi2 = hash_3d(ii + i2, jj + j2, kk + k2) % 12u;
        n2 = t2 * t2 * dot(GRAD3[gi2], vec3(x2, y2, z2));
    }
    
    var t3 = 0.6 - x3*x3 - y3*y3 - z3*z3;
    if (t3 > 0.0) {
        t3 *= t3;
        let gi3 = hash_3d(ii + 1, jj + 1, kk + 1) % 12u;
        n3 = t3 * t3 * dot(GRAD3[gi3], vec3(x3, y3, z3));
    }
    
    return 32.0 * (n0 + n1 + n2 + n3);
}

/// Fractal Brownian Motion using 3D Simplex Noise.
fn fbm_3d(p: vec3<f32>, octaves: i32) -> f32 {
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var frequency: f32 = 1.0;
    var max_value: f32 = 0.0;
    
    for (var i = 0; i < octaves; i++) {
        value += amplitude * simplex_noise_3d(p * frequency);
        max_value += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value / max_value;
}

/// Multi-octave ridge noise for detailed mountain ranges.
fn ridged_fbm_3d(p: vec3<f32>, octaves: i32) -> f32 {
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var frequency: f32 = 1.0;
    var max_value: f32 = 0.0;
    var weight: f32 = 1.0;
    
    for (var i = 0; i < octaves; i++) {
        let n = (1.0 - abs(simplex_noise_3d(p * frequency))) * weight;
        value += n * amplitude;
        max_value += amplitude;
        weight = clamp(n * 2.0, 0.0, 1.0);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value / max_value;
}

/// Apply domain warping for organic terrain shapes.
fn domain_warp_3d(p: vec3<f32>, amplitude: f32, frequency: f32) -> vec3<f32> {
    let warp = vec3<f32>(
        simplex_noise_3d(p * frequency + vec3(0.0, 0.0, 0.0)),
        simplex_noise_3d(p * frequency + vec3(5.2, 1.3, 0.0)),
        simplex_noise_3d(p * frequency + vec3(0.0, 0.0, 9.4))
    );
    return p + warp * amplitude;
}

// ============================================================================
// Continentalness Spline (MUST match terrain.wgsl!)
// ============================================================================

const CONTINENTALNESS_SPLINE: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0),   // Deep ocean floor
    vec2(-0.5, -0.8),   // Deep ocean
    vec2(-0.2, -0.3),   // Continental shelf
    vec2( 0.0,  0.1),   // Coastline
    vec2( 0.4,  0.5),   // Inland plateau
    vec2( 1.0,  1.0)    // Mountain peaks
);

fn cubic_spline(t: f32, points: array<vec2<f32>, 6>) -> f32 {
    let clamped_t = clamp(t, -1.0, 1.0);
    
    if (clamped_t <= points[0].x) { return points[0].y; }
    if (clamped_t >= points[5].x) { return points[5].y; }
    
    if (clamped_t < points[1].x) {
        let seg_t = (clamped_t - points[0].x) / (points[1].x - points[0].x);
        return mix(points[0].y, points[1].y, seg_t);
    }
    if (clamped_t < points[2].x) {
        let seg_t = (clamped_t - points[1].x) / (points[2].x - points[1].x);
        return mix(points[1].y, points[2].y, seg_t);
    }
    if (clamped_t < points[3].x) {
        let seg_t = (clamped_t - points[2].x) / (points[3].x - points[2].x);
        return mix(points[2].y, points[3].y, seg_t);
    }
    if (clamped_t < points[4].x) {
        let seg_t = (clamped_t - points[3].x) / (points[4].x - points[3].x);
        return mix(points[3].y, points[4].y, seg_t);
    }
    
    let seg_t = (clamped_t - points[4].x) / (points[5].x - points[4].x);
    return mix(points[4].y, points[5].y, seg_t);
}

// ============================================================================
// Climate Model (MUST match terrain.wgsl!)
// ============================================================================

struct ClimateChannels {
    continentalness: f32,
    erosion: f32,
    peaks_valleys: f32,
}

fn get_climate_channels(world_pos: vec3<f32>, seed: f32) -> ClimateChannels {
    var result: ClimateChannels;
    
    let seed_offset = vec3(seed * 1000.0, 0.0, seed * 500.0);
    let warped_pos = domain_warp_3d(world_pos + seed_offset, 20.0, 0.01);
    
    // Continentalness
    let cont_raw = fbm_3d(warped_pos * 0.005, 4);
    result.continentalness = cubic_spline(cont_raw, CONTINENTALNESS_SPLINE);
    
    // Erosion
    let erosion_pos = world_pos + vec3(seed * 2000.0 + 1000.0, 0.0, seed * 1000.0);
    result.erosion = fbm_3d(erosion_pos * 0.01, 3);
    
    // Peaks & Valleys
    let peaks_pos = world_pos + vec3(seed * 1500.0, 0.0, seed * 750.0);
    result.peaks_valleys = ridged_fbm_3d(peaks_pos * 0.008, 4);
    
    return result;
}

/// Get terrain height - EXACT COPY from terrain.wgsl
fn get_terrain_height(world_x: f32, world_z: f32, seed: f32) -> f32 {
    let pos_3d = vec3(world_x, 0.0, world_z);
    let climate = get_climate_channels(pos_3d, seed);
    
    let sea_level = 0.0;
    
    // Base Height from Continentalness
    var base_height: f32;
    if (climate.continentalness < -0.2) {
        let ocean_t = (climate.continentalness + 1.0) / 0.8;
        base_height = sea_level - 15.0 + ocean_t * 10.0;
    } else if (climate.continentalness < 0.2) {
        let coast_t = (climate.continentalness + 0.2) / 0.4;
        base_height = sea_level - 5.0 + coast_t * 10.0;
    } else {
        let land_t = (climate.continentalness - 0.2) / 0.8;
        base_height = sea_level + 5.0 + land_t * 35.0;
    }
    
    // Ocean Mask
    let ocean_mask = smoothstep(-0.1, 0.1, climate.continentalness);
    
    // Apply Peaks/Valleys (masked by ocean)
    let mountain_strength = smoothstep(0.0, 0.5, climate.continentalness);
    let peak_height = climate.peaks_valleys * 25.0 * mountain_strength * ocean_mask;
    
    // Erosion Effect
    let erosion_factor = 1.0 - (climate.erosion * 0.5 + 0.5) * 0.3;
    
    // Detail Noise (also masked by ocean)
    let detail_pos = vec3(world_x * 0.1, 0.0, world_z * 0.1);
    let detail_strength = smoothstep(-0.1, 0.3, climate.continentalness) * 3.0 * ocean_mask;
    let detail = simplex_noise_3d(detail_pos + vec3(seed * 100.0)) * detail_strength;
    
    // Combine all contributions
    var height = base_height + peak_height * erosion_factor + detail;
    
    // Ocean floor ripples
    if (climate.continentalness < -0.2) {
        let ocean_floor_noise = simplex_noise_3d(vec3(world_x * 0.03, 0.0, world_z * 0.03)) * 1.0;
        height += ocean_floor_noise;
    }
    
    return height;
}

/// Get terrain normal for slope detection
fn get_terrain_normal(world_x: f32, world_z: f32, seed: f32) -> vec3<f32> {
    let delta = 0.5;
    let h0 = get_terrain_height(world_x, world_z, seed);
    let hx = get_terrain_height(world_x + delta, world_z, seed);
    let hz = get_terrain_height(world_x, world_z + delta, seed);
    
    let tangent_x = vec3(delta, hx - h0, 0.0);
    let tangent_z = vec3(0.0, hz - h0, delta);
    
    return normalize(cross(tangent_z, tangent_x));
}

// ============================================================================
// Deterministic Hash (For jitter and density)
// ============================================================================

fn hash2d(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash2d_vec2(p: vec2<f32>) -> vec2<f32> {
    let p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    let p4 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// ============================================================================
// Main Compute Shader
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs_place_vegetation(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_x = gid.x;
    let grid_z = gid.y;
    
    // Camera-centered grid
    let camera_pos = vec2(uniforms.camera_x, uniforms.camera_z);
    let half_grid = 32.0;
    let grid_pos = vec2(f32(grid_x) - half_grid, f32(grid_z) - half_grid) * uniforms.spacing;
    
    // === ADD JITTER: Random offset to break grid pattern ===
    let jitter_hash = hash2d_vec2(grid_pos * 0.1 + vec2(uniforms.seed));
    let jitter = (jitter_hash - 0.5) * uniforms.spacing * 1.8;  // Scatter within cell
    
    let world_pos = camera_pos + grid_pos + jitter;
    let world_x = world_pos.x;
    let world_z = world_pos.y;
    let seed = uniforms.seed;
    
    // === Density check (hash-based) ===
    let density_hash = hash2d(world_pos * 0.3 + vec2(seed * 0.01));
    if (density_hash > uniforms.density) {
        return;
    }
    
    // === Sample terrain (EXACT same function as terrain.wgsl) ===
    let height = get_terrain_height(world_x, world_z, seed);
    let normal = get_terrain_normal(world_x, world_z, seed);
    
    // === Placement Rules ===
    
    // Rule 1: No underwater vegetation
    if (height < 2.0) {
        return;
    }
    
    // Rule 2: No vegetation on steep slopes (> ~45 degrees)
    let slope = 1.0 - normal.y;
    if (slope > 0.45) {
        return;
    }
    
    // Rule 3: No vegetation in snow zones
    if (height > 85.0) {
        return;
    }
    
    // === Generate instance data ===
    let rand2 = hash2d_vec2(world_pos + vec2(seed * 7.0));
    
    // Atomic append
    let index = atomicAdd(&counter, 1u);
    if (index >= MAX_INSTANCES) {
        return;
    }
    
    // Write instance
    instances[index].position = vec3(world_x, height, world_z);
    instances[index].scale = 0.8 + rand2.x * 0.5;  // 0.8 to 1.3 scale
    instances[index].rotation = rand2.y * PI * 2.0;
    instances[index].type_id = 0u;
    instances[index]._padding = vec2(0.0);
}
