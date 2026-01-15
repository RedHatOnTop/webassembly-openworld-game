// ============================================================================
// Terrain Shader - Aether Engine
// ============================================================================
//
// GPU-driven terrain mesh system using Compute -> Storage -> Render pipeline.
// Implements 5-Channel Climate Model for procedural continental terrain.
// Uses Dual Contouring-ready density functions with triplanar texture mapping.
//
// Compute Shaders:
//   - cs_generate_indices: One-time index buffer generation (6 indices per quad)
//   - cs_update_terrain: Per-frame position and normal updates with climate model
//
// Reference: 
//   - docs/02_ARCHITECTURE/world_gen_strategy.md
//   - docs/04_TASKS/task_09_world_gen_macro.md
// ============================================================================


// ============================================================================
// NOISE LIBRARY (Embedded - WGSL doesn't support imports yet)
// ============================================================================

// Constants and Lookup Tables
const F3: f32 = 0.3333333333;  // 1/3
const G3: f32 = 0.1666666667;  // 1/6
const F4: f32 = 0.309016994;   // (sqrt(5) - 1) / 4
const G4: f32 = 0.138196601;   // (5 - sqrt(5)) / 20

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

/// Ridge Noise for sharp mountain peaks.
fn ridge_noise_3d(p: vec3<f32>) -> f32 {
    return 1.0 - abs(simplex_noise_3d(p));
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
// Continentalness Spline (Critical for World Shape)
// ============================================================================

// Control points for continentalness remapping
// Format: vec2(input_noise_value, output_continental_value)
// Creates distinct ocean/land boundaries with sharp coastlines
const CONTINENTALNESS_SPLINE: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0),   // Deep ocean floor (very deep)
    vec2(-0.5, -0.8),   // Deep ocean (flat floor)
    vec2(-0.2, -0.3),   // Continental shelf (gradual rise)
    vec2( 0.0,  0.1),   // Coastline (sharp transition)
    vec2( 0.4,  0.5),   // Inland plateau
    vec2( 1.0,  1.0)    // Mountain peaks
);

/// Linear spline interpolation for continentalness remapping.
/// Uses simple linear interpolation between control points.
fn cubic_spline(t: f32, points: array<vec2<f32>, 6>) -> f32 {
    let clamped_t = clamp(t, -1.0, 1.0);
    
    // Early exit for boundaries
    if (clamped_t <= points[0].x) { return points[0].y; }
    if (clamped_t >= points[5].x) { return points[5].y; }
    
    // Find segment and interpolate - unrolled loop to avoid issues
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
    
    // Last segment
    let seg_t = (clamped_t - points[4].x) / (points[5].x - points[4].x);
    return mix(points[4].y, points[5].y, seg_t);
}


// ============================================================================
// Shared Data Structures
// ============================================================================

/// Terrain vertex data structure.
struct Vertex {
    position: vec4<f32>,  // xyz = world position, w = 1.0
    normal: vec4<f32>,    // xyz = normal vector, w = 0.0
}

/// Per-chunk uniforms (used by compute shader).
struct ChunkUniforms {
    offset_x: f32,     // Chunk world offset X
    offset_z: f32,     // Chunk world offset Z
    grid_size: u32,    // Grid dimension
    _padding: u32,
}

/// Global uniforms (used by both compute and render).
struct GlobalUniforms {
    time: f32,
    seed: f32,
    debug_mode: u32,
    _padding0: u32,
    // Camera position for fog calculation
    camera_x: f32,
    camera_y: f32,
    camera_z: f32,
    _padding1: u32,
}

/// Camera uniform for rendering.
struct CameraUniform {
    view_proj: mat4x4<f32>,
}

/// Climate channel data for terrain generation.
struct ClimateChannels {
    continentalness: f32,  // -1.0 (deep ocean) to 1.0 (high peaks)
    erosion: f32,          // -1.0 (jagged) to 1.0 (flat/eroded)
    peaks_valleys: f32,    // Ridge noise for mountain spines
}


// ============================================================================
// Compute Shader Bindings
// ============================================================================

// Bind Group 0: Per-chunk resources
@group(0) @binding(0)
var<storage, read_write> vertices: array<Vertex>;

@group(0) @binding(1)
var<uniform> chunk_uniforms: ChunkUniforms;

// Bind Group 1: Global resources
@group(1) @binding(0)
var<storage, read_write> indices: array<u32>;

@group(1) @binding(1)
var<uniform> global_uniforms: GlobalUniforms;


// ============================================================================
// Climate Model Functions
// ============================================================================

/// Sample all climate channels at a world position.
fn get_climate_channels(world_pos: vec3<f32>, seed: f32) -> ClimateChannels {
    var result: ClimateChannels;
    
    // Offset positions by seed to create different worlds
    let seed_offset = vec3(seed * 1000.0, 0.0, seed * 500.0);
    
    // Apply domain warping for more organic shapes
    let warped_pos = domain_warp_3d(world_pos + seed_offset, 20.0, 0.01);
    
    // === Continentalness === (Visible within 100x100 grid)
    // Frequency 0.005 = features every ~200 units, visible in our grid
    let cont_raw = fbm_3d(warped_pos * 0.005, 4);
    // Remap through spline for distinct continental edges
    result.continentalness = cubic_spline(cont_raw, CONTINENTALNESS_SPLINE);
    
    // === Erosion === (Medium frequency for terrain variation)
    let erosion_pos = world_pos + vec3(seed * 2000.0 + 1000.0, 0.0, seed * 1000.0);
    result.erosion = fbm_3d(erosion_pos * 0.01, 3);
    
    // === Peaks & Valleys === (Ridge noise for hills and valleys)
    let peaks_pos = world_pos + vec3(seed * 1500.0, 0.0, seed * 750.0);
    result.peaks_valleys = ridged_fbm_3d(peaks_pos * 0.008, 4);
    
    return result;
}

/// Calculate terrain height using the climate model.
fn get_terrain_height(world_x: f32, world_z: f32, seed: f32) -> f32 {
    let pos_3d = vec3(world_x, 0.0, world_z);
    let climate = get_climate_channels(pos_3d, seed);
    
    // Sea level reference (0.0 = sea level)
    let sea_level = 0.0;
    
    // === Base Height from Continentalness ===
    // Scaled for 100x100 grid visibility
    // -1.0 (Ocean) -> -15m, 0.0 (Coast) -> +2m, +1.0 (Peaks) -> +40m
    var base_height: f32;
    if (climate.continentalness < -0.2) {
        // Ocean floor: relatively flat
        let ocean_t = (climate.continentalness + 1.0) / 0.8;
        base_height = sea_level - 15.0 + ocean_t * 10.0;
    } else if (climate.continentalness < 0.2) {
        // Coast: transition zone
        let coast_t = (climate.continentalness + 0.2) / 0.4;
        base_height = sea_level - 5.0 + coast_t * 10.0;
    } else {
        // Land: gradual rise
        let land_t = (climate.continentalness - 0.2) / 0.8;
        base_height = sea_level + 5.0 + land_t * 35.0;
    }
    
    // === Apply Peaks/Valleys on Land Only ===
    let mountain_mask = smoothstep(0.0, 0.5, climate.continentalness);
    let peak_height = climate.peaks_valleys * 25.0 * mountain_mask;
    
    // === Erosion Effect ===
    let erosion_factor = 1.0 - (climate.erosion * 0.5 + 0.5) * 0.3;
    
    // === Detail Noise ===
    let detail_pos = vec3(world_x * 0.1, 0.0, world_z * 0.1);
    let detail_strength = smoothstep(-0.1, 0.3, climate.continentalness) * 3.0;
    let detail = simplex_noise_3d(detail_pos + vec3(seed * 100.0)) * detail_strength;
    
    // Combine all contributions
    var height = base_height + peak_height * erosion_factor + detail;
    
    // Ocean floor ripples
    if (climate.continentalness < -0.2) {
        let ocean_floor_noise = simplex_noise_3d(vec3(world_x * 0.05, 0.0, world_z * 0.05)) * 1.5;
        height += ocean_floor_noise;
    }
    
    return height;
}


// ============================================================================
// Compute Shader: Index Generation (One-time)
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs_generate_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_size = chunk_uniforms.grid_size;
    let x = gid.x;
    let z = gid.y;
    
    if (x >= grid_size - 1u || z >= grid_size - 1u) {
        return;
    }
    
    let top_left = z * grid_size + x;
    let top_right = top_left + 1u;
    let bottom_left = (z + 1u) * grid_size + x;
    let bottom_right = bottom_left + 1u;
    
    let quad_index = z * (grid_size - 1u) + x;
    let base = quad_index * 6u;
    
    // Triangle 1: TL -> BL -> TR (CCW winding)
    indices[base + 0u] = top_left;
    indices[base + 1u] = bottom_left;
    indices[base + 2u] = top_right;
    
    // Triangle 2: TR -> BL -> BR (CCW winding)
    indices[base + 3u] = top_right;
    indices[base + 4u] = bottom_left;
    indices[base + 5u] = bottom_right;
}


// ============================================================================
// Compute Shader: Terrain Update (Per-frame)
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs_update_terrain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_size = chunk_uniforms.grid_size;
    let x = gid.x;
    let z = gid.y;
    
    if (x >= grid_size || z >= grid_size) {
        return;
    }
    
    let index = z * grid_size + x;
    let seed = global_uniforms.seed;
    
    // Calculate world position
    let half_size = f32(grid_size) * 0.5;
    let spacing = 1.0;
    
    let local_x = (f32(x) - half_size) * spacing;
    let local_z = (f32(z) - half_size) * spacing;
    
    // Add chunk offset for world position
    let world_x = local_x + chunk_uniforms.offset_x;
    let world_z = local_z + chunk_uniforms.offset_z;
    
    // Sample terrain height using climate model
    let world_y = get_terrain_height(world_x, world_z, seed);
    
    // Calculate normal using finite differences
    let delta = spacing;
    let height_px = get_terrain_height(world_x + delta, world_z, seed);
    let height_pz = get_terrain_height(world_x, world_z + delta, seed);
    
    let tangent_x = vec3<f32>(delta, height_px - world_y, 0.0);
    let tangent_z = vec3<f32>(0.0, height_pz - world_y, delta);
    let normal = normalize(cross(tangent_z, tangent_x));
    
    // Store local position (chunk offset applied in shader)
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

// Texture array for triplanar mapping
@group(2) @binding(0)
var terrain_textures: texture_2d_array<f32>;

@group(2) @binding(1)
var terrain_sampler: sampler;

// Render uniforms (for debug mode and fog)
@group(3) @binding(0)
var<uniform> render_uniforms: GlobalUniforms;


// ============================================================================
// Debug Mode Constants
// ============================================================================

const DEBUG_MODE_NORMAL: u32 = 0u;
const DEBUG_MODE_CONTINENTALNESS: u32 = 1u;
const DEBUG_MODE_EROSION: u32 = 2u;
const DEBUG_MODE_PEAKS: u32 = 3u;


// ============================================================================
// Texture Constants
// ============================================================================

const LAYER_GRASS: i32 = 0;
const LAYER_ROCK: i32 = 1;
const LAYER_SNOW: i32 = 2;
const TEXTURE_SCALE: f32 = 0.1;


// ============================================================================
// Vertex Shader
// ============================================================================

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
}

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
// Fragment Shader Helper Functions
// ============================================================================

fn sample_texture_layer(uv: vec2<f32>, layer: i32) -> vec4<f32> {
    return textureSample(terrain_textures, terrain_sampler, uv, layer);
}

fn triplanar_sample(world_pos: vec3<f32>, normal: vec3<f32>, layer: i32) -> vec4<f32> {
    let blend_weights = abs(normal);
    let weights = blend_weights / (blend_weights.x + blend_weights.y + blend_weights.z);
    
    let scaled_pos = world_pos * TEXTURE_SCALE;
    
    let x_proj = sample_texture_layer(scaled_pos.yz, layer);
    let y_proj = sample_texture_layer(scaled_pos.xz, layer);
    let z_proj = sample_texture_layer(scaled_pos.xy, layer);
    
    return x_proj * weights.x + y_proj * weights.y + z_proj * weights.z;
}

fn sample_terrain_texture(world_pos: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    let slope = normal.y;
    let height = world_pos.y;
    
    let grass = triplanar_sample(world_pos, normal, LAYER_GRASS);
    let rock = triplanar_sample(world_pos, normal, LAYER_ROCK);
    let snow = triplanar_sample(world_pos, normal, LAYER_SNOW);
    
    // Slope-based grass/rock blend
    let steep_threshold = 0.7;
    let steep_factor = 1.0 - smoothstep(steep_threshold - 0.2, steep_threshold + 0.1, slope);
    var base_color = mix(grass, rock, steep_factor);
    
    // Height-based snow blend
    let snow_start = 50.0;
    let snow_full = 80.0;
    let snow_factor = smoothstep(snow_start, snow_full, height) * slope;
    base_color = mix(base_color, snow, snow_factor);
    
    return base_color;
}

/// Get debug color for climate visualization.
fn get_debug_color(climate: ClimateChannels, debug_mode: u32) -> vec4<f32> {
    switch (debug_mode) {
        case DEBUG_MODE_CONTINENTALNESS: {
            // Visualize continentalness: Blue=Ocean, Green=Coast, Brown=Land, White=Peaks
            let c = climate.continentalness * 0.5 + 0.5; // Normalize to 0-1
            
            if (c < 0.25) {
                // Deep ocean (dark blue to blue)
                let t = c / 0.25;
                return vec4(0.0, t * 0.3, 0.3 + t * 0.4, 1.0);
            } else if (c < 0.45) {
                // Ocean to coast (blue to cyan)
                let t = (c - 0.25) / 0.2;
                return vec4(0.0, 0.3 + t * 0.4, 0.7 - t * 0.2, 1.0);
            } else if (c < 0.55) {
                // Coastal zone (cyan to green)
                let t = (c - 0.45) / 0.1;
                return vec4(t * 0.2, 0.7, 0.5 - t * 0.3, 1.0);
            } else if (c < 0.75) {
                // Inland (green to brown)
                let t = (c - 0.55) / 0.2;
                return vec4(0.2 + t * 0.4, 0.7 - t * 0.3, 0.2 - t * 0.1, 1.0);
            } else {
                // Mountains to peaks (brown to white)
                let t = (c - 0.75) / 0.25;
                return vec4(0.6 + t * 0.4, 0.4 + t * 0.6, 0.1 + t * 0.9, 1.0);
            }
        }
        case DEBUG_MODE_EROSION: {
            // Visualize erosion: Red=Jagged, Green=Flat
            let e = climate.erosion * 0.5 + 0.5; // Normalize to 0-1
            return vec4(1.0 - e, e, 0.2, 1.0);
        }
        case DEBUG_MODE_PEAKS: {
            // Visualize peaks/valleys: White=Ridges, Black=Valleys
            let p = climate.peaks_valleys;
            return vec4(p, p, p, 1.0);
        }
        default: {
            return vec4(1.0, 0.0, 1.0, 1.0); // Magenta = error
        }
    }
}


// ============================================================================
// Fragment Shader
// ============================================================================

// Fog constants
const FOG_COLOR: vec3<f32> = vec3(0.7, 0.8, 0.9);  // Light blue-grey (matches sky)
const FOG_START: f32 = 150.0;   // Start fading at this distance
const FOG_END: f32 = 350.0;     // Fully fogged at this distance

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let debug_mode = render_uniforms.debug_mode;
    let seed = render_uniforms.seed;
    
    // Get camera position for fog
    let camera_pos = vec3(render_uniforms.camera_x, render_uniforms.camera_y, render_uniforms.camera_z);
    
    // Calculate distance from camera for fog
    let frag_pos = in.world_position;
    let distance = length(frag_pos - camera_pos);
    
    // Debug mode visualization
    if (debug_mode != DEBUG_MODE_NORMAL) {
        // Use stored position (already in world space with chunk offset)
        let world_pos = vec3(frag_pos.x, 0.0, frag_pos.z);
        let climate = get_climate_channels(world_pos, seed);
        
        var debug_color = get_debug_color(climate, debug_mode);
        
        // Apply lighting
        let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
        let ambient = 0.4;
        let diffuse = max(dot(normal, light_dir), 0.0) * 0.6;
        debug_color = vec4(debug_color.rgb * (ambient + diffuse), 1.0);
        
        // Apply fog
        let fog_factor = smoothstep(FOG_START, FOG_END, distance);
        let fogged_color = mix(debug_color.rgb, FOG_COLOR, fog_factor);
        
        return vec4(fogged_color, 1.0);
    }
    
    // === Normal Rendering with Water, Terrain, and Fog ===
    let height = in.world_position.y;
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    
    var terrain_color: vec3<f32>;
    
    // Check if below sea level (water visualization)
    if (height < 0.0) {
        let depth = abs(height);
        let deep_water = vec3(0.02, 0.08, 0.25);
        let shallow_water = vec3(0.1, 0.35, 0.6);
        
        let depth_factor = smoothstep(0.0, 60.0, depth);
        terrain_color = mix(shallow_water, deep_water, depth_factor);
        
        // Apply lighting
        let ambient = 0.5;
        let diffuse = max(dot(normal, light_dir), 0.0) * 0.4;
        terrain_color = terrain_color * (ambient + diffuse);
        
        // Wave highlights
        let wave_highlight = pow(max(dot(normal, light_dir), 0.0), 8.0) * 0.15;
        terrain_color += vec3(wave_highlight);
    } else {
        // === Land Rendering ===
        let slope = normal.y;
        
        // Base colors
        let grass_color = vec3(0.25, 0.55, 0.2);
        let rock_color = vec3(0.45, 0.4, 0.35);
        let sand_color = vec3(0.76, 0.7, 0.5);
        let snow_color = vec3(0.95, 0.97, 1.0);
        
        // Height-based coloring
        if (height < 10.0) {
            let beach_factor = smoothstep(0.0, 10.0, height);
            terrain_color = mix(sand_color, grass_color, beach_factor);
        } else if (height < 100.0) {
            terrain_color = grass_color;
        } else {
            let snow_factor = smoothstep(100.0, 180.0, height);
            terrain_color = mix(grass_color, snow_color, snow_factor);
        }
        
        // Slope-based rock blending
        let rock_factor = 1.0 - smoothstep(0.5, 0.85, slope);
        terrain_color = mix(terrain_color, rock_color, rock_factor);
        
        // Apply lighting
        let ambient = 0.35;
        let diffuse = max(dot(normal, light_dir), 0.0) * 0.65;
        terrain_color = terrain_color * (ambient + diffuse);
    }
    
    // Apply distance fog
    let fog_factor = smoothstep(FOG_START, FOG_END, distance);
    let final_color = mix(terrain_color, FOG_COLOR, fog_factor);
    
    return vec4(final_color, 1.0);
}

