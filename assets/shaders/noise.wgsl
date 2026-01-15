// ============================================================================
// Noise Library - Aether Engine
// ============================================================================
//
// GPU-optimized noise functions for procedural terrain generation.
// Implements Simplex Noise (3D/4D), FBM, Ridge Noise, and Cubic Spline.
//
// Reference: docs/02_ARCHITECTURE/world_gen_strategy.md
// ============================================================================

// ============================================================================
// Constants and Lookup Tables
// ============================================================================

// Permutation table (256 entries, doubled for wrap-around)
// Using a hash-based approach for WGSL compatibility
const PERM_MULTIPLIER: u32 = 1664525u;
const PERM_INCREMENT: u32 = 1013904223u;

// Simplex noise skew factors
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

// 4D Gradient vectors (32 directions)
const GRAD4: array<vec4<f32>, 32> = array<vec4<f32>, 32>(
    vec4( 0.0,  1.0,  1.0,  1.0), vec4( 0.0,  1.0,  1.0, -1.0), vec4( 0.0,  1.0, -1.0,  1.0), vec4( 0.0,  1.0, -1.0, -1.0),
    vec4( 0.0, -1.0,  1.0,  1.0), vec4( 0.0, -1.0,  1.0, -1.0), vec4( 0.0, -1.0, -1.0,  1.0), vec4( 0.0, -1.0, -1.0, -1.0),
    vec4( 1.0,  0.0,  1.0,  1.0), vec4( 1.0,  0.0,  1.0, -1.0), vec4( 1.0,  0.0, -1.0,  1.0), vec4( 1.0,  0.0, -1.0, -1.0),
    vec4(-1.0,  0.0,  1.0,  1.0), vec4(-1.0,  0.0,  1.0, -1.0), vec4(-1.0,  0.0, -1.0,  1.0), vec4(-1.0,  0.0, -1.0, -1.0),
    vec4( 1.0,  1.0,  0.0,  1.0), vec4( 1.0,  1.0,  0.0, -1.0), vec4( 1.0, -1.0,  0.0,  1.0), vec4( 1.0, -1.0,  0.0, -1.0),
    vec4(-1.0,  1.0,  0.0,  1.0), vec4(-1.0,  1.0,  0.0, -1.0), vec4(-1.0, -1.0,  0.0,  1.0), vec4(-1.0, -1.0,  0.0, -1.0),
    vec4( 1.0,  1.0,  1.0,  0.0), vec4( 1.0,  1.0, -1.0,  0.0), vec4( 1.0, -1.0,  1.0,  0.0), vec4( 1.0, -1.0, -1.0,  0.0),
    vec4(-1.0,  1.0,  1.0,  0.0), vec4(-1.0,  1.0, -1.0,  0.0), vec4(-1.0, -1.0,  1.0,  0.0), vec4(-1.0, -1.0, -1.0,  0.0)
);

// ============================================================================
// Hash Functions (Permutation Alternative)
// ============================================================================

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

/// 4D hash for simplex noise.
fn hash_4d(x: i32, y: i32, z: i32, w: i32) -> u32 {
    return hash_int(x + i32(hash_int(y + i32(hash_int(z + i32(hash_int(w)))))));
}

// ============================================================================
// Simplex Noise 3D
// ============================================================================

/// 3D Simplex Noise.
/// Returns value in range [-1.0, 1.0].
fn simplex_noise_3d(p: vec3<f32>) -> f32 {
    // Skew input space to determine which simplex cell we're in
    let s = (p.x + p.y + p.z) * F3;
    let i = floor(p.x + s);
    let j = floor(p.y + s);
    let k = floor(p.z + s);
    
    // Unskew back to (x,y,z) space
    let t = (i + j + k) * G3;
    let X0 = i - t;
    let Y0 = j - t;
    let Z0 = k - t;
    
    // Position within cell
    let x0 = p.x - X0;
    let y0 = p.y - Y0;
    let z0 = p.z - Z0;
    
    // Determine which simplex we're in (6 possibilities)
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
    
    // Offsets for remaining corners
    let x1 = x0 - f32(i1) + G3;
    let y1 = y0 - f32(j1) + G3;
    let z1 = z0 - f32(k1) + G3;
    let x2 = x0 - f32(i2) + 2.0 * G3;
    let y2 = y0 - f32(j2) + 2.0 * G3;
    let z2 = z0 - f32(k2) + 2.0 * G3;
    let x3 = x0 - 1.0 + 3.0 * G3;
    let y3 = y0 - 1.0 + 3.0 * G3;
    let z3 = z0 - 1.0 + 3.0 * G3;
    
    // Convert to integer coordinates
    let ii = i32(i) & 255;
    let jj = i32(j) & 255;
    let kk = i32(k) & 255;
    
    // Calculate contribution from each corner
    var n0: f32 = 0.0;
    var n1: f32 = 0.0;
    var n2: f32 = 0.0;
    var n3: f32 = 0.0;
    
    // Corner 0
    var t0 = 0.6 - x0*x0 - y0*y0 - z0*z0;
    if (t0 > 0.0) {
        t0 *= t0;
        let gi0 = hash_3d(ii, jj, kk) % 12u;
        n0 = t0 * t0 * dot(GRAD3[gi0], vec3(x0, y0, z0));
    }
    
    // Corner 1
    var t1 = 0.6 - x1*x1 - y1*y1 - z1*z1;
    if (t1 > 0.0) {
        t1 *= t1;
        let gi1 = hash_3d(ii + i1, jj + j1, kk + k1) % 12u;
        n1 = t1 * t1 * dot(GRAD3[gi1], vec3(x1, y1, z1));
    }
    
    // Corner 2
    var t2 = 0.6 - x2*x2 - y2*y2 - z2*z2;
    if (t2 > 0.0) {
        t2 *= t2;
        let gi2 = hash_3d(ii + i2, jj + j2, kk + k2) % 12u;
        n2 = t2 * t2 * dot(GRAD3[gi2], vec3(x2, y2, z2));
    }
    
    // Corner 3
    var t3 = 0.6 - x3*x3 - y3*y3 - z3*z3;
    if (t3 > 0.0) {
        t3 *= t3;
        let gi3 = hash_3d(ii + 1, jj + 1, kk + 1) % 12u;
        n3 = t3 * t3 * dot(GRAD3[gi3], vec3(x3, y3, z3));
    }
    
    // Scale to [-1, 1]
    return 32.0 * (n0 + n1 + n2 + n3);
}

// ============================================================================
// Simplex Noise 4D (Time-variant)
// ============================================================================

/// 4D Simplex Noise for time-varying effects.
/// Returns value in range [-1.0, 1.0].
fn simplex_noise_4d(p: vec4<f32>) -> f32 {
    // Skew the 4D space to determine which cell we're in
    let s = (p.x + p.y + p.z + p.w) * F4;
    let i = floor(p.x + s);
    let j = floor(p.y + s);
    let k = floor(p.z + s);
    let l = floor(p.w + s);
    
    // Unskew back
    let t = (i + j + k + l) * G4;
    let X0 = i - t;
    let Y0 = j - t;
    let Z0 = k - t;
    let W0 = l - t;
    
    // Position within cell
    let x0 = p.x - X0;
    let y0 = p.y - Y0;
    let z0 = p.z - Z0;
    let w0 = p.w - W0;
    
    // Determine simplex by ranking coordinates
    var rank = vec4<i32>(0, 0, 0, 0);
    if (x0 > y0) { rank.x += 1; } else { rank.y += 1; }
    if (x0 > z0) { rank.x += 1; } else { rank.z += 1; }
    if (x0 > w0) { rank.x += 1; } else { rank.w += 1; }
    if (y0 > z0) { rank.y += 1; } else { rank.z += 1; }
    if (y0 > w0) { rank.y += 1; } else { rank.w += 1; }
    if (z0 > w0) { rank.z += 1; } else { rank.w += 1; }
    
    // Simplex corners
    let i1 = select(0, 1, rank.x >= 3);
    let j1 = select(0, 1, rank.y >= 3);
    let k1 = select(0, 1, rank.z >= 3);
    let l1 = select(0, 1, rank.w >= 3);
    
    let i2 = select(0, 1, rank.x >= 2);
    let j2 = select(0, 1, rank.y >= 2);
    let k2 = select(0, 1, rank.z >= 2);
    let l2 = select(0, 1, rank.w >= 2);
    
    let i3 = select(0, 1, rank.x >= 1);
    let j3 = select(0, 1, rank.y >= 1);
    let k3 = select(0, 1, rank.z >= 1);
    let l3 = select(0, 1, rank.w >= 1);
    
    // Offsets for remaining corners
    let x1 = x0 - f32(i1) + G4;
    let y1 = y0 - f32(j1) + G4;
    let z1 = z0 - f32(k1) + G4;
    let w1 = w0 - f32(l1) + G4;
    
    let x2 = x0 - f32(i2) + 2.0 * G4;
    let y2 = y0 - f32(j2) + 2.0 * G4;
    let z2 = z0 - f32(k2) + 2.0 * G4;
    let w2 = w0 - f32(l2) + 2.0 * G4;
    
    let x3 = x0 - f32(i3) + 3.0 * G4;
    let y3 = y0 - f32(j3) + 3.0 * G4;
    let z3 = z0 - f32(k3) + 3.0 * G4;
    let w3 = w0 - f32(l3) + 3.0 * G4;
    
    let x4 = x0 - 1.0 + 4.0 * G4;
    let y4 = y0 - 1.0 + 4.0 * G4;
    let z4 = z0 - 1.0 + 4.0 * G4;
    let w4 = w0 - 1.0 + 4.0 * G4;
    
    // Integer coordinates
    let ii = i32(i) & 255;
    let jj = i32(j) & 255;
    let kk = i32(k) & 255;
    let ll = i32(l) & 255;
    
    // Contributions from each corner
    var n0: f32 = 0.0;
    var n1: f32 = 0.0;
    var n2: f32 = 0.0;
    var n3: f32 = 0.0;
    var n4: f32 = 0.0;
    
    var t0 = 0.6 - x0*x0 - y0*y0 - z0*z0 - w0*w0;
    if (t0 > 0.0) {
        t0 *= t0;
        let gi0 = hash_4d(ii, jj, kk, ll) % 32u;
        n0 = t0 * t0 * dot(GRAD4[gi0], vec4(x0, y0, z0, w0));
    }
    
    var t1 = 0.6 - x1*x1 - y1*y1 - z1*z1 - w1*w1;
    if (t1 > 0.0) {
        t1 *= t1;
        let gi1 = hash_4d(ii + i1, jj + j1, kk + k1, ll + l1) % 32u;
        n1 = t1 * t1 * dot(GRAD4[gi1], vec4(x1, y1, z1, w1));
    }
    
    var t2 = 0.6 - x2*x2 - y2*y2 - z2*z2 - w2*w2;
    if (t2 > 0.0) {
        t2 *= t2;
        let gi2 = hash_4d(ii + i2, jj + j2, kk + k2, ll + l2) % 32u;
        n2 = t2 * t2 * dot(GRAD4[gi2], vec4(x2, y2, z2, w2));
    }
    
    var t3 = 0.6 - x3*x3 - y3*y3 - z3*z3 - w3*w3;
    if (t3 > 0.0) {
        t3 *= t3;
        let gi3 = hash_4d(ii + i3, jj + j3, kk + k3, ll + l3) % 32u;
        n3 = t3 * t3 * dot(GRAD4[gi3], vec4(x3, y3, z3, w3));
    }
    
    var t4 = 0.6 - x4*x4 - y4*y4 - z4*z4 - w4*w4;
    if (t4 > 0.0) {
        t4 *= t4;
        let gi4 = hash_4d(ii + 1, jj + 1, kk + 1, ll + 1) % 32u;
        n4 = t4 * t4 * dot(GRAD4[gi4], vec4(x4, y4, z4, w4));
    }
    
    // Scale to [-1, 1]
    return 27.0 * (n0 + n1 + n2 + n3 + n4);
}

// ============================================================================
// Fractal Brownian Motion (FBM)
// ============================================================================

/// Fractal Brownian Motion using 3D Simplex Noise.
/// Combines multiple octaves for natural-looking terrain.
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

/// FBM with 4D noise for time-variant effects.
fn fbm_4d(p: vec4<f32>, octaves: i32) -> f32 {
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var frequency: f32 = 1.0;
    var max_value: f32 = 0.0;
    
    for (var i = 0; i < octaves; i++) {
        value += amplitude * simplex_noise_4d(p * frequency);
        max_value += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value / max_value;
}

// ============================================================================
// Ridge Noise (for Mountains)
// ============================================================================

/// Ridge Noise for sharp mountain peaks.
/// Uses absolute value to create ridges where noise crosses zero.
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
        weight = clamp(n * 2.0, 0.0, 1.0);  // Weight successive octaves by previous
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value / max_value;
}

// ============================================================================
// Cubic Spline Interpolation
// ============================================================================

/// Cubic Hermite spline interpolation between control points.
/// Crucial for remapping noise values to create distinct terrain zones.
///
/// Control points format: vec2(input_value, output_value)
/// Points must be sorted by x (input) value.
fn cubic_spline(t: f32, points: array<vec2<f32>, 6>) -> f32 {
    // Clamp to valid range
    let clamped_t = clamp(t, points[0].x, points[5].x);
    
    // Find the segment containing t
    var segment: i32 = 0;
    for (var i = 0; i < 5; i++) {
        if (clamped_t >= points[i].x && clamped_t < points[i + 1].x) {
            segment = i;
            break;
        }
    }
    // Handle edge case for t == last point
    if (clamped_t >= points[5].x) {
        return points[5].y;
    }
    
    // Get the two endpoints of this segment
    let p0 = points[segment];
    let p1 = points[segment + 1];
    
    // Get adjacent points for tangent calculation (with boundary handling)
    let pm1 = select(points[segment], points[segment - 1], segment > 0);
    let p2 = select(points[segment + 1], points[segment + 2], segment < 4);
    
    // Calculate tangents using Catmull-Rom style
    let m0 = (p1.y - pm1.y) / (p1.x - pm1.x + 0.0001);
    let m1 = (p2.y - p0.y) / (p2.x - p0.x + 0.0001);
    
    // Normalize t to [0, 1] within this segment
    let segment_t = (clamped_t - p0.x) / (p1.x - p0.x);
    let t2 = segment_t * segment_t;
    let t3 = t2 * segment_t;
    
    // Hermite basis functions
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + segment_t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    
    // Interpolate
    let segment_width = p1.x - p0.x;
    return h00 * p0.y + h10 * segment_width * m0 + h01 * p1.y + h11 * segment_width * m1;
}

/// Simpler linear spline for performance-critical paths.
fn linear_spline(t: f32, points: array<vec2<f32>, 6>) -> f32 {
    let clamped_t = clamp(t, points[0].x, points[5].x);
    
    for (var i = 0; i < 5; i++) {
        if (clamped_t >= points[i].x && clamped_t <= points[i + 1].x) {
            let segment_t = (clamped_t - points[i].x) / (points[i + 1].x - points[i].x + 0.0001);
            return mix(points[i].y, points[i + 1].y, segment_t);
        }
    }
    
    return points[5].y;
}

// ============================================================================
// Domain Warping
// ============================================================================

/// Apply domain warping to create more organic terrain shapes.
/// Simulates geological folding and fluid-like terrain flow.
fn domain_warp_3d(p: vec3<f32>, amplitude: f32, frequency: f32) -> vec3<f32> {
    let warp = vec3<f32>(
        simplex_noise_3d(p * frequency + vec3(0.0, 0.0, 0.0)),
        simplex_noise_3d(p * frequency + vec3(5.2, 1.3, 0.0)),
        simplex_noise_3d(p * frequency + vec3(0.0, 0.0, 9.4))
    );
    return p + warp * amplitude;
}
