// ============================================================================
// Vegetation Render Shader
// ============================================================================
//
// Renders vegetation instances using indirect drawing.
// Supports wind animation (visual only, time-based is OK here).
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

struct CameraUniform {
    view_proj: mat4x4<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) type_id: u32,
}

// ============================================================================
// Bindings
// ============================================================================

// Vegetation bind group (group 0)
@group(0) @binding(0) var<storage, read> instances: array<GrassInstance>;
@group(0) @binding(1) var<uniform> uniforms: VegetationUniforms;
@group(0) @binding(2) var vegetation_textures: texture_2d_array<f32>;
@group(0) @binding(3) var vegetation_sampler: sampler;

// Camera bind group (group 1)
@group(1) @binding(0) var<uniform> camera: CameraUniform;

// ============================================================================
// Constants
// ============================================================================

const PI: f32 = 3.14159265359;
const WIND_STRENGTH: f32 = 0.15;
const WIND_SPEED: f32 = 2.0;
const WIND_WAVE_LENGTH: f32 = 0.3;

// ============================================================================
// Vertex Shader
// ============================================================================

@vertex
fn vs_vegetation(
    in: VertexInput,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    let instance = instances[instance_index];
    
    // Get local vertex position
    var local_pos = in.position;
    
    // === Wind Animation (Time-based is OK for visuals) ===
    // Only affect upper vertices (grass tip)
    let height_factor = max(local_pos.y, 0.0);  // 0 at base, 1 at tip
    let wind_phase = uniforms.time * WIND_SPEED + instance.position.x * WIND_WAVE_LENGTH + instance.position.z * WIND_WAVE_LENGTH * 0.7;
    let wind_offset = sin(wind_phase) * WIND_STRENGTH * height_factor * instance.scale;
    let wind_offset_z = cos(wind_phase * 0.8) * WIND_STRENGTH * 0.5 * height_factor * instance.scale;
    
    // === Y-axis Rotation (Billboarding) ===
    let rot_cos = cos(instance.rotation);
    let rot_sin = sin(instance.rotation);
    let rotated_x = local_pos.x * rot_cos - local_pos.z * rot_sin;
    let rotated_z = local_pos.x * rot_sin + local_pos.z * rot_cos;
    
    // === Apply transformations ===
    var world_pos = instance.position;
    world_pos.x += rotated_x * instance.scale + wind_offset;
    world_pos.y += local_pos.y * instance.scale;
    world_pos.z += rotated_z * instance.scale + wind_offset_z;
    
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4(world_pos, 1.0);
    out.uv = in.uv;
    out.world_position = world_pos;
    out.type_id = instance.type_id;
    
    return out;
}

// ============================================================================
// Fragment Shader
// ============================================================================

@fragment
fn fs_vegetation(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample texture from array using type_id
    let tex_color = textureSample(vegetation_textures, vegetation_sampler, in.uv, i32(in.type_id));
    
    // Alpha test (discard transparent pixels)
    if (tex_color.a < 0.5) {
        discard;
    }
    
    // Simple grass coloring (if using fallback checkerboard, tint it green)
    var final_color = tex_color.rgb;
    
    // Add slight color variation based on position
    let variation = sin(in.world_position.x * 0.5) * cos(in.world_position.z * 0.5) * 0.1;
    final_color.g += variation;
    
    // Simple lighting (sun from above-right)
    let light_dir = normalize(vec3(0.5, 1.0, 0.3));
    let ambient = 0.5;
    let diffuse = max(dot(vec3(0.0, 1.0, 0.0), light_dir), 0.0) * 0.5;
    final_color = final_color * (ambient + diffuse);
    
    // Distance fade (LOD)
    let camera_pos = vec3(uniforms.camera_x, uniforms.camera_y, uniforms.camera_z);
    let dist = length(in.world_position - camera_pos);
    let fade_start = 80.0;
    let fade_end = 120.0;
    let fade = 1.0 - smoothstep(fade_start, fade_end, dist);
    
    return vec4(final_color, tex_color.a * fade);
}
