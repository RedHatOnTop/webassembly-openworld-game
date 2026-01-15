// ============================================================================
// Structure Shader - Instanced Rendering with Weathering
// ============================================================================
//
// Renders GLTF meshes with per-instance transforms.
// Includes weathering effects: vertex warping for ruins, wind for trees.
//
// Reference: docs/04_TASKS/task_13_structures.md

// ============================================================================
// Uniforms
// ============================================================================

struct CameraUniform {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

// ============================================================================
// Vertex Input/Output
// ============================================================================

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
}

struct InstanceInput {
    @location(4) model_0: vec4<f32>,
    @location(5) model_1: vec4<f32>,
    @location(6) model_2: vec4<f32>,
    @location(7) model_3: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
}

// ============================================================================
// Vertex Shader
// ============================================================================

@vertex
fn vs_structure(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    // Reconstruct model matrix from instance data
    let model_matrix = mat4x4<f32>(
        instance.model_0,
        instance.model_1,
        instance.model_2,
        instance.model_3,
    );
    
    // Apply local position
    var local_pos = vertex.position;
    
    // === Weathering: Vertex Warping for Ruins ===
    // Makes straight pillars look old and crooked
    // This applies a subtle sine wave displacement based on height
    let warp_strength = 0.03;
    local_pos.x += sin(local_pos.y * 0.5) * warp_strength;
    local_pos.z += cos(local_pos.y * 0.7) * warp_strength;
    
    // Transform to world space
    let world_pos = model_matrix * vec4(local_pos, 1.0);
    
    // Transform normal (ignore translation)
    let normal_matrix = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz,
    );
    let world_normal = normalize(normal_matrix * vertex.normal);
    
    // Transform to clip space
    var output: VertexOutput;
    output.clip_position = camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = world_normal;
    output.uv = vertex.uv;
    output.color = vertex.color;
    
    return output;
}

// ============================================================================
// Fragment Shader
// ============================================================================

@fragment
fn fs_structure(in: VertexOutput) -> @location(0) vec4<f32> {
    // Base color from vertex color (or fallback)
    var base_color = in.color;
    
    // === Triplanar Moss: Blend green on upward-facing surfaces ===
    let moss_color = vec4(0.3, 0.5, 0.2, 1.0);
    let moss_strength = smoothstep(0.6, 0.9, in.world_normal.y);
    base_color = mix(base_color, moss_color, moss_strength * 0.4);
    
    // === Simple Lighting ===
    let light_dir = normalize(vec3(0.5, 1.0, 0.3));
    let diffuse = max(dot(in.world_normal, light_dir), 0.0);
    let ambient = 0.4;
    let lighting = ambient + diffuse * 0.6;
    
    // Apply lighting
    var final_color = base_color.rgb * lighting;
    
    // === Height-based darkening (ground contact) ===
    let ground_darken = smoothstep(0.0, 2.0, in.world_position.y);
    final_color *= 0.7 + ground_darken * 0.3;
    
    return vec4(final_color, base_color.a);
}
