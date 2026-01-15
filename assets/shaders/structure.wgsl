// ============================================================================
// Structure Shader - Instanced Rendering with Weathering
// ============================================================================
//
// Renders GLTF meshes with per-instance transforms.
// Handles models with or without vertex colors by using normal-based coloring.
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
    @location(4) local_y: f32,  // For tree/foliage detection
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
    
    var local_pos = vertex.position;
    
    // Transform to world space
    let world_pos = model_matrix * vec4(local_pos, 1.0);
    
    // Transform normal
    let normal_matrix = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz,
    );
    let world_normal = normalize(normal_matrix * vertex.normal);
    
    var output: VertexOutput;
    output.clip_position = camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = world_normal;
    output.uv = vertex.uv;
    output.color = vertex.color;
    output.local_y = vertex.position.y;
    
    return output;
}

// ============================================================================
// Fragment Shader
// ============================================================================

@fragment
fn fs_structure(in: VertexOutput) -> @location(0) vec4<f32> {
    // Check if vertex color is default white (no vertex colors in model)
    let has_vertex_color = !(in.color.r > 0.99 && in.color.g > 0.99 && in.color.b > 0.99);
    
    var base_color: vec3<f32>;
    
    if (has_vertex_color) {
        base_color = in.color.rgb;
    } else {
        // Generate procedural color based on surface properties
        // Trees: green for upper parts, brown for trunk
        // Rocks: gray
        // Ruins: beige/tan
        
        let is_foliage = in.local_y > 0.3 && in.world_normal.y > 0.2;
        let is_trunk = in.local_y < 0.3 && in.local_y > -0.5;
        
        if (is_foliage) {
            // Tree foliage - green with variation
            let green_variation = fract(in.world_position.x * 0.1 + in.world_position.z * 0.1) * 0.2;
            base_color = vec3(0.2 + green_variation * 0.1, 0.5 + green_variation, 0.15);
        } else if (is_trunk) {
            // Tree trunk - brown
            base_color = vec3(0.4, 0.25, 0.1);
        } else {
            // Rock/Ruin - gray/tan based on normal
            let rock_factor = abs(in.world_normal.x) + abs(in.world_normal.z);
            base_color = mix(
                vec3(0.6, 0.55, 0.5),  // Tan for flat surfaces
                vec3(0.5, 0.5, 0.5),   // Gray for sides
                rock_factor * 0.5
            );
        }
    }
    
    // === Moss overlay on upward-facing surfaces ===
    let moss_color = vec3(0.25, 0.4, 0.15);
    let moss_strength = smoothstep(0.7, 0.95, in.world_normal.y) * 0.5;
    base_color = mix(base_color, moss_color, moss_strength);
    
    // === Simple Lighting ===
    let light_dir = normalize(vec3(0.5, 1.0, 0.3));
    let diffuse = max(dot(in.world_normal, light_dir), 0.0);
    let ambient = 0.45;
    let lighting = ambient + diffuse * 0.55;
    
    var final_color = base_color * lighting;
    
    return vec4(final_color, 1.0);
}
