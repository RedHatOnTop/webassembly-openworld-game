// Cube Shader - Aether Engine
//
// 3D cube rendering with camera transformation.
// Demonstrates uniform buffers and MVP matrix multiplication.

// ============================================================================
// Camera Uniform
// ============================================================================

/// Camera uniform buffer containing the combined view-projection matrix.
/// Bound at Group 0, Binding 0.
struct CameraUniform {
    /// Combined model-view-projection matrix.
    /// Transforms local-space positions to clip-space.
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// ============================================================================
// Vertex Stage
// ============================================================================

/// Vertex input from the vertex buffer.
struct VertexInput {
    /// Vertex position in local space.
    @location(0) position: vec3<f32>,
    /// Vertex color (RGB, linear).
    @location(1) color: vec3<f32>,
}

/// Output from vertex shader / Input to fragment shader.
struct VertexOutput {
    /// Clip-space position (required builtin).
    @builtin(position) clip_position: vec4<f32>,
    /// Interpolated vertex color.
    @location(0) color: vec3<f32>,
}

/// Vertex shader entry point.
/// Transforms vertex position using the camera's view-projection matrix.
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Transform from local space to clip space using MVP matrix
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    // Pass color for interpolation
    out.color = in.color;
    return out;
}

// ============================================================================
// Fragment Stage
// ============================================================================

/// Fragment shader entry point.
/// Outputs the interpolated color to render target 0.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Output color with full opacity
    return vec4<f32>(in.color, 1.0);
}
