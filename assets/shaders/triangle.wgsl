// Triangle Shader - Aether Engine
//
// Basic vertex/fragment shader for rendering a colored triangle.
// Demonstrates the minimal rendering pipeline setup.

// ============================================================================
// Vertex Stage
// ============================================================================

/// Vertex input from the vertex buffer.
struct VertexInput {
    /// Vertex position in clip space (for this simple case).
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
/// Transforms vertex position and passes color to fragment stage.
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Pass position directly (already in clip space for this demo).
    // Z=0 places the triangle in the middle of the depth range.
    out.clip_position = vec4<f32>(in.position, 1.0);
    // Pass color for interpolation.
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
    // Output color with full opacity.
    return vec4<f32>(in.color, 1.0);
}
