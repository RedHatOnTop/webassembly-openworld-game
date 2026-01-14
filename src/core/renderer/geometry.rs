//! Geometry Primitives
//!
//! Defines vertex structures and geometry data for rendering.
//! All vertex data follows the standards defined in:
//! - `docs/01_STANDARDS/coordinate_systems.md` (CCW winding order)
//! - `docs/01_STANDARDS/data_layout.md` (memory alignment)

use bytemuck::{Pod, Zeroable};

// ============================================================================
// Vertex Definition
// ============================================================================

/// A vertex with position and color attributes.
///
/// Memory Layout (24 bytes total):
/// - `position`: [f32; 3] at offset 0 (12 bytes)
/// - `color`: [f32; 3] at offset 12 (12 bytes)
///
/// Uses `#[repr(C)]` to ensure predictable memory layout matching WGSL expectations.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    /// Vertex position in local/clip space.
    pub position: [f32; 3],
    /// Vertex color (RGB, linear color space).
    pub color: [f32; 3],
}

impl Vertex {
    /// Returns the vertex buffer layout descriptor for pipeline creation.
    ///
    /// This describes how vertex data is organized in the buffer:
    /// - Stride: 24 bytes per vertex
    /// - Step mode: Per-vertex (not per-instance)
    /// - Attributes: position at location 0, color at location 1
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position: vec3<f32> at location 0
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Color: vec3<f32> at location 1
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// ============================================================================
// Triangle Geometry
// ============================================================================

/// Triangle vertices in Counter-Clockwise (CCW) order.
///
/// Winding order is critical for correct face culling.
/// Per `coordinate_systems.md` Section 7:
/// - Front face: CCW (Counter-Clockwise)
/// - Back face culling is enabled
///
/// Vertex layout (viewed from front):
/// ```text
///        v0 (top, red)
///        /\
///       /  \
///      /    \
///     /______\
///   v1        v2
/// (green)    (blue)
/// ```
///
/// The vertices are in clip space where:
/// - X: [-1, +1] (left to right)
/// - Y: [-1, +1] (bottom to top)  
/// - Z: [0, 1] (near to far, 0.0 = center)
pub const TRIANGLE_VERTICES: &[Vertex] = &[
    // v0: Top vertex (red)
    Vertex {
        position: [0.0, 0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    // v1: Bottom-left vertex (green)
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    // v2: Bottom-right vertex (blue)
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
];

/// Number of vertices in the triangle.
pub const TRIANGLE_VERTEX_COUNT: u32 = 3;

// ============================================================================
// Cube Geometry
// ============================================================================

/// Cube vertices with per-face colors.
///
/// Uses 24 vertices (4 per face x 6 faces) to allow:
/// - Hard edges between faces (no shared normals)
/// - Distinct colors per face for visual identification
///
/// Cube is centered at origin with size 1.0 (-0.5 to +0.5 on each axis).
///
/// Face colors:
/// - Front  (+Z): Red
/// - Back   (-Z): Cyan
/// - Top    (+Y): Green
/// - Bottom (-Y): Magenta
/// - Right  (+X): Blue
/// - Left   (-X): Yellow
///
/// All faces use CCW winding order when viewed from outside the cube.
pub const CUBE_VERTICES: &[Vertex] = &[
    // Front face (+Z) - Red
    // Vertices ordered CCW when looking at front face from +Z
    Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0] }, // 0: bottom-left
    Vertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0] }, // 1: bottom-right
    Vertex { position: [ 0.5,  0.5,  0.5], color: [1.0, 0.0, 0.0] }, // 2: top-right
    Vertex { position: [-0.5,  0.5,  0.5], color: [1.0, 0.0, 0.0] }, // 3: top-left

    // Back face (-Z) - Cyan
    // Vertices ordered CCW when looking at back face from -Z
    Vertex { position: [ 0.5, -0.5, -0.5], color: [0.0, 1.0, 1.0] }, // 4: bottom-left (from -Z view)
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.0, 1.0, 1.0] }, // 5: bottom-right
    Vertex { position: [-0.5,  0.5, -0.5], color: [0.0, 1.0, 1.0] }, // 6: top-right
    Vertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 1.0, 1.0] }, // 7: top-left

    // Top face (+Y) - Green
    // Vertices ordered CCW when looking at top face from +Y
    Vertex { position: [-0.5,  0.5,  0.5], color: [0.0, 1.0, 0.0] }, // 8: front-left
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.0, 1.0, 0.0] }, // 9: front-right
    Vertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 1.0, 0.0] }, // 10: back-right
    Vertex { position: [-0.5,  0.5, -0.5], color: [0.0, 1.0, 0.0] }, // 11: back-left

    // Bottom face (-Y) - Magenta
    // Vertices ordered CCW when looking at bottom face from -Y
    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 0.0, 1.0] }, // 12: back-left (from -Y view)
    Vertex { position: [ 0.5, -0.5, -0.5], color: [1.0, 0.0, 1.0] }, // 13: back-right
    Vertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 0.0, 1.0] }, // 14: front-right
    Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 0.0, 1.0] }, // 15: front-left

    // Right face (+X) - Blue
    // Vertices ordered CCW when looking at right face from +X
    Vertex { position: [ 0.5, -0.5,  0.5], color: [0.0, 0.0, 1.0] }, // 16: front-bottom
    Vertex { position: [ 0.5, -0.5, -0.5], color: [0.0, 0.0, 1.0] }, // 17: back-bottom
    Vertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 0.0, 1.0] }, // 18: back-top
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.0, 0.0, 1.0] }, // 19: front-top

    // Left face (-X) - Yellow
    // Vertices ordered CCW when looking at left face from -X
    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 1.0, 0.0] }, // 20: back-bottom (from -X view)
    Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 1.0, 0.0] }, // 21: front-bottom
    Vertex { position: [-0.5,  0.5,  0.5], color: [1.0, 1.0, 0.0] }, // 22: front-top
    Vertex { position: [-0.5,  0.5, -0.5], color: [1.0, 1.0, 0.0] }, // 23: back-top
];

/// Cube indices for indexed drawing.
///
/// Each face uses 2 triangles (6 indices) in CCW winding order.
/// Total: 36 indices for 12 triangles (6 faces x 2 triangles).
pub const CUBE_INDICES: &[u16] = &[
    // Front face (+Z)
    0,  1,  2,   2,  3,  0,
    // Back face (-Z)
    4,  5,  6,   6,  7,  4,
    // Top face (+Y)
    8,  9, 10,  10, 11,  8,
    // Bottom face (-Y)
    12, 13, 14,  14, 15, 12,
    // Right face (+X)
    16, 17, 18,  18, 19, 16,
    // Left face (-X)
    20, 21, 22,  22, 23, 20,
];

/// Number of vertices in the cube.
pub const CUBE_VERTEX_COUNT: u32 = 24;

/// Number of indices in the cube (36 = 6 faces x 2 triangles x 3 vertices).
pub const CUBE_INDEX_COUNT: u32 = 36;
