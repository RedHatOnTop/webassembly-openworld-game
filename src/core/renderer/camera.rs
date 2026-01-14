//! Camera Module
//!
//! Provides 3D camera functionality for the Aether Engine.
//! Follows the standards defined in:
//! - `docs/01_STANDARDS/coordinate_systems.md` (Y-Up, right-handed, perspective_rh)
//! - `docs/01_STANDARDS/data_layout.md` (uniform buffer alignment)

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

// ============================================================================
// Camera Uniform (GPU-side)
// ============================================================================

/// GPU-side camera uniform data.
///
/// Contains the combined view-projection matrix for vertex transformation.
/// This struct is uploaded to a uniform buffer and read by the vertex shader.
///
/// Memory Layout (64 bytes):
/// - `view_proj`: mat4x4<f32> at offset 0 (64 bytes, 16-byte aligned)
///
/// The mat4x4 is stored as 4 columns of vec4, matching WGSL expectations.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    /// Combined model-view-projection matrix.
    /// Stored as column-major [[f32; 4]; 4] to match WGSL mat4x4<f32>.
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    /// Creates a new CameraUniform initialized to the identity matrix.
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

    /// Updates the uniform with a new view-projection matrix.
    ///
    /// # Arguments
    /// * `matrix` - The combined model-view-projection matrix.
    pub fn update_view_proj(&mut self, matrix: Mat4) {
        self.view_proj = matrix.to_cols_array_2d();
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Camera (CPU-side)
// ============================================================================

/// CPU-side camera for managing view and projection matrices.
///
/// Uses a right-handed coordinate system with Y-up convention,
/// matching the standards in `coordinate_systems.md`.
pub struct Camera {
    /// Camera position in world space.
    pub eye: Vec3,
    /// Point the camera is looking at.
    pub target: Vec3,
    /// Up vector (Y-up per coordinate_systems.md).
    pub up: Vec3,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Vertical field of view in radians.
    pub fovy: f32,
    /// Near clipping plane distance (must be > 0).
    pub znear: f32,
    /// Far clipping plane distance.
    pub zfar: f32,
}

impl Camera {
    /// Creates a new camera with default parameters.
    ///
    /// Default setup:
    /// - Eye at (0, 10, 20) looking at origin (for particle grid view)
    /// - 45 degree vertical FOV
    /// - Near plane at 0.1, far plane at 100.0
    ///
    /// # Arguments
    /// * `aspect` - Initial aspect ratio (width / height).
    pub fn new(aspect: f32) -> Self {
        Self {
            eye: Vec3::new(0.0, 10.0, 20.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            aspect,
            fovy: 45.0_f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
        }
    }

    /// Builds the view matrix.
    ///
    /// Uses `look_at_rh` for right-handed coordinate system.
    pub fn build_view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    /// Builds the projection matrix.
    ///
    /// Uses `perspective_rh` for:
    /// - Right-handed coordinate system
    /// - Z range [0, 1] (WebGPU/Vulkan convention)
    pub fn build_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar)
    }

    /// Builds the combined view-projection matrix.
    ///
    /// Returns `projection * view` for transforming world-space
    /// positions to clip-space.
    pub fn build_view_projection_matrix(&self) -> Mat4 {
        self.build_projection_matrix() * self.build_view_matrix()
    }

    /// Updates the aspect ratio.
    ///
    /// Call this when the window is resized to prevent distortion.
    ///
    /// # Arguments
    /// * `width` - New window width in pixels.
    /// * `height` - New window height in pixels.
    pub fn update_aspect(&mut self, width: u32, height: u32) {
        if height > 0 {
            self.aspect = width as f32 / height as f32;
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(16.0 / 9.0) // Default 16:9 aspect ratio
    }
}

// ============================================================================
// Compile-time Size Verification
// ============================================================================

/// Verify CameraUniform is exactly 64 bytes (size of mat4x4<f32>).
const _: () = assert!(std::mem::size_of::<CameraUniform>() == 64);
