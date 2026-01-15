//! Camera Module
//!
//! Provides 3D free camera (spectator mode) functionality for the Aether Engine.
//! Follows the standards defined in:
//! - `docs/01_STANDARDS/coordinate_systems.md` (Y-Up, right-handed, perspective_rh)
//! - `docs/01_STANDARDS/data_layout.md` (uniform buffer alignment)
//!
//! Features:
//! - WASD movement (planar XZ)
//! - Space/Shift for vertical movement
//! - Mouse look (right-click drag for yaw/pitch)

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
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    /// Combined view-projection matrix.
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

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
// Camera Controller
// ============================================================================

/// Input state for camera movement.
#[derive(Debug, Clone, Copy, Default)]
pub struct CameraInput {
    /// Forward/backward input (-1 to 1).
    pub forward: f32,
    /// Right/left input (-1 to 1).
    pub right: f32,
    /// Up/down input (-1 to 1).
    pub up: f32,
    /// Mouse delta for rotation (x = yaw, y = pitch).
    pub mouse_delta: (f32, f32),
    /// Whether mouse look is active (right mouse button held).
    pub mouse_look_active: bool,
}

/// Free camera controller for spectator-style movement.
pub struct CameraController {
    /// Movement speed in units per second.
    pub speed: f32,
    /// Mouse sensitivity for rotation.
    pub sensitivity: f32,
    /// Current input state.
    pub input: CameraInput,
    /// Sprint multiplier (when holding shift for fast movement, or ctrl for slow).
    pub sprint_multiplier: f32,
}

impl CameraController {
    /// Creates a new camera controller with default settings.
    pub fn new() -> Self {
        Self {
            speed: 50.0,         // Units per second (terrain is big)
            sensitivity: 0.003,  // Mouse sensitivity
            input: CameraInput::default(),
            sprint_multiplier: 1.0,
        }
    }

    /// Process keyboard input for movement.
    pub fn process_keyboard(&mut self, key: winit::keyboard::KeyCode, pressed: bool) -> bool {
        use winit::keyboard::KeyCode;
        
        let value = if pressed { 1.0 } else { 0.0 };
        
        match key {
            KeyCode::KeyW => {
                self.input.forward = if pressed { 1.0 } else if self.input.forward > 0.0 { 0.0 } else { self.input.forward };
                true
            }
            KeyCode::KeyS => {
                self.input.forward = if pressed { -1.0 } else if self.input.forward < 0.0 { 0.0 } else { self.input.forward };
                true
            }
            KeyCode::KeyA => {
                self.input.right = if pressed { -1.0 } else if self.input.right < 0.0 { 0.0 } else { self.input.right };
                true
            }
            KeyCode::KeyD => {
                self.input.right = if pressed { 1.0 } else if self.input.right > 0.0 { 0.0 } else { self.input.right };
                true
            }
            KeyCode::Space => {
                self.input.up = if pressed { 1.0 } else if self.input.up > 0.0 { 0.0 } else { self.input.up };
                true
            }
            KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                self.input.up = if pressed { -1.0 } else if self.input.up < 0.0 { 0.0 } else { self.input.up };
                // Also enable sprint when shift is held
                self.sprint_multiplier = if pressed { 3.0 } else { 1.0 };
                true
            }
            KeyCode::ControlLeft | KeyCode::ControlRight => {
                // Slow movement when control is held
                self.sprint_multiplier = if pressed { 0.25 } else { 1.0 };
                true
            }
            _ => false,
        }
    }

    /// Process mouse button input for look mode.
    pub fn process_mouse_button(&mut self, button: winit::event::MouseButton, pressed: bool) {
        use winit::event::MouseButton;
        
        if button == MouseButton::Right {
            self.input.mouse_look_active = pressed;
        }
    }

    /// Process mouse movement for camera rotation.
    pub fn process_mouse_motion(&mut self, delta: (f64, f64)) {
        if self.input.mouse_look_active {
            self.input.mouse_delta.0 += delta.0 as f32;
            self.input.mouse_delta.1 += delta.1 as f32;
        }
    }

    /// Get the effective speed with sprint multiplier.
    pub fn effective_speed(&self) -> f32 {
        self.speed * self.sprint_multiplier
    }

    /// Clear mouse delta after processing.
    pub fn clear_mouse_delta(&mut self) {
        self.input.mouse_delta = (0.0, 0.0);
    }
}

impl Default for CameraController {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Free Camera (CPU-side with Yaw/Pitch)
// ============================================================================

/// Free camera with position and Euler angles for spectator-style control.
pub struct Camera {
    /// Camera position in world space.
    pub position: Vec3,
    /// Yaw angle in radians (rotation around Y axis).
    pub yaw: f32,
    /// Pitch angle in radians (rotation around local X axis).
    pub pitch: f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Vertical field of view in radians.
    pub fovy: f32,
    /// Near clipping plane distance.
    pub znear: f32,
    /// Far clipping plane distance.
    pub zfar: f32,
}

impl Camera {
    /// Creates a new camera with default parameters.
    pub fn new(aspect: f32) -> Self {
        Self {
            position: Vec3::new(50.0, 80.0, 50.0), // Start elevated above terrain center
            yaw: -std::f32::consts::FRAC_PI_4,    // Look toward terrain center (-45 degrees)
            pitch: -0.5,                           // Look slightly down
            aspect,
            fovy: 60.0_f32.to_radians(),          // Wider FOV for terrain viewing
            znear: 0.1,
            zfar: 2000.0,                         // Far plane for large terrain
        }
    }

    /// Get the forward direction vector from yaw/pitch.
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        ).normalize()
    }

    /// Get the right direction vector.
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    /// Get the up direction vector (camera-relative).
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize()
    }

    /// Update camera from controller input.
    pub fn update(&mut self, controller: &mut CameraController, delta_time: f32) {
        let speed = controller.effective_speed();
        let sensitivity = controller.sensitivity;
        let input = &controller.input;

        // Apply mouse rotation
        self.yaw -= input.mouse_delta.0 * sensitivity;
        self.pitch -= input.mouse_delta.1 * sensitivity;

        // Clamp pitch to prevent flipping
        self.pitch = self.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.1,
            std::f32::consts::FRAC_PI_2 - 0.1,
        );

        // Calculate movement vectors
        let forward = Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize();
        let right = Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos()).normalize();

        // Apply movement
        let velocity = (forward * input.forward + right * input.right) * speed * delta_time;
        self.position += velocity;

        // Vertical movement (global Y axis)
        self.position.y += input.up * speed * delta_time;

        // Prevent going below ground
        self.position.y = self.position.y.max(1.0);

        // Clear mouse delta
        controller.clear_mouse_delta();
    }

    /// Builds the view matrix from position and rotation.
    pub fn build_view_matrix(&self) -> Mat4 {
        let target = self.position + self.forward();
        Mat4::look_at_rh(self.position, target, Vec3::Y)
    }

    /// Builds the projection matrix.
    pub fn build_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar)
    }

    /// Builds the combined view-projection matrix.
    pub fn build_view_projection_matrix(&self) -> Mat4 {
        self.build_projection_matrix() * self.build_view_matrix()
    }

    /// Updates the aspect ratio.
    pub fn update_aspect(&mut self, width: u32, height: u32) {
        if height > 0 {
            self.aspect = width as f32 / height as f32;
        }
    }

    /// Returns the camera's XZ position for terrain chunk centering.
    pub fn get_xz_position(&self) -> (f32, f32) {
        (self.position.x, self.position.z)
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(16.0 / 9.0)
    }
}

// ============================================================================
// Compile-time Size Verification
// ============================================================================

const _: () = assert!(std::mem::size_of::<CameraUniform>() == 64);
