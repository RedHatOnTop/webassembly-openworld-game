# Coordinate Systems

This document defines the coordinate system conventions used throughout the Aether Engine. Strict adherence to these standards is mandatory to prevent transformation bugs and ensure consistent rendering across all subsystems.

---

## Table of Contents

1. [World Coordinate System](#1-world-coordinate-system)
2. [Normalized Device Coordinates (NDC)](#2-normalized-device-coordinates-ndc)
3. [Texture Coordinate System](#3-texture-coordinate-system)
4. [Transformation Matrix Conventions](#4-transformation-matrix-conventions)
5. [Coordinate Space Summary Table](#5-coordinate-space-summary-table)
6. [Pitfalls and Cautions](#6-pitfalls-and-cautions)

---

## 1. World Coordinate System

Aether Engine uses a **Y-Up, Right-Handed** coordinate system, which is the default convention for the `glam` crate in Rust.

### Axis Definition

| Axis | Direction        | Description              |
|------|------------------|--------------------------|
| +X   | Right            | Positive horizontal axis |
| +Y   | Up               | Positive vertical axis   |
| +Z   | Toward viewer    | Out of the screen        |

### Visual Reference

```
        +Y (Up)
         |
         |
         |_______ +X (Right)
        /
       /
      +Z (Forward/Toward Viewer)
```

### Cross Product Rule

For a right-handed system: `X cross Y = Z`

```rust
use glam::Vec3;

let x = Vec3::X;
let y = Vec3::Y;
let z = x.cross(y);
assert_eq!(z, Vec3::Z);
```

---

## 2. Normalized Device Coordinates (NDC)

WebGPU uses a specific NDC convention that differs from OpenGL.

### WebGPU NDC Ranges

| Axis | Range      | Notes                                    |
|------|------------|------------------------------------------|
| X    | [-1, +1]   | -1 is left edge, +1 is right edge        |
| Y    | [-1, +1]   | -1 is bottom edge, +1 is top edge        |
| Z    | [0, 1]     | 0 is near plane, 1 is far plane          |

### Comparison with Other APIs

| API     | X Range   | Y Range   | Z Range   |
|---------|-----------|-----------|-----------|
| WebGPU  | [-1, +1]  | [-1, +1]  | [0, 1]    |
| OpenGL  | [-1, +1]  | [-1, +1]  | [-1, +1]  |
| Vulkan  | [-1, +1]  | [-1, +1]  | [0, 1]    |
| DirectX | [-1, +1]  | [-1, +1]  | [0, 1]    |

### Projection Matrix Considerations

When constructing projection matrices, use `glam`'s WebGPU-compatible functions:

```rust
use glam::Mat4;

// Perspective projection for WebGPU (Z: 0 to 1, Y-up)
let projection = Mat4::perspective_rh(
    fov_y_radians,  // Vertical field of view
    aspect_ratio,   // Width / Height
    z_near,         // Near plane (must be > 0)
    z_far,          // Far plane
);
```

> [!IMPORTANT]
> Do NOT use `perspective_lh` or OpenGL-style projection matrices. WebGPU requires reversed-Z or standard [0,1] depth range.

---

## 3. Texture Coordinate System

### UV Origin Convention

Aether Engine uses **Top-Left origin** for texture coordinates, matching WebGPU's default behavior.

| Coordinate | Position    |
|------------|-------------|
| (0, 0)     | Top-Left    |
| (1, 0)     | Top-Right   |
| (0, 1)     | Bottom-Left |
| (1, 1)     | Bottom-Right|

### Visual Reference

```
(0,0) -------- (1,0)
  |              |
  |   Texture    |
  |              |
(0,1) -------- (1,1)
```

### Comparison with OpenGL

| API     | UV Origin    |
|---------|--------------|
| WebGPU  | Top-Left     |
| OpenGL  | Bottom-Left  |
| Vulkan  | Top-Left     |
| DirectX | Top-Left     |

### Image Loading Considerations

When loading images from common formats (PNG, JPEG), the data is typically stored with the first pixel at the top-left, which matches WebGPU's expectation. **No vertical flip is required.**

```rust
// Pseudo-code: Load image without flipping
let image_data = load_image("texture.png");
// Data is already in correct orientation for WebGPU
queue.write_texture(/* ... */, &image_data, /* ... */);
```

---

## 4. Transformation Matrix Conventions

### Matrix Order

Aether Engine uses **column-major** matrices with **post-multiplication** (vector on the right):

```
transformed_point = Model * View * Projection * point
                  = MVP * point
```

### Transform Composition Order

When combining transforms, apply in this order (right to left in multiplication):

1. Scale
2. Rotation
3. Translation

```rust
use glam::{Mat4, Vec3, Quat};

let transform = Mat4::from_scale_rotation_translation(
    scale,       // Vec3
    rotation,    // Quat
    translation, // Vec3
);
// Equivalent to: Translation * Rotation * Scale
```

### View Matrix Construction

```rust
use glam::{Mat4, Vec3};

let view = Mat4::look_at_rh(
    eye,     // Camera position
    target,  // Look-at point
    up,      // Up vector (typically Vec3::Y)
);
```

---

## 5. Coordinate Space Summary Table

| Space          | Handedness   | Y Direction | Z Range  | Origin         |
|----------------|--------------|-------------|----------|----------------|
| World          | Right-handed | Up          | N/A      | World center   |
| View/Camera    | Right-handed | Up          | N/A      | Camera position|
| Clip           | Right-handed | Up          | [0, w]   | Projected      |
| NDC            | Right-handed | Up          | [0, 1]   | Screen center  |
| Screen/Pixel   | N/A          | Down        | N/A      | Top-Left       |
| Texture UV     | N/A          | Down        | N/A      | Top-Left       |

---

## 6. Pitfalls and Cautions

> [!CAUTION]
> **Critical Issues to Avoid**

### 6.1. Depth Buffer Range Mismatch

**Problem:** Using OpenGL-style projection matrices with Z range [-1, 1] instead of WebGPU's [0, 1].

**Symptom:** Objects render incorrectly, depth testing fails, or nothing appears on screen.

**Solution:** Always use `_rh` (right-handed) projection functions from `glam` which produce [0, 1] depth range:
- `Mat4::perspective_rh()`
- `Mat4::orthographic_rh()`

---

### 6.2. Texture Flip on Load

**Problem:** Flipping textures vertically when loading, assuming OpenGL conventions.

**Symptom:** Textures appear upside-down.

**Solution:** Do NOT flip images. WebGPU expects top-left origin, which matches standard image file formats.

---

### 6.3. Incorrect Cross Product Handedness

**Problem:** Computing normals or tangent-space basis with left-handed cross product order.

**Symptom:** Normals point inward, lighting appears inverted.

**Solution:** Verify cross product order. For outward-facing normals on a counter-clockwise wound triangle:

```rust
let edge1 = v1 - v0;
let edge2 = v2 - v0;
let normal = edge1.cross(edge2).normalize(); // Correct for CCW winding
```

---

### 6.4. Screen Space Y-Axis Confusion

**Problem:** Confusing world-space Y-up with screen-space Y-down.

**Symptom:** UI elements or screen-space effects appear inverted.

**Solution:** Remember that after projection to screen space, Y increases downward. Apply appropriate sign flips for screen-space calculations:

```rust
// Convert NDC to screen coordinates
let screen_x = (ndc_x + 1.0) * 0.5 * screen_width;
let screen_y = (1.0 - ndc_y) * 0.5 * screen_height; // Note: 1.0 - ndc_y
```

---

### 6.5. Matrix Multiplication Order

**Problem:** Applying transforms in incorrect order (e.g., translating before scaling).

**Symptom:** Objects appear at wrong positions or scales.

**Solution:** Always compose transforms as: `Translation * Rotation * Scale * vertex`

```rust
// Correct: TRS order
let model = Mat4::from_translation(pos) 
          * Mat4::from_quat(rot) 
          * Mat4::from_scale(scale);
```

---

## 7. Winding Order and Culling

### Winding Order

Aether Engine uses **Counter-Clockwise (CCW)** as the front face definition.

| Property       | Value                |
|----------------|----------------------|
| **Front Face** | CCW (Counter-Clockwise) |
| **Cull Mode**  | Back (Back-facing triangles are culled) |

### WebGPU Configuration

```rust
let rasterization_state = wgpu::PrimitiveState {
    topology: wgpu::PrimitiveTopology::TriangleList,
    front_face: wgpu::FrontFace::Ccw,  // Counter-clockwise is front
    cull_mode: Some(wgpu::Face::Back), // Cull back faces
    ..Default::default()
};
```

### Visual Reference

When looking at a triangle from the front (the visible side), vertices should be ordered counter-clockwise:

```
        v1
       /  \
      /    \
     /  CCW \
    /   -->  \
   v0 ------- v2

Index order: [0, 1, 2] = Front-facing (visible)
Index order: [0, 2, 1] = Back-facing (culled)
```

> [!IMPORTANT]
> Ensure all procedural meshes (terrain, particles) generate indices in CCW order relative to the camera. Incorrect winding will result in invisible geometry due to back-face culling.

### Normal Computation from Winding

For CCW-wound triangles, compute the outward-facing normal using:

```rust
let edge1 = v1 - v0;
let edge2 = v2 - v0;
let normal = edge1.cross(edge2).normalize(); // Points toward viewer for CCW
```

---

## References

- [WebGPU Specification - Coordinate Systems](https://www.w3.org/TR/webgpu/#coordinate-systems)
- [glam Documentation](https://docs.rs/glam/latest/glam/)
- [wgpu Coordinate System Notes](https://github.com/gfx-rs/wgpu)
