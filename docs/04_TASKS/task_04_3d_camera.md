# Task 04: 3D Camera and Rotating Cube

| Status | Priority | Depends On |
|--------|----------|------------|
| Pending | High | Task 03 (Hello Triangle) |

## Goal

Implement a 3D Perspective Camera and render a rotating cube to verify depth testing, matrix transformations, and the complete 3D rendering pipeline. This establishes the foundation for all future 3D rendering in Aether Engine.

---

## Relevant Documentation

| Document | Purpose |
|----------|---------|
| [coordinate_systems.md](../01_STANDARDS/coordinate_systems.md) | Y-Up convention, `perspective_rh`, winding order |
| [data_layout.md](../01_STANDARDS/data_layout.md) | Uniform buffer alignment (mat4x4 = 64 bytes) |

---

## Implementation Steps

### Step 1: Camera Uniforms

Create `src/core/renderer/camera.rs`.

**Requirements:**
- Define `CameraUniform` struct for GPU-side data.
- Define `Camera` struct for CPU-side camera management.
- Implement view-projection matrix calculation using `glam`.

**CameraUniform Struct (GPU):**
```rust
use bytemuck::{Pod, Zeroable};

/// GPU-side camera uniform data.
/// Contains the combined view-projection matrix for vertex transformation.
///
/// Memory Layout (64 bytes):
/// - `view_proj`: mat4x4<f32> at offset 0 (64 bytes, 16-byte aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    /// Combined view-projection matrix.
    /// Transforms world-space positions to clip-space.
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

    /// Updates the uniform with a new view-projection matrix.
    pub fn update_view_proj(&mut self, view_proj: glam::Mat4) {
        self.view_proj = view_proj.to_cols_array_2d();
    }
}
```

**Camera Struct (CPU):**
```rust
use glam::{Mat4, Vec3};

/// CPU-side camera for managing view and projection.
pub struct Camera {
    /// Camera position in world space.
    pub eye: Vec3,
    /// Point the camera is looking at.
    pub target: Vec3,
    /// Up vector (typically Y-up per coordinate_systems.md).
    pub up: Vec3,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Vertical field of view in radians.
    pub fovy: f32,
    /// Near clipping plane (must be > 0).
    pub znear: f32,
    /// Far clipping plane.
    pub zfar: f32,
}

impl Camera {
    /// Creates a new camera with default parameters.
    pub fn new(aspect: f32) -> Self {
        Self {
            eye: Vec3::new(0.0, 2.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            aspect,
            fovy: 45.0_f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
        }
    }

    /// Builds the combined view-projection matrix.
    ///
    /// Uses right-handed coordinate system with Z range [0, 1]
    /// per WebGPU requirements (coordinate_systems.md Section 2).
    pub fn build_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        proj * view
    }
}
```

**Critical Constraints:**
| Requirement | Implementation |
|-------------|----------------|
| Right-handed system | `Mat4::look_at_rh()` |
| Z range [0, 1] | `Mat4::perspective_rh()` (WebGPU compatible) |
| Y-Up convention | `up: Vec3::Y` |
| 16-byte alignment | `mat4x4` is naturally 16-byte aligned |

---

### Step 2: Cube Shader

Create `assets/shaders/cube.wgsl`.

**Requirements:**
- Add uniform binding for camera matrix (Group 0, Binding 0).
- Transform vertex positions using view-projection matrix.
- Color faces distinctively for visual verification.

**Shader Structure:**
```wgsl
// ============================================================================
// Camera Uniform
// ============================================================================

struct CameraUniform {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// ============================================================================
// Vertex Stage
// ============================================================================

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Transform from local space to clip space
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

// ============================================================================
// Fragment Stage
// ============================================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
```

---

### Step 3: Cube Geometry

Update `src/core/renderer/geometry.rs`.

**Requirements:**
- Define 24 vertices (4 per face x 6 faces) to allow distinct face colors.
- Define 36 indices (6 indices per face x 6 faces).
- Maintain CCW winding order for front-facing triangles.

**Cube Vertex Data:**
```rust
/// Cube vertices with per-face colors.
/// 24 vertices (4 per face) to allow hard edges and distinct colors.
/// Centered at origin, size 1.0 (-0.5 to +0.5 on each axis).
pub const CUBE_VERTICES: &[Vertex] = &[
    // Front face (Z+) - Red
    Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0] },
    Vertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [1.0, 0.0, 0.0] },
    Vertex { position: [-0.5,  0.5,  0.5], color: [1.0, 0.0, 0.0] },

    // Back face (Z-) - Cyan
    Vertex { position: [ 0.5, -0.5, -0.5], color: [0.0, 1.0, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.0, 1.0, 1.0] },
    Vertex { position: [-0.5,  0.5, -0.5], color: [0.0, 1.0, 1.0] },
    Vertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 1.0, 1.0] },

    // Top face (Y+) - Green
    Vertex { position: [-0.5,  0.5,  0.5], color: [0.0, 1.0, 0.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.0, 1.0, 0.0] },
    Vertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 1.0, 0.0] },
    Vertex { position: [-0.5,  0.5, -0.5], color: [0.0, 1.0, 0.0] },

    // Bottom face (Y-) - Magenta
    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 0.0, 1.0] },
    Vertex { position: [ 0.5, -0.5, -0.5], color: [1.0, 0.0, 1.0] },
    Vertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 0.0, 1.0] },
    Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 0.0, 1.0] },

    // Right face (X+) - Blue
    Vertex { position: [ 0.5, -0.5,  0.5], color: [0.0, 0.0, 1.0] },
    Vertex { position: [ 0.5, -0.5, -0.5], color: [0.0, 0.0, 1.0] },
    Vertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 0.0, 1.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.0, 0.0, 1.0] },

    // Left face (X-) - Yellow
    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 1.0, 0.0] },
    Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 1.0, 0.0] },
    Vertex { position: [-0.5,  0.5,  0.5], color: [1.0, 1.0, 0.0] },
    Vertex { position: [-0.5,  0.5, -0.5], color: [1.0, 1.0, 0.0] },
];

/// Cube indices for indexed drawing.
/// Each face uses 2 triangles (6 indices) in CCW winding order.
pub const CUBE_INDICES: &[u16] = &[
    // Front face
    0, 1, 2, 2, 3, 0,
    // Back face
    4, 5, 6, 6, 7, 4,
    // Top face
    8, 9, 10, 10, 11, 8,
    // Bottom face
    12, 13, 14, 14, 15, 12,
    // Right face
    16, 17, 18, 18, 19, 16,
    // Left face
    20, 21, 22, 22, 23, 20,
];

pub const CUBE_VERTEX_COUNT: u32 = 24;
pub const CUBE_INDEX_COUNT: u32 = 36;
```

**Face Color Reference:**
| Face | Direction | Color |
|------|-----------|-------|
| Front | +Z | Red |
| Back | -Z | Cyan |
| Top | +Y | Green |
| Bottom | -Y | Magenta |
| Right | +X | Blue |
| Left | -X | Yellow |

---

### Step 4: Renderer Update

Update `src/core/renderer/mod.rs`.

**New Resources Required:**
- Camera uniform buffer
- Bind group layout and bind group
- Index buffer for cube
- Updated render pipeline with bind group layout

**Bind Group Layout:**
```rust
let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("Camera Bind Group Layout"),
    entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }],
});
```

**Uniform Buffer Creation:**
```rust
let camera_uniform = CameraUniform::new();
let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("Camera Uniform Buffer"),
    contents: bytemuck::cast_slice(&[camera_uniform]),
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
});
```

**Bind Group Creation:**
```rust
let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("Camera Bind Group"),
    layout: &camera_bind_group_layout,
    entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: camera_buffer.as_entire_binding(),
    }],
});
```

**Index Buffer Creation:**
```rust
let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("Cube Index Buffer"),
    contents: bytemuck::cast_slice(CUBE_INDICES),
    usage: wgpu::BufferUsages::INDEX,
});
```

**Pipeline Layout Update:**
```rust
let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    label: Some("Cube Pipeline Layout"),
    bind_group_layouts: &[&camera_bind_group_layout],
    push_constant_ranges: &[],
});
```

**Render Loop Update:**
```rust
// In render() method:

// Rotate cube over time (or orbit camera)
let rotation = glam::Mat4::from_rotation_y(state.time);
let model = rotation;

// Update camera view-projection
let view_proj = self.camera.build_view_projection_matrix() * model;
self.camera_uniform.update_view_proj(view_proj);

// Write updated uniform to GPU
self.ctx.queue.write_buffer(
    &self.camera_buffer,
    0,
    bytemuck::cast_slice(&[self.camera_uniform]),
);

// In render pass:
render_pass.set_pipeline(&self.cube_pipeline);
render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
render_pass.set_vertex_buffer(0, self.cube_vertex_buffer.slice(..));
render_pass.set_index_buffer(self.cube_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
render_pass.draw_indexed(0..CUBE_INDEX_COUNT, 0, 0..1);
```

---

### Step 5: Handle Aspect Ratio on Resize

**Requirements:**
- Update camera aspect ratio when window is resized.
- Prevent distortion by maintaining correct projection.

**Implementation:**
```rust
// In Renderer::resize()
pub fn resize(&mut self, width: u32, height: u32) {
    self.ctx.resize(width, height);
    
    // Update camera aspect ratio
    if height > 0 {
        self.camera.aspect = width as f32 / height as f32;
    }
}
```

---

## File Structure After Completion

```
assets/
  shaders/
    triangle.wgsl         # (unchanged)
    cube.wgsl             # NEW
src/
  core/
    mod.rs                # (unchanged)
    time.rs               # (unchanged)
    renderer/
      mod.rs              # MODIFIED (add camera, cube pipeline, index buffer)
      context.rs          # (unchanged)
      geometry.rs         # MODIFIED (add cube data)
      camera.rs           # NEW
  game/
    mod.rs                # (unchanged)
    state.rs              # (unchanged)
  main.rs                 # (unchanged)
```

---

## Verification

### Automated Checks

```bash
# Native build
cargo build

# WASM build
cargo build --target wasm32-unknown-unknown

# Run (Native)
cargo run
```

### Runtime Verification

1. **Visual Inspection:**
   - A colored cube should appear in the center of the screen.
   - The cube should rotate continuously (one full rotation per ~6.28 seconds at 1 rad/s).
   - All 6 faces should be visible during rotation with distinct colors.

2. **Depth Test Verification:**
   - Front faces should correctly occlude back faces.
   - No Z-fighting or face bleeding artifacts.

3. **Aspect Ratio Verification:**
   - Resize the window.
   - Cube should maintain correct proportions (not stretched/squashed).

4. **Coordinate System Verification:**
   - Top face (green, +Y) should face upward.
   - When cube rotation angle is 0, front face (red, +Z) should face the camera.

### Expected Visual Result

```
        +-------+
       /  Top  /|
      / Green /R|
     +-------+ i|
     | Front | g|
     |  Red  | h|
     |       | t|
     +-------+--+
```

---

## Acceptance Criteria

- [ ] `Camera` struct implements `perspective_rh` projection (Z range [0, 1]).
- [ ] `CameraUniform` uses `#[repr(C)]` and `bytemuck` for GPU mapping.
- [ ] Uniform buffer created with `UNIFORM | COPY_DST` usage.
- [ ] Bind group layout matches shader expectations (Group 0, Binding 0).
- [ ] Cube has 24 vertices with 6 distinct face colors.
- [ ] Index buffer uses `u16` indices with CCW winding.
- [ ] `draw_indexed()` used for efficient rendering.
- [ ] Cube rotates over time using `state.time`.
- [ ] Depth testing correctly occludes back faces.
- [ ] Aspect ratio updates on window resize.
- [ ] No projection distortion.
- [ ] Builds successfully on both Native and WASM targets.

---

## Future Impact Analysis

| Future Feature | Compatibility Check | Status |
|----------------|---------------------|--------|
| Multiple Objects | Bind group structure supports per-object transforms | [ ] Verified |
| Instanced Rendering | Can add instance buffer alongside vertex buffer | [ ] Verified |
| Shadow Mapping | Camera uniform structure can be reused for light matrices | [ ] Verified |
| Deferred Rendering | Uniform binding pattern extends to G-Buffer passes | [ ] Verified |

---

## References

- [WebGPU Uniform Buffers](https://www.w3.org/TR/webgpu/#buffer-binding)
- [glam Mat4 Documentation](https://docs.rs/glam/latest/glam/f32/struct.Mat4.html)
- [WGSL Uniform Variables](https://www.w3.org/TR/WGSL/#uniform-variables)
