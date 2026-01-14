# Task 03: Hello Triangle

| Status | Priority | Depends On |
|--------|----------|------------|
| Pending | High | Task 02 (Game Loop) |

## Goal

Implement the basic rendering pipeline to draw a multi-colored triangle using WGSL shaders and Vertex Buffers. This verifies the pipeline architecture and establishes the foundation for all future geometry rendering.

---

## Relevant Documentation

| Document | Purpose |
|----------|---------|
| [coordinate_systems.md](../01_STANDARDS/coordinate_systems.md) | Winding order (CCW), Y-Up convention, NDC ranges |
| [data_layout.md](../01_STANDARDS/data_layout.md) | Vertex struct alignment and bytemuck usage |

---

## Implementation Steps

### Step 1: Shader Asset

Create `assets/shaders/triangle.wgsl`.

**Requirements:**
- Define a Vertex Shader that:
  - Accepts `position: vec3<f32>` and `color: vec3<f32>` as vertex inputs.
  - Outputs clip-space position and passes color to fragment stage.
- Define a Fragment Shader that:
  - Receives interpolated color from vertex stage.
  - Outputs the final fragment color to render target 0.

**Shader Structure:**
```wgsl
// Vertex Input
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

// Vertex Output / Fragment Input
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
```

**Constraint:** Triangle vertices MUST be defined in **Counter-Clockwise (CCW)** order when viewed from the front, per `coordinate_systems.md` Section 7.

---

### Step 2: Vertex Structure

Create `src/core/renderer/geometry.rs`.

**Requirements:**
- Define `Vertex` struct with proper memory layout for GPU.
- Implement `Vertex::desc()` returning `wgpu::VertexBufferLayout`.
- Use `bytemuck` for safe memory mapping.

**Struct Definition:**
```rust
use bytemuck::{Pod, Zeroable};

/// Vertex with position and color attributes.
/// Memory layout: [x, y, z, r, g, b] = 24 bytes total.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    /// Vertex position in clip space or local space.
    pub position: [f32; 3],
    /// Vertex color (RGB, linear).
    pub color: [f32; 3],
}

impl Vertex {
    /// Returns the vertex buffer layout descriptor for the pipeline.
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position: vec3<f32>
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color: vec3<f32>
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}
```

**Alignment Notes (per `data_layout.md`):**
- `[f32; 3]` has 4-byte alignment and 12-byte size.
- Total struct size: 24 bytes (no padding required for vertex buffers).
- `#[repr(C)]` ensures predictable memory layout.

---

### Step 3: Pipeline Initialization

Update `src/core/renderer/mod.rs` (or create `src/core/renderer/pipeline.rs`).

**Requirements:**
- Load the shader source using `include_str!` for compile-time embedding.
- Create `wgpu::RenderPipeline` with AAA-standard configuration.
- Store pipeline and vertex buffer in Renderer struct.

**Pipeline Configuration:**
```rust
// Shader loading
let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("Triangle Shader"),
    source: wgpu::ShaderSource::Wgsl(include_str!("../../../assets/shaders/triangle.wgsl").into()),
});

// Pipeline creation
let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    label: Some("Triangle Pipeline"),
    layout: None, // Auto-layout for simple shaders
    vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        buffers: &[Vertex::desc()],
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    },
    fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        targets: &[Some(wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })],
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    }),
    primitive: wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,      // CCW is front (per standards)
        cull_mode: Some(wgpu::Face::Back),     // Cull back faces
        polygon_mode: wgpu::PolygonMode::Fill,
        unclipped_depth: false,
        conservative: false,
    },
    depth_stencil: Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }),
    multisample: wgpu::MultisampleState {
        count: 1,
        mask: !0,
        alpha_to_coverage_enabled: false,
    },
    multiview: None,
    cache: None,
});
```

**Critical Settings:**
| Setting | Value | Reason |
|---------|-------|--------|
| `front_face` | `Ccw` | Per `coordinate_systems.md` Section 7 |
| `cull_mode` | `Back` | Standard back-face culling |
| `depth_compare` | `Less` | Standard depth test (closer objects win) |
| `topology` | `TriangleList` | Standard triangle rendering |

---

### Step 4: Vertex Buffer Creation

**Triangle Vertices (CCW Order):**
```rust
// Define triangle vertices in Counter-Clockwise order.
// Positioned in clip space (NDC-like, Z=0 for simplicity).
//
//        v0 (top, red)
//        /\
//       /  \
//      /    \
//     /______\
//   v1        v2
// (bottom-left, green)  (bottom-right, blue)

const TRIANGLE_VERTICES: &[Vertex] = &[
    // Top vertex (red)
    Vertex {
        position: [0.0, 0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    // Bottom-left vertex (green)
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    // Bottom-right vertex (blue)
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
];
```

**Buffer Creation:**
```rust
use wgpu::util::DeviceExt;

let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("Triangle Vertex Buffer"),
    contents: bytemuck::cast_slice(TRIANGLE_VERTICES),
    usage: wgpu::BufferUsages::VERTEX,
});
```

---

### Step 5: Render Pass Update

Update `Renderer::render()` to draw the triangle.

**Drawing Commands:**
```rust
// Inside render pass (after setting up render_pass)
{
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Main Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear_color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &self.ctx.depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    // Draw triangle
    render_pass.set_pipeline(&self.pipeline);
    render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
    render_pass.draw(0..3, 0..1);
}
```

---

## File Structure After Completion

```
assets/
  shaders/
    triangle.wgsl           # NEW
src/
  core/
    mod.rs                  # (unchanged)
    time.rs                 # (unchanged)
    renderer/
      mod.rs                # MODIFIED (add pipeline, vertex buffer)
      context.rs            # (unchanged)
      geometry.rs           # NEW
  game/
    mod.rs                  # (unchanged)
    state.rs                # (unchanged)
  main.rs                   # (unchanged)
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

1. **Run the engine** (native or WASM).
2. **Visual Inspection:**
   - A multi-colored triangle (red top, green bottom-left, blue bottom-right) should appear.
   - Triangle is centered on the pulsing background from Task 02.
   - Triangle should NOT disappear or flicker (culling test passed).
3. **Color Interpolation:**
   - Colors should smoothly blend between vertices (hardware interpolation).

### Culling Test

The triangle MUST be visible. If invisible:
- Check vertex order is CCW (counter-clockwise).
- Verify `front_face: Ccw` in pipeline.
- Verify `cull_mode: Some(Face::Back)` in pipeline.

---

## Acceptance Criteria

- [ ] `triangle.wgsl` shader compiles without errors.
- [ ] `Vertex` struct correctly implements `Pod`, `Zeroable`, and `desc()`.
- [ ] Render pipeline uses CCW front face and back-face culling.
- [ ] Depth testing enabled with Depth32Float format.
- [ ] Vertex buffer created and bound correctly.
- [ ] Triangle renders with correct colors (red/green/blue vertices).
- [ ] Triangle is visible (not culled) proving correct winding order.
- [ ] Pulsing background still visible behind triangle.
- [ ] Builds successfully on both Native and WASM targets.

---

## Future Impact Analysis

| Future Feature | Compatibility Check | Status |
|----------------|---------------------|--------|
| Instanced Rendering | Vertex layout supports additional instance buffers | [ ] Verified |
| Skeletal Animation | Can extend Vertex with bone weights/indices | [ ] Verified |
| Deferred Rendering | Pipeline targets can be extended for G-Buffer | [ ] Verified |
| Material System | Pipeline layout can accept bind groups | [ ] Verified |

---

## References

- [WebGPU Vertex Buffers](https://www.w3.org/TR/webgpu/#vertex-buffers)
- [WGSL Specification](https://www.w3.org/TR/WGSL/)
- [wgpu Examples - Triangle](https://github.com/gfx-rs/wgpu/tree/trunk/examples)
