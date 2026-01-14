# Task 05: Compute Shader Particles

| Status | Priority | Depends On |
|--------|----------|------------|
| Pending | High | Task 04 (3D Camera) |

## Goal

Implement a Compute Shader that modifies a Storage Buffer (particle positions) in real-time, and render that buffer as a grid of points. This establishes the "Compute -> Storage -> Render" pipeline that is the foundation for GPU-driven terrain generation (see `terrain_compute.md`).

---

## Relevant Documentation

| Document | Purpose |
|----------|---------|
| [terrain_compute.md](../03_SUBSYSTEMS/terrain_compute.md) | Reference for compute shader dispatching and storage buffer patterns |
| [data_layout.md](../01_STANDARDS/data_layout.md) | Storage buffer alignment (std430) |

---

## Implementation Steps

### Step 1: Shader Asset

Create `assets/shaders/particles.wgsl`.

**Requirements:**
- Define a `Particle` struct with `position: vec4<f32>` (16-byte aligned).
- Define a `ParticleUniforms` struct with `time: f32` and `grid_size: u32`.
- Implement Compute Shader that animates particle Y positions using sine wave.
- Implement Vertex Shader that reads directly from the Storage Buffer.

**Shader Structure:**

```wgsl
// ============================================================================
// Data Structures
// ============================================================================

struct Particle {
    position: vec4<f32>,  // xyz = position, w = unused (padding)
}

struct ParticleUniforms {
    time: f32,
    grid_size: u32,
    _pad0: u32,
    _pad1: u32,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
}

// ============================================================================
// Bindings - Compute Shader
// ============================================================================

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(1) @binding(0)
var<uniform> uniforms: ParticleUniforms;

// ============================================================================
// Compute Shader
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_size = uniforms.grid_size;
    
    // Bounds check
    if (gid.x >= grid_size || gid.y >= grid_size) {
        return;
    }
    
    // Calculate particle index
    let index = gid.x + gid.y * grid_size;
    
    // Calculate world position
    let half_size = f32(grid_size) * 0.5;
    let spacing = 0.1;  // Distance between particles
    
    let x = (f32(gid.x) - half_size) * spacing;
    let z = (f32(gid.y) - half_size) * spacing;
    
    // Animate Y position with sine wave
    let y = sin(x * 2.0 + uniforms.time) * cos(z * 2.0 + uniforms.time) * 0.5;
    
    // Write to storage buffer
    particles[index].position = vec4<f32>(x, y, z, 1.0);
}

// ============================================================================
// Bindings - Vertex/Fragment Shaders
// ============================================================================

@group(0) @binding(0)
var<storage, read> particles_read: array<Particle>;

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

// ============================================================================
// Vertex Shader
// ============================================================================

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let particle = particles_read[vertex_index];
    
    var out: VertexOutput;
    out.clip_position = camera.view_proj * particle.position;
    
    // Color based on height (y position)
    let height_normalized = (particle.position.y + 0.5) / 1.0;
    out.color = vec3<f32>(
        height_normalized,
        0.5 + height_normalized * 0.5,
        1.0 - height_normalized * 0.5
    );
    
    return out;
}

// ============================================================================
// Fragment Shader
// ============================================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
```

**Critical Notes:**
- Compute shader uses separate bind group layout from render shaders.
- Storage buffer is bound as `read_write` in compute, `read` in vertex.
- Workgroup size `8x8x1` = 64 invocations per workgroup.

---

### Step 2: Particle System

Create `src/core/renderer/particles.rs`.

**Requirements:**
- Define particle constants and structs.
- Create Storage Buffer with `STORAGE | VERTEX | COPY_DST` usage.
- Create Compute Pipeline.
- Create Render Pipeline for point rendering.

**Constants:**
```rust
/// Grid dimension (GRID_SIZE x GRID_SIZE = total particles)
pub const GRID_SIZE: u32 = 100;

/// Total number of particles
pub const PARTICLE_COUNT: u32 = GRID_SIZE * GRID_SIZE; // 10,000
```

**Particle Struct (CPU):**
```rust
use bytemuck::{Pod, Zeroable};

/// GPU-side particle data.
/// 
/// Memory Layout (16 bytes):
/// - position: vec4<f32> (16 bytes, 16-byte aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Particle {
    pub position: [f32; 4],  // xyz = position, w = padding
}

impl Particle {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: [x, y, z, 1.0],
        }
    }
}

// Verify size at compile time
const _: () = assert!(std::mem::size_of::<Particle>() == 16);
```

**Particle Uniforms:**
```rust
/// Uniforms for particle compute shader.
///
/// Memory Layout (16 bytes):
/// - time: f32 (4 bytes)
/// - grid_size: u32 (4 bytes)
/// - _pad: [u32; 2] (8 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ParticleUniforms {
    pub time: f32,
    pub grid_size: u32,
    pub _pad: [u32; 2],
}

// Verify alignment
const _: () = assert!(std::mem::size_of::<ParticleUniforms>() == 16);
```

**Particle System Struct:**
```rust
pub struct ParticleSystem {
    /// Storage buffer containing particle positions.
    pub particle_buffer: wgpu::Buffer,
    /// Uniform buffer for compute shader parameters.
    pub uniform_buffer: wgpu::Buffer,
    /// Compute pipeline for particle animation.
    pub compute_pipeline: wgpu::ComputePipeline,
    /// Bind group for compute shader (storage buffer).
    pub compute_bind_group_0: wgpu::BindGroup,
    /// Bind group for compute shader (uniforms).
    pub compute_bind_group_1: wgpu::BindGroup,
    /// Bind group for render shader (storage buffer, read-only).
    pub render_bind_group_0: wgpu::BindGroup,
    /// Render pipeline for point drawing.
    pub render_pipeline: wgpu::RenderPipeline,
    /// Current uniforms state.
    pub uniforms: ParticleUniforms,
}
```

**Buffer Creation:**
```rust
// Storage buffer (initially zero-filled, compute shader populates it)
let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("Particle Storage Buffer"),
    size: (std::mem::size_of::<Particle>() * PARTICLE_COUNT as usize) as u64,
    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});

// Uniform buffer
let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("Particle Uniform Buffer"),
    contents: bytemuck::cast_slice(&[uniforms]),
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
});
```

---

### Step 3: Bind Group Layouts

**Compute Bind Group Layout 0 (Storage Buffer):**
```rust
let compute_storage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("Compute Storage Bind Group Layout"),
    entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }],
});
```

**Compute Bind Group Layout 1 (Uniforms):**
```rust
let compute_uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("Compute Uniform Bind Group Layout"),
    entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }],
});
```

**Render Bind Group Layout 0 (Storage Buffer, Read-Only):**
```rust
let render_storage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("Render Storage Bind Group Layout"),
    entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }],
});
```

---

### Step 4: Compute Pipeline

```rust
let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("Particle Compute Shader"),
    source: wgpu::ShaderSource::Wgsl(
        include_str!("../../../assets/shaders/particles.wgsl").into(),
    ),
});

let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    label: Some("Particle Compute Pipeline Layout"),
    bind_group_layouts: &[&compute_storage_layout, &compute_uniform_layout],
    push_constant_ranges: &[],
});

let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some("Particle Compute Pipeline"),
    layout: Some(&compute_pipeline_layout),
    module: &compute_shader,
    entry_point: Some("cs_main"),
    compilation_options: wgpu::PipelineCompilationOptions::default(),
    cache: None,
});
```

---

### Step 5: Render Pipeline

```rust
let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("Particle Render Shader"),
    source: wgpu::ShaderSource::Wgsl(
        include_str!("../../../assets/shaders/particles.wgsl").into(),
    ),
});

let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    label: Some("Particle Render Pipeline Layout"),
    // Group 0: Storage buffer (read-only)
    // Group 1: Camera uniform (reuse existing camera bind group layout)
    bind_group_layouts: &[&render_storage_layout, &camera_bind_group_layout],
    push_constant_ranges: &[],
});

let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    label: Some("Particle Render Pipeline"),
    layout: Some(&render_pipeline_layout),
    vertex: wgpu::VertexState {
        module: &render_shader,
        entry_point: Some("vs_main"),
        buffers: &[],  // No vertex buffers - reading from storage
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    },
    fragment: Some(wgpu::FragmentState {
        module: &render_shader,
        entry_point: Some("fs_main"),
        targets: &[Some(wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })],
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    }),
    primitive: wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::PointList,  // Draw points
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,  // No culling for points
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

---

### Step 6: Renderer Integration

Update `src/core/renderer/mod.rs`.

**Initialization:**
```rust
// Add to Renderer struct
particle_system: ParticleSystem,

// In Renderer::new()
let particle_system = ParticleSystem::new(
    &ctx.device,
    ctx.config.format,
    &camera_bind_group_layout,
);
log::info!("Particle system initialized ({} particles)", PARTICLE_COUNT);
```

**Render Loop Update:**
```rust
pub fn render(&mut self, state: &GameState, _alpha: f32) -> Result<(), wgpu::SurfaceError> {
    // ... acquire frame, create encoder ...

    // Update particle uniforms
    self.particle_system.uniforms.time = state.time;
    self.ctx.queue.write_buffer(
        &self.particle_system.uniform_buffer,
        0,
        bytemuck::cast_slice(&[self.particle_system.uniforms]),
    );

    // Update camera uniform
    let view_proj = self.camera.build_view_projection_matrix();
    self.camera_uniform.update_view_proj(view_proj);
    self.ctx.queue.write_buffer(
        &self.camera_buffer,
        0,
        bytemuck::cast_slice(&[self.camera_uniform]),
    );

    // ==========================================
    // Compute Pass - Animate particles
    // ==========================================
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Particle Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.particle_system.compute_pipeline);
        compute_pass.set_bind_group(0, &self.particle_system.compute_bind_group_0, &[]);
        compute_pass.set_bind_group(1, &self.particle_system.compute_bind_group_1, &[]);

        // Dispatch workgroups: ceil(GRID_SIZE / 8)
        let workgroups = (GRID_SIZE + 7) / 8;
        compute_pass.dispatch_workgroups(workgroups, workgroups, 1);
    }

    // ==========================================
    // Render Pass - Draw particles
    // ==========================================
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

        // Draw particles
        render_pass.set_pipeline(&self.particle_system.render_pipeline);
        render_pass.set_bind_group(0, &self.particle_system.render_bind_group_0, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.draw(0..PARTICLE_COUNT, 0..1);
    }

    // ... submit and present ...
}
```

---

### Step 7: Camera Adjustment

Update camera position for better particle view:

```rust
// In Camera::new()
Self {
    eye: Vec3::new(0.0, 8.0, 12.0),  // Higher and further back
    target: Vec3::ZERO,
    // ...
}
```

---

## File Structure After Completion

```
assets/
  shaders/
    triangle.wgsl         # (unchanged)
    cube.wgsl             # (unchanged)
    particles.wgsl        # NEW
src/
  core/
    mod.rs                # (unchanged)
    time.rs               # (unchanged)
    renderer/
      mod.rs              # MODIFIED (add compute pass, particle rendering)
      context.rs          # (unchanged)
      geometry.rs         # (unchanged)
      camera.rs           # MODIFIED (camera position)
      particles.rs        # NEW
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
   - A grid of 10,000 points should appear on screen.
   - Points should form a wave pattern (sine-cosine surface).
   - Wave should animate smoothly over time.

2. **Performance Verification:**
   - Frame rate should remain at 60Hz or display refresh rate.
   - No stuttering or frame drops despite 10,000 dynamic particles.
   - GPU utilization should be minimal (compute is efficient).

3. **Depth Test Verification:**
   - Particles closer to camera should occlude distant particles.

4. **Color Verification:**
   - Particles at wave peaks (high Y) should be one color.
   - Particles at wave troughs (low Y) should be another color.
   - Smooth color gradient between.

### Expected Visual Result

```
Top View (conceptual):
+----------------------------------+
|   . . . . . . . . . . . . . .   |
|  . . . . . . . . . . . . . . .  |
| . . . . . . . . . . . . . . . . |
|. . . . . . . . . . . . . . . . .|
| . . . . . . . . . . . . . . . . |
|  . . . . . . . . . . . . . . .  |
|   . . . . . . . . . . . . . .   |
+----------------------------------+

Side View (animated):
      ~     ~     ~
   ~     ~     ~     ~
~     ~     ~     ~     ~
```

---

## Workgroup Dispatch Calculation

For a 100x100 grid with workgroup size 8x8:

```
dispatch_x = ceil(100 / 8) = 13
dispatch_y = ceil(100 / 8) = 13
dispatch_z = 1

Total workgroups: 13 * 13 * 1 = 169
Total invocations: 169 * 64 = 10,816
Active invocations: 10,000 (rest are bounds-checked out)
```

---

## Acceptance Criteria

- [ ] `particles.wgsl` contains compute and render shaders.
- [ ] `Particle` struct is 16-byte aligned (vec4).
- [ ] Storage buffer created with `STORAGE | VERTEX | COPY_DST`.
- [ ] Compute pipeline uses separate bind group layouts.
- [ ] Render pipeline uses `PrimitiveTopology::PointList`.
- [ ] Vertex shader reads from storage buffer via `vertex_index`.
- [ ] Compute pass dispatches before render pass.
- [ ] Wave animation responds to `state.time`.
- [ ] 10,000 particles render at 60+ FPS.
- [ ] Depth testing works for particles.
- [ ] Builds successfully on both Native and WASM targets.

---

## Future Impact Analysis

| Future Feature | Compatibility Check | Status |
|----------------|---------------------|--------|
| Terrain Compute | Storage buffer pattern identical | [ ] Verified |
| Indirect Draw | Can add DrawIndirect buffer to pattern | [ ] Verified |
| GPU Culling | Compute dispatch pattern extensible | [ ] Verified |
| Particle Physics | Storage buffer can store velocity/acceleration | [ ] Verified |

---

## Performance Notes

### Why This Approach is Fast

1. **Zero CPU-GPU Data Transfer:** Particle positions are generated on GPU, never touched by CPU.
2. **No Vertex Buffer Binding:** Storage buffer is read directly in vertex shader.
3. **Parallel Compute:** 10,000 particles updated in parallel across GPU cores.
4. **Minimal Draw Calls:** Single draw call for all 10,000 particles.

### Scaling Expectations

| Particle Count | Expected Performance |
|----------------|---------------------|
| 10,000 | 60+ FPS on all GPUs |
| 100,000 | 60+ FPS on mid-range GPUs |
| 1,000,000 | 60+ FPS on high-end GPUs |

---

## References

- [WebGPU Compute Shaders](https://www.w3.org/TR/webgpu/#compute-pipeline)
- [WGSL Storage Buffers](https://www.w3.org/TR/WGSL/#buffer-storage-class)
- [wgpu Examples - Compute](https://github.com/gfx-rs/wgpu/tree/trunk/examples)
