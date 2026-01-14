# Task 06: Terrain Mesh Generation

## Status: NOT STARTED

## Goal

Transform the particle grid (Task 05) into a solid terrain mesh by generating indices and normals on the GPU, and applying basic directional lighting. This establishes the foundation for the procedural terrain system.

## Priority: HIGH

## Estimated Effort: 4-6 hours

## Dependencies

* Task 05: GPU Compute Foundation (COMPLETED)
* `docs/01_STANDARDS/coordinate_systems.md` (Y-Up, Right-handed)
* `docs/01_STANDARDS/data_layout.md` (Buffer alignment)
* `docs/03_SUBSYSTEMS/renderer_deferred.md` (Reference for future lighting)

## Architecture Overview

```
+------------------+     +------------------+     +------------------+
|  Compute Pass 1  | --> |  Compute Pass 2  | --> |   Render Pass    |
|  Index Generation|     | Position/Normals |     |  draw_indexed()  |
|  (once at init)  |     |  (every frame)   |     |  TriangleList    |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
  Index Buffer            Vertex Buffer             Fragment Shader
  (99*99*6 u32)         (100*100 Vertex)          (Directional Light)
```

## Data Structures

### Vertex Structure (Updated from Particle)

```wgsl
struct Vertex {
    position: vec4<f32>,  // xyz = world position, w = 1.0
    normal: vec4<f32>,    // xyz = normal vector, w = 0.0
}
// Total: 32 bytes per vertex (aligned)
```

### Index Buffer

```
Total Quads: (GRID_SIZE - 1) * (GRID_SIZE - 1) = 99 * 99 = 9,801
Triangles per Quad: 2
Indices per Triangle: 3
Total Indices: 9,801 * 6 = 58,806
Buffer Size: 58,806 * 4 bytes = 235,224 bytes
```

### Uniforms (Extended)

```wgsl
struct TerrainUniforms {
    time: f32,
    grid_size: u32,
    light_dir: vec3<f32>,  // Normalized direction TO light
    _pad: u32,
}
```

## Implementation Steps

### Step 1: Shader Update (terrain.wgsl)

Rename `assets/shaders/particles.wgsl` to `assets/shaders/terrain.wgsl`.

#### 1.1 Update Vertex Structure

```wgsl
struct Vertex {
    position: vec4<f32>,
    normal: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(1) var<storage, read_write> indices: array<u32>;
```

#### 1.2 Compute Shader - Index Generation (cs_generate_indices)

Run once at initialization (or when grid size changes).

```wgsl
@compute @workgroup_size(8, 8, 1)
fn cs_generate_indices(@builtin(global_invocation_id) id: vec3<u32>) {
    let grid_size = uniforms.grid_size;
    let x = id.x;
    let z = id.y;
    
    // Only process interior cells (99x99 for 100x100 grid)
    if (x >= grid_size - 1u || z >= grid_size - 1u) {
        return;
    }
    
    // Calculate vertex indices for this quad
    let top_left = z * grid_size + x;
    let top_right = top_left + 1u;
    let bottom_left = (z + 1u) * grid_size + x;
    let bottom_right = bottom_left + 1u;
    
    // Calculate index buffer offset (6 indices per quad)
    let quad_index = z * (grid_size - 1u) + x;
    let base = quad_index * 6u;
    
    // Triangle 1: top-left, bottom-left, top-right (CCW)
    indices[base + 0u] = top_left;
    indices[base + 1u] = bottom_left;
    indices[base + 2u] = top_right;
    
    // Triangle 2: top-right, bottom-left, bottom-right (CCW)
    indices[base + 3u] = top_right;
    indices[base + 4u] = bottom_left;
    indices[base + 5u] = bottom_right;
}
```

#### 1.3 Compute Shader - Position and Normal Update (cs_update_terrain)

Run every frame to animate terrain.

```wgsl
// Helper function to calculate height at a grid position
fn get_height(x: f32, z: f32, time: f32) -> f32 {
    return sin(x * 0.5 + time) * cos(z * 0.5 + time) * 2.0;
}

@compute @workgroup_size(8, 8, 1)
fn cs_update_terrain(@builtin(global_invocation_id) id: vec3<u32>) {
    let grid_size = uniforms.grid_size;
    let x = id.x;
    let z = id.y;
    
    if (x >= grid_size || z >= grid_size) {
        return;
    }
    
    let index = z * grid_size + x;
    let time = uniforms.time;
    
    // World position (centered grid)
    let half_size = f32(grid_size) * 0.5;
    let world_x = f32(x) - half_size;
    let world_z = f32(z) - half_size;
    let world_y = get_height(world_x, world_z, time);
    
    // Calculate normal using finite differences
    let delta = 1.0;
    let height_px = get_height(world_x + delta, world_z, time);
    let height_pz = get_height(world_x, world_z + delta, time);
    
    // Tangent vectors
    let tangent_x = vec3<f32>(delta, height_px - world_y, 0.0);
    let tangent_z = vec3<f32>(0.0, height_pz - world_y, delta);
    
    // Normal = cross(tangent_z, tangent_x) for CCW winding
    let normal = normalize(cross(tangent_z, tangent_x));
    
    // Store vertex data
    vertices[index].position = vec4<f32>(world_x, world_y, world_z, 1.0);
    vertices[index].normal = vec4<f32>(normal, 0.0);
}
```

#### 1.4 Vertex Shader

```wgsl
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let vertex = vertices[vertex_index];
    
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vertex.position;
    out.world_normal = vertex.normal.xyz;
    out.world_position = vertex.position.xyz;
    return out;
}
```

#### 1.5 Fragment Shader with Directional Lighting

```wgsl
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalize interpolated normal
    let normal = normalize(in.world_normal);
    
    // Light direction (pointing towards light source)
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    
    // Ambient light
    let ambient = 0.2;
    
    // Diffuse lighting (Lambertian)
    let diffuse = max(dot(normal, light_dir), 0.0);
    
    // Base terrain color (green-brown gradient based on height)
    let height = in.world_position.y;
    let grass_color = vec3<f32>(0.2, 0.5, 0.15);
    let rock_color = vec3<f32>(0.4, 0.35, 0.3);
    let t = clamp((height + 2.0) / 4.0, 0.0, 1.0);
    let base_color = mix(rock_color, grass_color, t);
    
    // Final color
    let lighting = ambient + diffuse * 0.8;
    let final_color = base_color * lighting;
    
    return vec4<f32>(final_color, 1.0);
}
```

### Step 2: Renderer Update (terrain.rs)

Rename `src/core/renderer/particles.rs` to `src/core/renderer/terrain.rs`.

#### 2.1 Update Constants

```rust
pub const GRID_SIZE: u32 = 100;
pub const VERTEX_COUNT: u32 = GRID_SIZE * GRID_SIZE; // 10,000
pub const INDEX_COUNT: u32 = (GRID_SIZE - 1) * (GRID_SIZE - 1) * 6; // 58,806
```

#### 2.2 Update Vertex Structure

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TerrainVertex {
    pub position: [f32; 4], // 16 bytes
    pub normal: [f32; 4],   // 16 bytes
}
// Total: 32 bytes

const _: () = assert!(std::mem::size_of::<TerrainVertex>() == 32);
```

#### 2.3 Create Index Buffer

```rust
// Index buffer with STORAGE usage for compute shader write
let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("Terrain Index Buffer"),
    size: (std::mem::size_of::<u32>() * INDEX_COUNT as usize) as u64,
    usage: wgpu::BufferUsages::INDEX 
         | wgpu::BufferUsages::STORAGE 
         | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
```

#### 2.4 Create Pipelines

```rust
// Index generation pipeline (dispatched once)
let index_gen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some("Index Generation Pipeline"),
    layout: Some(&compute_pipeline_layout),
    module: &shader,
    entry_point: Some("cs_generate_indices"),
    // ...
});

// Terrain update pipeline (dispatched every frame)
let terrain_update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some("Terrain Update Pipeline"),
    layout: Some(&compute_pipeline_layout),
    module: &shader,
    entry_point: Some("cs_update_terrain"),
    // ...
});

// Render pipeline with TriangleList topology
let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    // ...
    primitive: wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: Some(wgpu::Face::Back),
        // ...
    },
    // ...
});
```

#### 2.5 TerrainSystem Struct

```rust
pub struct TerrainSystem {
    // Vertex storage buffer
    pub vertex_buffer: wgpu::Buffer,
    // Index storage/index buffer
    pub index_buffer: wgpu::Buffer,
    // Uniform buffer
    pub uniform_buffer: wgpu::Buffer,
    pub uniforms: TerrainUniforms,
    
    // Pipelines
    pub index_gen_pipeline: wgpu::ComputePipeline,
    pub terrain_update_pipeline: wgpu::ComputePipeline,
    pub render_pipeline: wgpu::RenderPipeline,
    
    // Bind groups
    pub compute_bind_group: wgpu::BindGroup,
    pub render_bind_group: wgpu::BindGroup,
    
    // Flags
    pub indices_generated: bool,
}
```

### Step 3: Rendering Integration (mod.rs)

#### 3.1 Compute Pass

```rust
// Compute pass
{
    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Terrain Compute Pass"),
        timestamp_writes: None,
    });
    
    // Generate indices (once)
    if !self.terrain_system.indices_generated {
        compute_pass.set_pipeline(&self.terrain_system.index_gen_pipeline);
        compute_pass.set_bind_group(0, &self.terrain_system.compute_bind_group, &[]);
        let workgroups = TerrainSystem::workgroup_count();
        compute_pass.dispatch_workgroups(workgroups, workgroups, 1);
        self.terrain_system.indices_generated = true;
    }
    
    // Update positions and normals (every frame)
    compute_pass.set_pipeline(&self.terrain_system.terrain_update_pipeline);
    compute_pass.set_bind_group(0, &self.terrain_system.compute_bind_group, &[]);
    compute_pass.set_bind_group(1, &self.terrain_system.uniform_bind_group, &[]);
    let workgroups = TerrainSystem::workgroup_count();
    compute_pass.dispatch_workgroups(workgroups, workgroups, 1);
}
```

#### 3.2 Render Pass

```rust
// Draw terrain mesh
render_pass.set_pipeline(&self.terrain_system.render_pipeline);
render_pass.set_bind_group(0, &self.terrain_system.render_bind_group, &[]);
render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
render_pass.set_index_buffer(
    self.terrain_system.index_buffer.slice(..),
    wgpu::IndexFormat::Uint32,
);
render_pass.draw_indexed(0..INDEX_COUNT, 0, 0..1);

// Draw cube (floating above terrain)
render_pass.set_pipeline(&self.pipeline);
render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
render_pass.draw_indexed(0..CUBE_INDEX_COUNT, 0, 0..1);
```

### Step 4: Module Updates

#### 4.1 Update mod.rs imports

```rust
pub mod terrain;  // Renamed from particles

use terrain::{TerrainSystem, INDEX_COUNT};
```

#### 4.2 Update Renderer struct

```rust
pub struct Renderer {
    // ...existing fields...
    terrain_system: TerrainSystem,  // Renamed from particle_system
}
```

## Verification Checklist

1. **Build Verification:**
   - [ ] `cargo build` succeeds without errors
   - [ ] `cargo build --target wasm32-unknown-unknown` succeeds

2. **Visual Verification:**
   - [ ] Solid waving surface appears (not dots)
   - [ ] Surface has proper shading (lighter on top, darker on slopes)
   - [ ] No visible gaps or holes in the mesh
   - [ ] Winding order is correct (no inside-out faces)

3. **Performance Verification:**
   - [ ] Smooth 60 FPS maintained
   - [ ] Index generation only happens once
   - [ ] Position/Normal update runs every frame

4. **Integration Verification:**
   - [ ] Rotating cube still renders above/within the terrain
   - [ ] Camera controls work correctly
   - [ ] Pulsing background still visible at horizon

## Technical Notes

### Normal Calculation

The normal is calculated using finite differences:
- Sample height at current position (h0)
- Sample height at x+1 (hx)
- Sample height at z+1 (hz)
- Create tangent vectors: T_x = (1, hx-h0, 0), T_z = (0, hz-h0, 1)
- Normal = normalize(cross(T_z, T_x))

The cross product order (T_z x T_x) produces an upward-facing normal for our CCW winding order.

### Index Buffer Generation

Each quad in the grid produces 2 triangles (6 indices):
```
TL----TR     Indices (CCW winding):
|    / |     Triangle 1: TL, BL, TR
|   /  |     Triangle 2: TR, BL, BR
|  /   |
| /    |
BL----BR
```

### Memory Layout

| Buffer | Size | Usage |
|--------|------|-------|
| Vertex Buffer | 320,000 bytes (10,000 x 32) | STORAGE, VERTEX |
| Index Buffer | 235,224 bytes (58,806 x 4) | STORAGE, INDEX |
| Uniform Buffer | 32 bytes | UNIFORM |

## Future Enhancements

* LOD (Level of Detail) system for distant terrain
* Frustum culling per terrain chunk
* Texture mapping (grass, rock, snow based on height/slope)
* Normal mapping for surface detail
* Shadow mapping integration (see renderer_deferred.md)

## Related Documents

* [Task 05: GPU Compute Foundation](task_05_compute_particles.md)
* [Terrain Compute System](../03_SUBSYSTEMS/terrain_compute.md)
* [Deferred Renderer](../03_SUBSYSTEMS/renderer_deferred.md)
* [Coordinate Systems](../01_STANDARDS/coordinate_systems.md)
