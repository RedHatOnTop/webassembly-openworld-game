# Terrain Compute Pipeline

Technical specification for Aether Engine's GPU-driven terrain generation using Dual Contouring.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Phase 1: Density Calculation](#2-phase-1-density-calculation)
3. [Phase 2: Vertex Generation](#3-phase-2-vertex-generation)
4. [Phase 3: Index Generation](#4-phase-3-index-generation)
5. [Indirect Drawing](#5-indirect-drawing)
6. [LOD Strategy](#6-lod-strategy)
7. [Buffer Layout](#7-buffer-layout)
8. [Pitfalls and Cautions](#8-pitfalls-and-cautions)

---

## 1. Overview

Aether Engine generates terrain meshes **entirely on the GPU** using compute shaders. The CPU does NOT generate mesh data.

### Why GPU-Driven?

| Approach     | Mesh Generation | Latency      | CPU Load |
|--------------|-----------------|--------------|----------|
| CPU-based    | CPU             | High (copies)| High     |
| GPU-driven   | GPU             | Minimal      | Near zero|

### Pipeline Stages

```mermaid
graph LR
    P1[Phase 1: Density] --> P2[Phase 2: Vertices]
    P2 --> P3[Phase 3: Indices]
    P3 --> DRAW[Indirect Draw]
```

---

## 2. Phase 1: Density Calculation

Calculate Signed Distance Field (SDF) values at each grid corner using noise functions.

### WGSL Shader

```wgsl
struct ChunkParams {
    origin: vec3<f32>,
    size: f32,
    resolution: u32,  // e.g., 32
    _pad: vec3<u32>,
}

@group(0) @binding(0) var<uniform> chunk: ChunkParams;
@group(0) @binding(1) var<storage, read_write> density: array<f32>;

fn simplex_noise_3d(p: vec3<f32>) -> f32 {
    // Simplex noise implementation
    // Returns value in range [-1, 1]
}

fn terrain_sdf(world_pos: vec3<f32>) -> f32 {
    // Base terrain: height-based
    let height_density = world_pos.y - 64.0;
    
    // Noise layers (FBM)
    var noise_val = 0.0;
    var freq = 0.02;
    var amp = 32.0;
    for (var i = 0u; i < 4u; i++) {
        noise_val += simplex_noise_3d(world_pos * freq) * amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    
    return height_density - noise_val;
}

@compute @workgroup_size(4, 4, 4)
fn cs_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = chunk.resolution + 1u;  // 33 corners for 32 cells
    if (any(gid >= vec3<u32>(res))) { return; }
    
    let cell_size = chunk.size / f32(chunk.resolution);
    let world_pos = chunk.origin + vec3<f32>(gid) * cell_size;
    
    let idx = gid.x + gid.y * res + gid.z * res * res;
    density[idx] = terrain_sdf(world_pos);
}
```

### Dispatch

```rust
// 33^3 corners for 32^3 cells
let dispatch = (33 + 3) / 4;  // = 9
pass.dispatch_workgroups(dispatch, dispatch, dispatch);
```

---

## 3. Phase 2: Vertex Generation

Detect sign changes (surface crossings) and generate vertices using QEF or averaging.

### Algorithm

For each cell with sign change:
1. Find edge intersections (where density crosses zero)
2. Compute average position (simple) or solve QEF (accurate)
3. Compute surface normal from density gradient

### WGSL Shader

```wgsl
struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
}

@group(0) @binding(0) var<storage, read> density: array<f32>;
@group(0) @binding(1) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read_write> vertex_count: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> cell_to_vertex: array<u32>;

const CORNER_OFFSETS: array<vec3<u32>, 8> = array<vec3<u32>, 8>(
    vec3<u32>(0u, 0u, 0u), vec3<u32>(1u, 0u, 0u),
    vec3<u32>(1u, 1u, 0u), vec3<u32>(0u, 1u, 0u),
    vec3<u32>(0u, 0u, 1u), vec3<u32>(1u, 0u, 1u),
    vec3<u32>(1u, 1u, 1u), vec3<u32>(0u, 1u, 1u),
);

fn get_density(cell: vec3<u32>, corner: u32) -> f32 {
    let pos = cell + CORNER_OFFSETS[corner];
    let res = chunk.resolution + 1u;
    return density[pos.x + pos.y * res + pos.z * res * res];
}

fn has_sign_change(cell: vec3<u32>) -> bool {
    let d0 = get_density(cell, 0u);
    for (var i = 1u; i < 8u; i++) {
        if ((d0 < 0.0) != (get_density(cell, i) < 0.0)) {
            return true;
        }
    }
    return false;
}

fn compute_vertex(cell: vec3<u32>) -> Vertex {
    // Simple averaging of edge intersections
    var sum_pos = vec3<f32>(0.0);
    var count = 0u;
    
    // Check all 12 edges
    for (var e = 0u; e < 12u; e++) {
        let c0 = EDGE_CORNERS[e].x;
        let c1 = EDGE_CORNERS[e].y;
        let d0 = get_density(cell, c0);
        let d1 = get_density(cell, c1);
        
        if ((d0 < 0.0) != (d1 < 0.0)) {
            let t = d0 / (d0 - d1);
            let p0 = get_corner_position(cell, c0);
            let p1 = get_corner_position(cell, c1);
            sum_pos += mix(p0, p1, t);
            count++;
        }
    }
    
    let position = sum_pos / f32(count);
    let normal = normalize(compute_gradient(position));
    
    return Vertex(position, normal);
}

@compute @workgroup_size(4, 4, 4)
fn cs_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= vec3<u32>(chunk.resolution))) { return; }
    
    let cell_idx = gid.x + gid.y * chunk.resolution + gid.z * chunk.resolution * chunk.resolution;
    
    if (!has_sign_change(gid)) {
        cell_to_vertex[cell_idx] = 0xFFFFFFFFu;  // Invalid
        return;
    }
    
    let vertex = compute_vertex(gid);
    let vertex_idx = atomicAdd(&vertex_count, 1u);
    vertices[vertex_idx] = vertex;
    cell_to_vertex[cell_idx] = vertex_idx;
}
```

---

## 4. Phase 3: Index Generation

Connect vertices to form quads. Each internal edge shared by 4 cells generates one quad.

### WGSL Shader

```wgsl
@group(0) @binding(0) var<storage, read> cell_to_vertex: array<u32>;
@group(0) @binding(1) var<storage, read_write> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirect: DrawIndexedIndirect;

struct DrawIndexedIndirect {
    index_count: atomic<u32>,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@compute @workgroup_size(4, 4, 4)
fn cs_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = chunk.resolution;
    if (any(gid >= vec3<u32>(res - 1u))) { return; }
    
    // For each edge direction (X, Y, Z), generate quad if all 4 cells have vertices
    
    // X-aligned edge (cells sharing this edge: [0,0,0], [0,1,0], [0,0,1], [0,1,1])
    {
        let v0 = cell_to_vertex[cell_index(gid)];
        let v1 = cell_to_vertex[cell_index(gid + vec3<u32>(0u, 1u, 0u))];
        let v2 = cell_to_vertex[cell_index(gid + vec3<u32>(0u, 1u, 1u))];
        let v3 = cell_to_vertex[cell_index(gid + vec3<u32>(0u, 0u, 1u))];
        
        if (v0 != 0xFFFFFFFFu && v1 != 0xFFFFFFFFu && 
            v2 != 0xFFFFFFFFu && v3 != 0xFFFFFFFFu) {
            emit_quad(v0, v1, v2, v3);
        }
    }
    
    // Similar for Y and Z edges...
}

fn emit_quad(v0: u32, v1: u32, v2: u32, v3: u32) {
    let base = atomicAdd(&indirect.index_count, 6u);
    indices[base + 0u] = v0;
    indices[base + 1u] = v1;
    indices[base + 2u] = v2;
    indices[base + 3u] = v0;
    indices[base + 4u] = v2;
    indices[base + 5u] = v3;
}
```

---

## 5. Indirect Drawing

The CPU does not know how many vertices/indices were generated. Use indirect drawing.

### IndirectArgs Buffer Structure

```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndexedIndirect {
    index_count: u32,      // Filled by GPU
    instance_count: u32,   // Set to 1
    first_index: u32,      // Set to 0
    base_vertex: i32,      // Set to 0
    first_instance: u32,   // Set to 0
}
```

### Rust Setup

```rust
let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("Indirect Args"),
    size: std::mem::size_of::<DrawIndexedIndirect>() as u64,
    usage: wgpu::BufferUsages::INDIRECT 
         | wgpu::BufferUsages::STORAGE 
         | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});

// Initialize with instance_count = 1
queue.write_buffer(&indirect_buffer, 0, bytemuck::bytes_of(&DrawIndexedIndirect {
    index_count: 0,
    instance_count: 1,
    first_index: 0,
    base_vertex: 0,
    first_instance: 0,
}));
```

### Draw Call

```rust
render_pass.draw_indexed_indirect(&indirect_buffer, 0);
```

---

## 6. LOD Strategy

Use a Quadtree-based chunk system. Only leaf nodes are rendered.

### Quadtree Structure

```
LOD 0: 1 chunk (512m)
       |
LOD 1: 4 chunks (256m each)
       |
LOD 2: 16 chunks (128m each)
       |
LOD 3: 64 chunks (64m each)
```

### Selection Criteria

```rust
fn should_subdivide(chunk: &Chunk, camera_pos: Vec3) -> bool {
    let distance = chunk.center().distance(camera_pos);
    let threshold = chunk.size * LOD_FACTOR;  // e.g., 2.0
    distance < threshold && chunk.lod < MAX_LOD
}
```

### Rendering

Only render leaf nodes (chunks that are not subdivided):

```rust
fn collect_visible_chunks(node: &QuadtreeNode, camera: &Camera, out: &mut Vec<ChunkId>) {
    if !camera.frustum_contains(node.bounds) {
        return;
    }
    
    if node.is_leaf() {
        out.push(node.chunk_id);
    } else {
        for child in &node.children {
            collect_visible_chunks(child, camera, out);
        }
    }
}
```

---

## 7. Buffer Layout

### Per-Chunk Buffers

| Buffer          | Size Calculation           | Example (32^3) |
|-----------------|----------------------------|----------------|
| Density         | (res+1)^3 * 4              | 143 KB         |
| Vertices        | res^3 * 24 (vec3+vec3)     | 786 KB         |
| Cell-to-Vertex  | res^3 * 4                  | 131 KB         |
| Indices         | res^3 * 6 * 4              | 786 KB         |
| Indirect        | 20 bytes                   | 20 B           |
| **Total**       |                            | ~1.8 MB        |

---

## 8. Pitfalls and Cautions

> [!CAUTION]
> **Critical Issues**

### 8.1. T-Junctions Between LOD Levels

**Problem:** Adjacent chunks with different LODs have mismatched vertices at boundaries.

**Symptom:** Visible cracks/holes between chunks.

**Solutions:**
1. **Skirts:** Extend geometry below surface at chunk edges
2. **Stitching:** Generate transition meshes between LOD levels
3. **Constrain vertices:** Force boundary vertices to match lower LOD grid

```
LOD 1  |  LOD 0
       |
  x----x----x
  |    |    |
  x----+----x  <- T-junction causes crack
       |
  x----x----x
```

### 8.2. Atomic Overflow

**Problem:** `atomicAdd` on vertex/index count exceeds buffer size.

**Symptom:** Buffer corruption, GPU crash.

**Solution:** Check bounds before writing:

```wgsl
let idx = atomicAdd(&vertex_count, 1u);
if (idx >= MAX_VERTICES) {
    atomicSub(&vertex_count, 1u);
    return;
}
vertices[idx] = vertex;
```

### 8.3. Indirect Buffer Usage Flags

**Problem:** Missing required usage flags on indirect buffer.

**Symptom:** Validation error.

**Solution:**
```rust
usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
```

### 8.4. Workgroup Size Mismatch

**Problem:** Dispatch count doesn't cover all cells.

**Symptom:** Missing terrain at chunk edges.

**Solution:** Round up dispatch count:
```rust
let dispatch = (resolution + workgroup_size - 1) / workgroup_size;
```

### 8.5. Density Gradient Precision

**Problem:** Using too small epsilon for gradient calculation.

**Symptom:** Noisy or incorrect normals.

**Solution:** Use epsilon proportional to cell size:
```wgsl
let eps = cell_size * 0.1;
let gradient = vec3<f32>(
    terrain_sdf(p + vec3<f32>(eps, 0.0, 0.0)) - terrain_sdf(p - vec3<f32>(eps, 0.0, 0.0)),
    terrain_sdf(p + vec3<f32>(0.0, eps, 0.0)) - terrain_sdf(p - vec3<f32>(0.0, eps, 0.0)),
    terrain_sdf(p + vec3<f32>(0.0, 0.0, eps)) - terrain_sdf(p - vec3<f32>(0.0, 0.0, eps))
) / (2.0 * eps);
```

---

## References

- [Dual Contouring Paper](https://www.cs.wustl.edu/~taoju/research/dualContour.pdf)
- [GPU Gems 3 - GPU Terrain](https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-1-generating-complex-procedural-terrains-using-gpu)
- [Transvoxel Algorithm](https://transvoxel.org/)
