# Task 10: Multi-Chunk Management System

**Status:** 📋 Planned  
**Priority:** High  
**Estimated Effort:** 6-8 hours  
**Dependencies:** Task 09 (World Generation - Macro Structure)

---

## Objective

Implement a chunk manager that renders a grid of chunks (e.g., 7x7 = 49 chunks) around the camera, enabling infinite world exploration with seamless terrain streaming and proper horizon visibility.

### Current Problem

![Single chunk island effect](file:///C:/Users/jin14/.gemini/antigravity/brain/86c7659d-3c2b-4a90-a047-b701549620f6/uploaded_image_2_1768467755777.png)

The terrain generation math works correctly (Task 09), but we are limited to viewing only a **single chunk** (100x100 vertices). This creates an "island floating in void" effect that breaks immersion and prevents exploration of the procedurally generated world.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         ChunkManager                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Chunk Pool (Fixed Size: 49)                │    │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐            │    │
│  │  │ C0  │ C1  │ C2  │ ... │ C46 │ C47 │ C48 │            │    │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┘            │    │
│  │       ↓ Pre-allocated GPU Buffers (No runtime alloc)    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │  update_chunks()  │                        │
│                    │  (Camera-driven)  │                        │
│                    └─────────┬─────────┘                        │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐              │
│         ▼                    ▼                    ▼             │
│    In-Range?            Out-of-Range?        Dirty Flag?        │
│    (Keep Active)      (Recycle Position)   (Compute Update)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Step 1: Chunk Architecture

**File:** `src/core/renderer/chunk.rs` (NEW)

Create a `Chunk` struct that encapsulates per-chunk GPU resources:

```rust
/// Chunk size in world units (matches GRID_SIZE from terrain.rs)
pub const CHUNK_SIZE: f32 = 100.0;

/// Render distance in chunks (7x7 grid = 3 in each direction)
pub const RENDER_DISTANCE: i32 = 3;

/// Total chunks in the pool (diameter = 2 * RENDER_DISTANCE + 1)
pub const CHUNK_POOL_SIZE: usize = ((RENDER_DISTANCE * 2 + 1) * (RENDER_DISTANCE * 2 + 1)) as usize;

/// A single terrain chunk with its own GPU resources.
pub struct Chunk {
    /// Chunk coordinate (not world position)
    pub coord: (i32, i32),
    
    /// Vertex storage buffer (pre-allocated, never reallocated)
    pub vertex_buffer: wgpu::Buffer,
    
    /// Whether this chunk needs GPU compute update
    pub dirty: bool,
    
    /// Whether this chunk is actively rendered
    pub active: bool,
}
```

> [!IMPORTANT]
> **Zero-Allocation Constraint:** All 49 chunk vertex buffers must be allocated at initialization time. When the camera moves, we **recycle** existing buffers by updating their coordinates - we never allocate new GPU memory at runtime.

---

### Step 2: Terrain System Refactor

**File:** `src/core/renderer/terrain.rs` (MODIFY)

Refactor `TerrainSystem` to manage a pool of chunks:

```rust
pub struct TerrainSystem {
    /// Pool of pre-allocated chunks
    chunks: Vec<Chunk>,
    
    /// Current center chunk coordinate (camera position)
    center_chunk: (i32, i32),
    
    /// Shared index buffer (all chunks use same topology)
    pub index_buffer: wgpu::Buffer,
    
    /// Compute pipelines (shared across chunks)
    pub index_gen_pipeline: wgpu::ComputePipeline,
    pub terrain_update_pipeline: wgpu::ComputePipeline,
    
    // ... existing fields
}
```

#### Key Methods

```rust
impl TerrainSystem {
    /// Initialize chunk pool with pre-allocated GPU buffers
    pub fn new(...) -> Self {
        let mut chunks = Vec::with_capacity(CHUNK_POOL_SIZE);
        for i in 0..CHUNK_POOL_SIZE {
            chunks.push(Chunk::new(device, (0, 0))); // Initial dummy coords
        }
        // Assign initial positions around origin
        Self::assign_initial_positions(&mut chunks);
        // ...
    }
    
    /// Update chunk visibility based on camera position
    pub fn update_chunks(&mut self, camera_x: f32, camera_z: f32) {
        // 1. Calculate camera's chunk coordinate
        let camera_chunk = (
            (camera_x / CHUNK_SIZE).floor() as i32,
            (camera_z / CHUNK_SIZE).floor() as i32,
        );
        
        // 2. Skip if camera hasn't moved to a new chunk
        if camera_chunk == self.center_chunk {
            return;
        }
        self.center_chunk = camera_chunk;
        
        // 3. Build set of required chunk coordinates
        let required: HashSet<(i32, i32)> = /* 7x7 grid around camera_chunk */;
        
        // 4. Find chunks that are out of range (candidates for recycling)
        let mut free_chunks: Vec<usize> = vec![];
        for (i, chunk) in self.chunks.iter().enumerate() {
            if !required.contains(&chunk.coord) {
                free_chunks.push(i);
            }
        }
        
        // 5. Find new positions that need chunks
        let existing: HashSet<(i32, i32)> = self.chunks.iter()
            .map(|c| c.coord).collect();
        let new_positions: Vec<(i32, i32)> = required.difference(&existing).collect();
        
        // 6. Recycle: Move free chunks to new positions and mark dirty
        for (new_pos, free_idx) in new_positions.iter().zip(free_chunks.iter()) {
            self.chunks[*free_idx].coord = *new_pos;
            self.chunks[*free_idx].dirty = true;
            self.chunks[*free_idx].active = true;
        }
    }
    
    /// Get list of dirty chunks that need compute update
    pub fn get_dirty_chunks(&self) -> Vec<usize> {
        self.chunks.iter()
            .enumerate()
            .filter(|(_, c)| c.dirty)
            .map(|(i, _)| i)
            .collect()
    }
}
```

---

### Step 3: Compute Pass Update

**File:** `src/core/renderer/mod.rs` (MODIFY)

Update the compute pass to process only dirty chunks:

```rust
// Compute Pass
{
    let mut compute_pass = encoder.begin_compute_pass(...);
    
    // Process only dirty chunks
    for chunk_idx in self.terrain_system.get_dirty_chunks() {
        let chunk = &self.terrain_system.chunks[chunk_idx];
        
        // Update chunk offset uniform
        self.terrain_system.update_chunk_uniform(queue, chunk.coord);
        
        // Bind chunk's vertex buffer
        compute_pass.set_bind_group(0, &chunk.compute_bind_group, &[]);
        compute_pass.set_bind_group(1, &self.terrain_system.compute_uniform_group, &[]);
        
        // Dispatch compute shader
        let workgroups = TerrainSystem::vertex_workgroup_count();
        compute_pass.dispatch_workgroups(workgroups, workgroups, 1);
        
        // Clear dirty flag (done in CPU after submit)
    }
}
```

---

### Step 4: Render Pass Update

**File:** `src/core/renderer/mod.rs` (MODIFY)

Update the render pass to draw all active chunks:

```rust
// Render Pass
{
    let mut render_pass = encoder.begin_render_pass(...);
    
    // Bind shared resources once
    render_pass.set_pipeline(&self.terrain_system.render_pipeline);
    render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
    render_pass.set_bind_group(2, &self.terrain_system.texture_bind_group, &[]);
    render_pass.set_index_buffer(...);
    
    // Draw each active chunk
    for chunk in self.terrain_system.chunks.iter().filter(|c| c.active) {
        // Update per-chunk offset uniform
        // Option A: Push constants (fast, if supported)
        // Option B: Update uniform buffer (slower, always works)
        
        render_pass.set_bind_group(0, &chunk.render_bind_group, &[]);
        render_pass.draw_indexed(0..INDEX_COUNT, 0, 0..1);
    }
}
```

---

### Step 5: Distance Fog (Atmosphere)

**File:** `assets/shaders/terrain.wgsl` (MODIFY)

Add distance-based fog to the fragment shader to hide chunk boundaries:

```wgsl
// Fog constants
const FOG_COLOR: vec3<f32> = vec3(0.7, 0.8, 0.9);  // Light blue-grey
const FOG_START: f32 = 200.0;   // Start fading
const FOG_END: f32 = 400.0;     // Fully fogged

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // ... existing terrain color calculation ...
    
    // Calculate distance from camera
    let camera_pos = vec3(render_uniforms.camera_x, render_uniforms.camera_y, render_uniforms.camera_z);
    let frag_pos = in.world_position.xyz;
    let distance = length(frag_pos - camera_pos);
    
    // Apply fog
    let fog_factor = smoothstep(FOG_START, FOG_END, distance);
    let final_color = mix(terrain_color, FOG_COLOR, fog_factor);
    
    return vec4(final_color, 1.0);
}
```

> [!TIP]
> The fog creates a natural "horizon" effect, making far chunks fade into the sky. This is crucial for immersion and hides the finite render distance.

---

### Step 6: Uniform Buffer Updates

**File:** `src/core/renderer/terrain.rs` (MODIFY)

Add camera position to the render uniform for fog calculations:

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RenderUniforms {
    pub time: f32,
    pub seed: f32,
    pub debug_mode: u32,
    pub _padding0: u32,
    
    // NEW: Camera position for fog
    pub camera_x: f32,
    pub camera_y: f32,
    pub camera_z: f32,
    pub _padding1: u32,
    
    // Chunk offset (updated per-draw)
    pub chunk_offset_x: f32,
    pub chunk_offset_z: f32,
    pub _padding2: [u32; 2],
}
```

---

## Memory Budget

| Resource | Count | Size Each | Total |
|----------|-------|-----------|-------|
| Vertex Buffer | 49 chunks | 320 KB | **15.7 MB** |
| Index Buffer | 1 (shared) | 235 KB | **235 KB** |
| Uniforms | 49 + 1 | ~128 bytes | **6.3 KB** |
| **Total GPU Memory** | | | **~16 MB** |

---

## Verification Checklist

- [ ] Engine starts and displays 7x7 grid of terrain chunks
- [ ] Moving camera (WASD) causes new chunks to stream in seamlessly
- [ ] No visible "pop-in" - new chunks appear with fog fade
- [ ] No GPU memory growth over time (check with GPU profiler)
- [ ] Frame rate stable at 60 FPS with all 49 chunks
- [ ] Terrain is continuous across chunk boundaries (no seams)
- [ ] Debug mode (F2) shows continentalness visualization across all chunks

---

## Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `src/core/renderer/chunk.rs` | **NEW** | Chunk struct and pool management |
| `src/core/renderer/terrain.rs` | MODIFY | Multi-chunk terrain system |
| `src/core/renderer/mod.rs` | MODIFY | Multi-chunk compute and render loops |
| `assets/shaders/terrain.wgsl` | MODIFY | Add distance fog |

---

## References

- [world_gen_strategy.md](file:///c:/Users/jin14/Projects/webassembly-openworld-game/docs/02_ARCHITECTURE/world_gen_strategy.md) - Infinite world philosophy
- [Task 09](file:///c:/Users/jin14/Projects/webassembly-openworld-game/docs/04_TASKS/task_09_world_gen_macro.md) - Macro terrain generation (prerequisite)

---

## Commit Message

```
feat: implement multi-chunk terrain streaming with distance fog (Task 10)
```
