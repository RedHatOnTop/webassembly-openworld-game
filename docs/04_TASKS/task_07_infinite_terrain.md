# Task 07: Infinite Terrain System

## Status: NOT STARTED

## Goal

Implement an "Infinite Scrolling" terrain system where the mesh follows the camera, and the terrain shape updates procedurally based on world coordinates. Add debug features including Wireframe/Solid toggle and double-sided rendering.

## Priority: HIGH

## Estimated Effort: 4-6 hours

## Dependencies

* Task 06: Terrain Mesh Generation (COMPLETED)
* `docs/01_STANDARDS/coordinate_systems.md` (Y-Up, Right-handed)
* `docs/03_SUBSYSTEMS/terrain_compute.md` (Compute pipeline)

## Architecture Overview

```
Camera Movement
      |
      v
+------------------+     +------------------+     +------------------+
| Calculate Chunk  | --> |  Update Uniforms | --> |   Compute Pass   |
|    Offset        |     |  (chunk_offset)  |     | (world coords)   |
+------------------+     +------------------+     +------------------+
                                                          |
                                                          v
                                                 +------------------+
                                                 |   Render Pass    |
                                                 | (Fill/Wireframe) |
                                                 +------------------+
```

## Key Concepts

### Chunk Offset Strategy

The terrain mesh is a fixed-size grid (100x100) that "follows" the camera. Instead of creating new geometry, we shift the world-space offset used in the compute shader:

```
World Position = Grid Position + Chunk Offset
```

This means:
- The mesh vertices stay in the same relative positions
- The height function receives world coordinates
- Terrain features remain fixed in world space as camera moves

### Wireframe Mode

WebGPU supports `PolygonMode::Line` for wireframe rendering, but it requires the `POLYGON_MODE_LINE` feature. This must be explicitly requested during device creation.

## Implementation Steps

### Step 1: Context Update (context.rs)

Update `RenderContext::new` to request the `POLYGON_MODE_LINE` feature.

#### 1.1 Feature Request

```rust
// Request polygon mode line feature for wireframe rendering
let required_features = wgpu::Features::POLYGON_MODE_LINE;

let (device, queue) = adapter
    .request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Aether Engine Device"),
            required_features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    )
    .await
    .unwrap();
```

#### 1.2 Fallback Handling

```rust
// Check if feature is available, fallback gracefully
let adapter_features = adapter.features();
let polygon_mode_supported = adapter_features.contains(wgpu::Features::POLYGON_MODE_LINE);

let required_features = if polygon_mode_supported {
    wgpu::Features::POLYGON_MODE_LINE
} else {
    log::warn!("POLYGON_MODE_LINE not supported, wireframe mode disabled");
    wgpu::Features::empty()
};
```

#### 1.3 Store Feature Flag

```rust
pub struct RenderContext {
    // ...existing fields...
    pub polygon_mode_supported: bool,
}
```

### Step 2: Shader Update (terrain.wgsl)

#### 2.1 Update TerrainUniforms

```wgsl
struct TerrainUniforms {
    time: f32,
    grid_size: u32,
    chunk_offset_x: f32,  // World offset X
    chunk_offset_z: f32,  // World offset Z
}
```

#### 2.2 Update cs_update_terrain

```wgsl
@compute @workgroup_size(8, 8, 1)
fn cs_update_terrain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_size = uniforms.grid_size;
    let x = gid.x;
    let z = gid.y;
    
    if (x >= grid_size || z >= grid_size) {
        return;
    }
    
    let index = z * grid_size + x;
    let time = uniforms.time;
    
    // Grid-local position (centered)
    let half_size = f32(grid_size) * 0.5;
    let spacing = 0.5;
    
    let local_x = (f32(x) - half_size) * spacing;
    let local_z = (f32(z) - half_size) * spacing;
    
    // World position = local + chunk offset
    let world_x = local_x + uniforms.chunk_offset_x;
    let world_z = local_z + uniforms.chunk_offset_z;
    
    // Height uses WORLD coordinates (terrain stays fixed in world)
    let world_y = get_height(world_x, world_z, time);
    
    // Position in mesh-local space (for rendering)
    // The mesh is centered around (chunk_offset_x, 0, chunk_offset_z)
    vertices[index].position = vec4<f32>(world_x, world_y, world_z, 1.0);
    
    // Normal calculation also uses world coordinates
    let delta = spacing;
    let height_px = get_height(world_x + delta, world_z, time);
    let height_pz = get_height(world_x, world_z + delta, time);
    
    let tangent_x = vec3<f32>(delta, height_px - world_y, 0.0);
    let tangent_z = vec3<f32>(0.0, height_pz - world_y, delta);
    let normal = normalize(cross(tangent_z, tangent_x));
    
    vertices[index].normal = vec4<f32>(normal, 0.0);
}
```

### Step 3: Infinite Loop Logic (terrain.rs)

#### 3.1 Update TerrainUniforms Struct

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TerrainUniforms {
    pub time: f32,
    pub grid_size: u32,
    pub chunk_offset_x: f32,
    pub chunk_offset_z: f32,
}

impl TerrainUniforms {
    pub fn new() -> Self {
        Self {
            time: 0.0,
            grid_size: GRID_SIZE,
            chunk_offset_x: 0.0,
            chunk_offset_z: 0.0,
        }
    }
}

// Compile-time size verification (16 bytes)
const _: () = assert!(std::mem::size_of::<TerrainUniforms>() == 16);
```

#### 3.2 Update Method

```rust
impl TerrainSystem {
    /// Updates terrain based on camera position.
    /// Recalculates chunk offset to keep terrain centered on camera.
    pub fn update_camera_position(&mut self, camera_x: f32, camera_z: f32) {
        // Snap to grid spacing to prevent jitter
        let spacing = 0.5;
        let snap_x = (camera_x / spacing).floor() * spacing;
        let snap_z = (camera_z / spacing).floor() * spacing;
        
        self.uniforms.chunk_offset_x = snap_x;
        self.uniforms.chunk_offset_z = snap_z;
    }
    
    /// Updates time for animation.
    pub fn update_time(&mut self, time: f32) {
        self.uniforms.time = time;
    }
}
```

### Step 4: Camera Controls (camera.rs)

#### 4.1 Add Movement Methods

```rust
impl Camera {
    /// Move camera forward/backward along view direction.
    pub fn move_forward(&mut self, distance: f32) {
        let forward = (self.target - self.eye).normalize();
        let movement = forward * distance;
        self.eye += movement;
        self.target += movement;
    }
    
    /// Move camera left/right (strafe).
    pub fn move_right(&mut self, distance: f32) {
        let forward = (self.target - self.eye).normalize();
        let right = forward.cross(self.up).normalize();
        let movement = right * distance;
        self.eye += movement;
        self.target += movement;
    }
    
    /// Get camera XZ position for terrain offset.
    pub fn get_xz_position(&self) -> (f32, f32) {
        (self.eye.x, self.eye.z)
    }
}
```

### Step 5: Input Handling and Debug Features (mod.rs)

#### 5.1 Add Debug State

```rust
pub struct Renderer {
    // ...existing fields...
    
    // Debug state
    wireframe_mode: bool,
    double_sided_mode: bool,
    
    // Alternative pipelines
    terrain_wireframe_pipeline: Option<wgpu::RenderPipeline>,
    terrain_double_sided_pipeline: Option<wgpu::RenderPipeline>,
}
```

#### 5.2 Create Alternative Pipelines

```rust
fn create_terrain_wireframe_pipeline(
    device: &wgpu::Device,
    // ...other params...
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        // ...same as solid pipeline except:
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Line, // Wireframe!
            // ...
        },
        // ...
    })
}

fn create_terrain_double_sided_pipeline(
    device: &wgpu::Device,
    // ...other params...
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        // ...same as solid pipeline except:
        primitive: wgpu::PrimitiveState {
            // ...
            cull_mode: None, // Double-sided!
            // ...
        },
        // ...
    })
}
```

#### 5.3 Input Processing

```rust
impl Renderer {
    /// Handle keyboard input for debug features.
    pub fn handle_key(&mut self, key: winit::keyboard::KeyCode, pressed: bool) {
        if !pressed { return; }
        
        match key {
            KeyCode::F1 => {
                self.wireframe_mode = !self.wireframe_mode;
                log::info!("Wireframe mode: {}", self.wireframe_mode);
            }
            KeyCode::F2 => {
                self.double_sided_mode = !self.double_sided_mode;
                log::info!("Double-sided mode: {}", self.double_sided_mode);
            }
            KeyCode::KeyW => self.camera.move_forward(1.0),
            KeyCode::KeyS => self.camera.move_forward(-1.0),
            KeyCode::KeyA => self.camera.move_right(-1.0),
            KeyCode::KeyD => self.camera.move_right(1.0),
            _ => {}
        }
    }
}
```

#### 5.4 Pipeline Selection in Render

```rust
// In render() method:
let terrain_pipeline = match (self.wireframe_mode, self.double_sided_mode) {
    (true, _) => self.terrain_wireframe_pipeline.as_ref()
        .unwrap_or(&self.terrain_system.render_pipeline),
    (false, true) => self.terrain_double_sided_pipeline.as_ref()
        .unwrap_or(&self.terrain_system.render_pipeline),
    (false, false) => &self.terrain_system.render_pipeline,
};

render_pass.set_pipeline(terrain_pipeline);
```

### Step 6: Main Loop Integration (main.rs)

#### 6.1 Handle Window Events

```rust
Event::WindowEvent { event, .. } => match event {
    WindowEvent::KeyboardInput { event, .. } => {
        if let PhysicalKey::Code(key_code) = event.physical_key {
            renderer.handle_key(key_code, event.state.is_pressed());
        }
    }
    // ...existing handlers...
}
```

#### 6.2 Update Terrain Each Frame

```rust
// In game loop, before render:
let (cam_x, cam_z) = renderer.get_camera_xz();
renderer.update_terrain_offset(cam_x, cam_z);
```

## Data Flow

```
1. User Input (WASD) -> Camera Movement
2. Camera Position -> TerrainSystem::update_camera_position()
3. Uniforms Updated -> chunk_offset_x, chunk_offset_z
4. Compute Shader -> world_pos = local_pos + chunk_offset
5. Height Function -> get_height(world_x, world_z, time)
6. Terrain Features Fixed in World Space
```

## Verification Checklist

1. **Build Verification:**
   - [ ] `cargo build` succeeds without errors
   - [ ] `cargo build --target wasm32-unknown-unknown` succeeds

2. **Infinite Terrain:**
   - [ ] Press W: Camera moves forward, terrain continues
   - [ ] Press S: Camera moves backward, terrain continues
   - [ ] Press A/D: Camera strafes, terrain continues
   - [ ] Mountains/valleys stay fixed in world space

3. **Debug Features:**
   - [ ] Press F1: Toggle wireframe mode
   - [ ] Wireframe shows triangle structure clearly
   - [ ] Press F2: Toggle double-sided rendering
   - [ ] Terrain visible from below when double-sided

4. **Performance:**
   - [ ] Smooth 60 FPS maintained
   - [ ] No index buffer regeneration during movement
   - [ ] Only compute shader runs for position updates

## Memory Layout

### TerrainUniforms (16 bytes)

| Offset | Field | Type | Size |
|--------|-------|------|------|
| 0 | time | f32 | 4 |
| 4 | grid_size | u32 | 4 |
| 8 | chunk_offset_x | f32 | 4 |
| 12 | chunk_offset_z | f32 | 4 |

## Keyboard Controls Summary

| Key | Action |
|-----|--------|
| W | Move camera forward |
| S | Move camera backward |
| A | Strafe camera left |
| D | Strafe camera right |
| F1 | Toggle wireframe mode |
| F2 | Toggle double-sided mode |

## Future Enhancements

* Mouse look for camera rotation
* Multiple terrain LOD levels
* Chunk streaming for truly infinite worlds
* Biome-based terrain generation
* Collision detection with terrain

## Related Documents

* [Task 06: Terrain Mesh Generation](task_06_terrain_mesh.md)
* [Terrain Compute System](../03_SUBSYSTEMS/terrain_compute.md)
* [Coordinate Systems](../01_STANDARDS/coordinate_systems.md)
