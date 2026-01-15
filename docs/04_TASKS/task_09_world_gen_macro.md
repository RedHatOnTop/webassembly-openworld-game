# Task 09: World Generation - Macro Structure and Noise Infrastructure

| Status | Priority | Depends On |
|--------|----------|------------|
| Pending | High | Task 07 (Infinite Terrain) |

## Goal

Implement the GPU Noise Library and the 5-Channel Climate Model to generate the base continental terrain shape. This task focuses exclusively on macro-level terrain structure; caves, detailed erosion, and biomes are out of scope.

Reference: [world_gen_strategy.md](../02_ARCHITECTURE/world_gen_strategy.md)

---

## Implementation Steps

### Step 1: GPU Noise Library

Create `assets/shaders/noise.wgsl` as a standalone WGSL library.

**Required Functions:**

#### 1.1 Simplex Noise 3D

```wgsl
fn simplex_noise_3d(p: vec3<f32>) -> f32
```

Implementation should include:
- Skew/unskew factors: $F_3 = \frac{1}{3}$, $G_3 = \frac{1}{6}$
- Gradient table (12 edges of a cube)
- Permutation table (256 entries, can use hash function alternative)
- Return value in range $[-1.0, 1.0]$

#### 1.2 Simplex Noise 4D

```wgsl
fn simplex_noise_4d(p: vec4<f32>) -> f32
```

For time-variant noise sampling. Skew factors:
- $F_4 = \frac{\sqrt{5} - 1}{4} \approx 0.309$
- $G_4 = \frac{5 - \sqrt{5}}{20} \approx 0.138$

#### 1.3 Fractal Brownian Motion

```wgsl
fn fbm_3d(p: vec3<f32>, octaves: i32) -> f32
```

Standard FBM with configurable octaves:
$$\text{fbm}(p) = \sum_{i=0}^{n-1} \frac{N(p \cdot 2^i)}{2^i}$$

#### 1.4 Ridge Noise

```wgsl
fn ridge_noise_3d(p: vec3<f32>) -> f32
```

For mountain spines:
$$R(p) = 1.0 - |N(p)|$$

#### 1.5 Cubic Spline (Critical for Continentalness)

```wgsl
fn cubic_spline(t: f32, points: array<vec2<f32>, 6>) -> f32
```

Implements Catmull-Rom or similar cubic interpolation through control points.

**Continentalness Control Points (from architecture doc):**
```wgsl
const CONTINENTALNESS_SPLINE: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0),   // Deep ocean stays deep
    vec2(-0.4, -0.5),   // Gradual shelf transition
    vec2(-0.1,  0.0),   // Sharp coastal definition
    vec2( 0.3,  0.4),   // Inland plateau
    vec2( 0.7,  0.6),   // Mountain foothills
    vec2( 1.0,  1.0)    // Peak preservation
);
```

---

### Step 2: Shader Integration

Modify `assets/shaders/terrain.wgsl` to use the noise library.

#### 2.1 Include Noise Library

If WGSL imports are not yet supported, paste the contents of `noise.wgsl` at the top of `terrain.wgsl`. Add a comment marker for future extraction:

```wgsl
// ============================================================
// NOISE LIBRARY (TODO: Extract to noise.wgsl when imports work)
// ============================================================
```

#### 2.2 Climate Channel Sampling

Implement the climate channel sampler:

```wgsl
struct ClimateChannels {
    continentalness: f32,  // -1.0 (deep ocean) to 1.0 (high peaks)
    erosion: f32,          // -1.0 (jagged) to 1.0 (flat/eroded)
    peaks_valleys: f32,    // Ridge noise for mountain spines
};

fn get_climate_channels(world_pos: vec3<f32>, seed: f32) -> ClimateChannels {
    var result: ClimateChannels;
    
    // Continentalness: Low frequency noise + spline remap
    let cont_raw = fbm_3d(world_pos * 0.0005 + vec3(seed, 0.0, 0.0), 4);
    result.continentalness = cubic_spline(cont_raw, CONTINENTALNESS_SPLINE);
    
    // Erosion: Controls terrain roughness
    result.erosion = fbm_3d(world_pos * 0.001 + vec3(0.0, seed, 0.0), 3);
    
    // Peaks & Valleys: Ridge noise for mountain definition
    result.peaks_valleys = ridge_noise_3d(world_pos * 0.002 + vec3(0.0, 0.0, seed));
    
    return result;
}
```

#### 2.3 Height Calculation

Replace the existing simple height function with continentalness-based terrain:

```wgsl
fn get_terrain_height(world_pos: vec2<f32>, seed: f32) -> f32 {
    let pos_3d = vec3(world_pos.x, 0.0, world_pos.y);
    let climate = get_climate_channels(pos_3d, seed);
    
    // Base height from continentalness
    // Deep ocean (-1.0) -> -50m, High peaks (1.0) -> +200m
    let base_height = climate.continentalness * 125.0 + 75.0;
    
    // Add peaks/valleys modulated by continentalness
    // Only apply mountain detail on land (continentalness > 0)
    let mountain_mask = smoothstep(-0.1, 0.3, climate.continentalness);
    let peak_contribution = climate.peaks_valleys * 80.0 * mountain_mask;
    
    // Erosion reduces height variation
    // High erosion -> flatter terrain
    let erosion_factor = 1.0 - (climate.erosion * 0.5 + 0.5) * 0.6;
    
    // Final height with detail noise
    let detail = simplex_noise_3d(pos_3d * 0.05) * 5.0;
    
    return base_height + peak_contribution * erosion_factor + detail;
}
```

#### 2.4 Update Compute Shader Entry Point

Modify `cs_update_terrain` to use the new height calculation:

```wgsl
@compute @workgroup_size(8, 8, 1)
fn cs_update_terrain(@builtin(global_invocation_id) id: vec3<u32>) {
    let vertex_index = id.x + id.y * uniforms.grid_size;
    if (vertex_index >= uniforms.vertex_count) { return; }
    
    // Calculate world position
    let grid_x = f32(id.x);
    let grid_z = f32(id.y);
    let world_x = uniforms.chunk_offset.x + grid_x * uniforms.scale;
    let world_z = uniforms.chunk_offset.y + grid_z * uniforms.scale;
    
    // Sample terrain height using climate model
    let height = get_terrain_height(vec2(world_x, world_z), uniforms.seed);
    
    // Update vertex
    vertices[vertex_index].position = vec3(world_x, height, world_z);
    
    // Store climate data for debug visualization (optional)
    // vertices[vertex_index].debug_data = pack_climate_channels(...);
}
```

---

### Step 3: Rust Updates

Modify `src/game/terrain.rs` (or equivalent terrain module).

#### 3.1 Update TerrainUniforms

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainUniforms {
    pub chunk_offset: [f32; 2],
    pub scale: f32,
    pub grid_size: u32,
    pub vertex_count: u32,
    pub seed: f32,           // NEW: World generation seed
    pub sea_level: f32,      // NEW: For ocean cutoff
    pub debug_mode: u32,     // NEW: Visualization mode
}
```

#### 3.2 Initialize Seed

Add a world seed to the terrain system initialization:

```rust
impl TerrainSystem {
    pub fn new(/* ... */, world_seed: u32) -> Self {
        let seed = world_seed as f32 * 0.001; // Normalize for shader
        // ...
    }
}
```

---

### Step 4: Debug Visualization

Add debug visualization modes to the fragment shader.

#### 4.1 Fragment Shader Debug Modes

Add to `terrain.wgsl` fragment shader:

```wgsl
// Debug mode constants
const DEBUG_MODE_NORMAL: u32 = 0u;
const DEBUG_MODE_CONTINENTALNESS: u32 = 1u;
const DEBUG_MODE_EROSION: u32 = 2u;
const DEBUG_MODE_PEAKS: u32 = 3u;

@fragment
fn fs_terrain(in: VertexOutput) -> @location(0) vec4<f32> {
    // Recalculate climate for debug (or pass via vertex data)
    let climate = get_climate_channels(in.world_position, uniforms.seed);
    
    switch (uniforms.debug_mode) {
        case DEBUG_MODE_CONTINENTALNESS: {
            // Black = Deep Ocean (-1), Blue = Ocean, Green = Coast, Brown = Land, White = Peaks
            let c = climate.continentalness * 0.5 + 0.5; // Normalize to 0-1
            if (c < 0.3) {
                // Ocean gradient (black to blue)
                return vec4(0.0, 0.0, c / 0.3, 1.0);
            } else if (c < 0.5) {
                // Coastal (blue to green)
                let t = (c - 0.3) / 0.2;
                return vec4(0.0, t, 1.0 - t, 1.0);
            } else {
                // Land to peaks (green to white)
                let t = (c - 0.5) / 0.5;
                return vec4(t, 0.5 + t * 0.5, t * 0.5, 1.0);
            }
        }
        case DEBUG_MODE_EROSION: {
            // Red = Jagged (low erosion), Green = Flat (high erosion)
            let e = climate.erosion * 0.5 + 0.5;
            return vec4(1.0 - e, e, 0.0, 1.0);
        }
        case DEBUG_MODE_PEAKS: {
            // Grayscale for ridge intensity
            let p = climate.peaks_valleys;
            return vec4(p, p, p, 1.0);
        }
        default: {
            // Normal rendering with basic lighting
            let light_dir = normalize(vec3(0.5, 1.0, 0.3));
            let diffuse = max(dot(in.normal, light_dir), 0.2);
            
            // Color by altitude
            let altitude_color = select(
                vec3(0.2, 0.5, 0.2),  // Low: grass green
                vec3(0.5, 0.4, 0.3),  // High: rock brown
                in.world_position.y > 100.0
            );
            
            return vec4(altitude_color * diffuse, 1.0);
        }
    }
}
```

#### 4.2 Input Handling in main.rs

Add keyboard input to toggle debug modes:

```rust
// In input handling section
WindowEvent::KeyboardInput { event, .. } => {
    if event.state == ElementState::Pressed {
        match event.physical_key {
            PhysicalKey::Code(KeyCode::F1) => {
                terrain_system.set_debug_mode(0); // Normal
                log::info!("Debug Mode: Normal Rendering");
            }
            PhysicalKey::Code(KeyCode::F2) => {
                terrain_system.set_debug_mode(1); // Continentalness
                log::info!("Debug Mode: Continentalness");
            }
            PhysicalKey::Code(KeyCode::F3) => {
                terrain_system.set_debug_mode(2); // Erosion
                log::info!("Debug Mode: Erosion");
            }
            PhysicalKey::Code(KeyCode::F4) => {
                terrain_system.set_debug_mode(3); // Peaks & Valleys
                log::info!("Debug Mode: Peaks & Valleys");
            }
            _ => {}
        }
    }
}
```

#### 4.3 Uniform Update Method

```rust
impl TerrainSystem {
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.uniforms.debug_mode = mode;
        self.update_uniform_buffer();
    }
}
```

---

## File Structure After Completion

```
assets/
  shaders/
    noise.wgsl              # NEW - Standalone noise library
    terrain.wgsl            # MODIFIED - Climate-based terrain
src/
  game/
    terrain.rs              # MODIFIED - Updated uniforms
  main.rs                   # MODIFIED - Debug key bindings
```

---

## Verification

### Build Verification

```bash
# Native build
cargo build

# WASM build
cargo build --target wasm32-unknown-unknown
```

No shader compilation errors should occur.

### Visual Verification

1. **Run the Engine** and fly the camera to a high altitude (Y > 500).

2. **Normal Mode (F1):**
   - Terrain should show distinct continental shapes
   - NOT random noise blobs
   - Clear distinction between low areas (future oceans) and high areas (mountains)

3. **Continentalness Mode (F2):**
   - Black/Blue regions = Deep Ocean / Shelf
   - Green regions = Coastal zones
   - Brown/White regions = Inland / Mountain peaks
   - Transitions should be smooth but with clear continental edges

4. **Erosion Mode (F3):**
   - Red regions = Jagged/uneroded terrain
   - Green regions = Flat/eroded terrain
   - Should correlate loosely with terrain features

5. **Peaks Mode (F4):**
   - White lines = Mountain ridges and spines
   - Dark regions = Valleys
   - Ridge patterns should look natural, not grid-aligned

### Expected Visual Result

When viewed from above, the terrain should resemble a **continental map**:

```
+----------------------------------------------------------+
|    ~~~        [Dark Blue: Deep Ocean]         ~~~        |
|  ~~   ~~~~~~                           ~~~~~    ~~~      |
|    [Shelf]     +----------------+      [Shelf]           |
|                | Coastal Green  |                        |
|   ~~          |   +--------+   |          ~~             |
|               |   | Inland |   |                         |
|               |   | Brown  |   |                         |
|               |   +--------+   |                         |
|               |  [White Peaks] |                         |
|               +----------------+                         |
|    ~~~                                      ~~~          |
+----------------------------------------------------------+
```

---

## Acceptance Criteria

- [ ] `noise.wgsl` contains working `simplex_noise_3d`, `simplex_noise_4d`, `fbm_3d`, `ridge_noise_3d`, and `cubic_spline` functions.
- [ ] `terrain.wgsl` uses `get_climate_channels()` for terrain generation.
- [ ] Continentalness spline creates distinct ocean/land boundaries.
- [ ] Debug visualization modes (F1-F4) work correctly.
- [ ] Terrain viewed from altitude shows continental structure, not random noise.
- [ ] No performance regression (maintain 60fps at standard view distance).
- [ ] Both Native and WASM builds compile successfully.

---

## Notes

- **Octave Count:** Start with 4 octaves for continentalness, 3 for erosion. Adjust based on performance.
- **Seed Variation:** The seed offset ensures different channels produce independent noise patterns.
- **Future Work:** Task 10 will add caves (Spaghetti/Cheese noise), Task 11 will add biomes.
