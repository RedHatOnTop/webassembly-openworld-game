# Task 11: Biome & Material System

**Status:** 🔲 Not Started  
**Priority:** High  
**Depends On:** Task 10 (Multi-Chunk Terrain)

---

## Goal

Implement a data-driven Biome system using Temperature/Humidity noise channels, apply Triplanar Texturing with a Texture2DArray, and polish seabed geometry to eliminate underwater artifacts.

**Reference:** [world_gen_strategy.md](../02_ARCHITECTURE/world_gen_strategy.md)

---

## Implementation Steps

### Step 1: Climate Channels

**Files:** `assets/shaders/terrain.wgsl`

| Function | Description |
|----------|-------------|
| `get_temperature(pos, seed)` | Latitude-based + noise. Returns -1.0 (cold) to 1.0 (hot) |
| `get_humidity(pos, seed)` | Noise-based. Returns -1.0 (dry) to 1.0 (wet) |

**Seabed Fix:** In `get_terrain_height`:
```wgsl
// Smooth out peaks underwater (sediment simulation)
let ocean_fade = smoothstep(-0.3, -0.1, climate.continentalness);
let smooth_peaks = climate.peaks_valleys * ocean_fade;
```

---

### Step 2: Texture System

**File:** `src/core/renderer/texture.rs`

1. **Load Textures:** `grass.png`, `rock.png`, `sand.png`, `snow.png`
2. **Create Texture2DArray:** Single layered texture for shader efficiency
3. **Fallback:** Generate procedural noise textures if images missing

```rust
pub struct TextureArrayBuilder {
    layers: Vec<image::RgbaImage>,
}

impl TextureArrayBuilder {
    pub fn add_layer_from_file(&mut self, path: &str) -> Result<()>;
    pub fn add_procedural_layer(&mut self, generator: fn(u32, u32) -> [u8; 4]);
    pub fn build(self, device: &Device, queue: &Queue) -> TextureArray;
}
```

---

### Step 3: Biome Data Structure

**File:** `src/core/renderer/terrain.rs`

```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Biome {
    pub color: [f32; 3],       // Base tint
    pub texture_index: u32,    // Index into Texture2DArray
    pub roughness: f32,        // PBR roughness
    pub _padding: [f32; 3],    // 32-byte alignment
}
```

**Biome Table:**

| Biome | Texture | Trigger Condition |
|-------|---------|-------------------|
| Ocean | sand | `continentalness < -0.2` |
| Beach | sand | `height < 5.0` |
| Plains | grass | `temperature > 0.0, humidity > -0.3` |
| Desert | sand | `temperature > 0.5, humidity < -0.5` |
| Mountains | rock | `slope > 0.7` or `height > 80` |
| Snow Peaks | snow | `height > 120` or `temperature < -0.5` |

---

### Step 4: Triplanar Shader Logic

**File:** `assets/shaders/terrain.wgsl`

```wgsl
fn triplanar_sample(world_pos: vec3<f32>, normal: vec3<f32>, tex_index: i32) -> vec4<f32> {
    let blend = abs(normal);
    let blend_norm = blend / (blend.x + blend.y + blend.z);
    
    let scale = 0.1;  // UV scale
    let x_proj = textureSample(textures, tex_sampler, world_pos.yz * scale, tex_index);
    let y_proj = textureSample(textures, tex_sampler, world_pos.xz * scale, tex_index);
    let z_proj = textureSample(textures, tex_sampler, world_pos.xy * scale, tex_index);
    
    return x_proj * blend_norm.x + y_proj * blend_norm.y + z_proj * blend_norm.z;
}
```

**Biome Blending Logic:**
```wgsl
fn get_biome_color(pos: vec3<f32>, normal: vec3<f32>, height: f32) -> vec3<f32> {
    let temp = get_temperature(pos);
    let humidity = get_humidity(pos);
    let slope = 1.0 - normal.y;  // 0=flat, 1=vertical
    
    // Height-based overrides
    if (height > SNOW_LINE) { return triplanar_sample(pos, normal, LAYER_SNOW).rgb; }
    if (height < 0.0) { return triplanar_sample(pos, normal, LAYER_SAND).rgb; }  // Seabed
    
    // Slope override: cliffs always rock
    if (slope > 0.6) { return triplanar_sample(pos, normal, LAYER_ROCK).rgb; }
    
    // Climate-based biome selection
    let base = select_biome(temp, humidity);
    return triplanar_sample(pos, normal, base.texture_index).rgb;
}
```

---

## Verification Checklist

- [ ] **Snow Caps:** High peaks (>120m) show white snow texture
- [ ] **Cliffs:** Steep slopes (>45°) render rock regardless of biome
- [ ] **Beaches:** Low coastal areas (<5m) show sand
- [ ] **Seabed:** Underwater terrain is smooth, no spiky artifacts
- [ ] **No Stretching:** Triplanar texturing eliminates UV distortion on cliffs
- [ ] **Fallback Textures:** Engine runs without asset files (procedural fallback)

---

## Files Modified

| File | Changes |
|------|---------|
| `terrain.wgsl` | Temperature/humidity functions, triplanar sampling, biome blending |
| `texture.rs` | Texture2DArray creation, procedural fallbacks |
| `terrain.rs` | Biome struct, BiomeTable buffer |
| `mod.rs` | Bind group for texture array |

---

## Notes

- Triplanar sampling is 3x texture reads per fragment - acceptable for terrain
- Biome transitions use `smoothstep` for soft blending (no hard edges)
- Consider LOD-based texture mip selection for distant chunks
