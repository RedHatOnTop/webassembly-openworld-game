# Deferred Renderer

Technical specification for Aether Engine's Deferred Rendering Pipeline optimized for WebGPU.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [G-Buffer Specification](#2-g-buffer-specification)
3. [Render Pass 1: G-Buffer Pass](#3-render-pass-1-g-buffer-pass)
4. [Render Pass 2: Lighting Pass (Compute)](#4-render-pass-2-lighting-pass-compute)
5. [Render Pass 3: Skybox and Translucency](#5-render-pass-3-skybox-and-translucency)
6. [Render Pass 4: Post-Processing](#6-render-pass-4-post-processing)
7. [Bind Group Layouts](#7-bind-group-layouts)
8. [Depth Reconstruction](#8-depth-reconstruction)
9. [Pitfalls and Cautions](#9-pitfalls-and-cautions)

---

## 1. Pipeline Overview

| Pass | Type    | Input                  | Output           |
|------|---------|------------------------|------------------|
| 1    | Render  | Geometry, Materials    | G-Buffer         |
| 2    | Compute | G-Buffer, Lights       | HDR Buffer       |
| 3    | Render  | Skybox, Transparent    | HDR Buffer       |
| 4    | Render  | HDR Buffer             | Swapchain        |

---

## 2. G-Buffer Specification

### Attachment Layout

| Attachment | Name             | Format          | Content                              |
|------------|------------------|-----------------|--------------------------------------|
| 0          | Albedo+Metalness | `Rgba8Unorm`    | RGB: Albedo, A: Metalness            |
| 1          | Normal+Roughness | `Rgba16Float`   | RGB: World Normal, A: Roughness      |
| Depth      | Depth            | `Depth32Float`  | Depth (for position reconstruction)  |

> [!IMPORTANT]
> World-space normals require `Rgba16Float` to prevent banding. Using `Rgba8Unorm` causes visible lighting discontinuities.

### Rust Texture Creation

```rust
fn create_gbuffer(device: &wgpu::Device, width: u32, height: u32) -> GBuffer {
    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    
    let albedo_metalness = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("GBuffer: Albedo+Metalness"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    
    let normal_roughness = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("GBuffer: Normal+Roughness"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("GBuffer: Depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    
    GBuffer { albedo_metalness, normal_roughness, depth, width, height }
}
```

---

## 3. Render Pass 1: G-Buffer Pass

### Fragment Shader Output (WGSL)

```wgsl
struct GBufferOutput {
    @location(0) albedo_metalness: vec4<f32>,
    @location(1) normal_roughness: vec4<f32>,
}

@fragment
fn fs_gbuffer(input: VertexOutput) -> GBufferOutput {
    let albedo = textureSample(albedo_texture, material_sampler, input.uv).rgb;
    let arm = textureSample(arm_texture, material_sampler, input.uv).rgb;
    let normal_map = textureSample(normal_texture, material_sampler, input.uv).rgb * 2.0 - 1.0;
    
    // TBN transform
    let bitangent = cross(input.world_normal, input.world_tangent.xyz) * input.world_tangent.w;
    let tbn = mat3x3<f32>(input.world_tangent.xyz, bitangent, input.world_normal);
    let world_normal = normalize(tbn * normal_map);
    
    var output: GBufferOutput;
    output.albedo_metalness = vec4<f32>(albedo, arm.b);
    output.normal_roughness = vec4<f32>(world_normal, arm.g);
    return output;
}
```

---

## 4. Render Pass 2: Lighting Pass (Compute)

### Light Structures (WGSL)

```wgsl
struct DirectionalLight {
    direction: vec3<f32>, _pad0: f32,
    color: vec3<f32>, intensity: f32,
}

struct PointLight {
    position: vec3<f32>, radius: f32,
    color: vec3<f32>, intensity: f32,
}

struct LightingUniforms {
    sun: DirectionalLight,
    ambient_color: vec3<f32>, ambient_intensity: f32,
    point_light_count: u32, _pad: vec3<u32>,
}
```

### Compute Shader (WGSL)

```wgsl
@group(0) @binding(0) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_depth: texture_depth_2d;
@group(0) @binding(3) var output_hdr: texture_storage_2d<rgba16float, write>;

@group(1) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(1) var<uniform> lighting: LightingUniforms;
@group(1) @binding(2) var<storage, read> point_lights: array<PointLight>;

@compute @workgroup_size(8, 8, 1)
fn cs_lighting(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(gbuffer_albedo);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let pixel = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    
    let albedo_m = textureLoad(gbuffer_albedo, pixel, 0);
    let normal_r = textureLoad(gbuffer_normal, pixel, 0);
    let depth = textureLoad(gbuffer_depth, pixel, 0);
    
    if (depth >= 1.0) { textureStore(output_hdr, pixel, vec4<f32>(0.0)); return; }
    
    let albedo = albedo_m.rgb;
    let metalness = albedo_m.a;
    let normal = normalize(normal_r.rgb);
    let roughness = max(normal_r.a, 0.04);
    
    let world_pos = reconstruct_position(uv, depth);
    let view_dir = normalize(camera.position - world_pos);
    
    // PBR lighting calculation...
    var total = calculate_pbr(albedo, normal, roughness, metalness, view_dir, world_pos);
    total += lighting.ambient_color * lighting.ambient_intensity * albedo;
    
    textureStore(output_hdr, pixel, vec4<f32>(total, 1.0));
}
```

---

## 5. Render Pass 3: Skybox and Translucency

Render skybox and transparent objects with forward rendering onto HDR buffer.

---

## 6. Render Pass 4: Post-Processing

### ACES Tonemapping (WGSL)

```wgsl
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

@fragment
fn fs_postprocess(input: FullscreenInput) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_buffer, linear_sampler, input.uv).rgb;
    let tonemapped = aces_filmic(hdr * exposure);
    return vec4<f32>(pow(tonemapped, vec3<f32>(1.0 / 2.2)), 1.0);
}
```

---

## 7. Bind Group Layouts

```rust
let gbuffer_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("G-Buffer Bind Group"),
    entries: &[
        wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
    ],
});
```

---

## 8. Depth Reconstruction

Reconstruct world position from depth to save G-Buffer bandwidth.

```wgsl
fn reconstruct_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), depth, 1.0);
    var view_pos = camera.inverse_projection * ndc;
    view_pos /= view_pos.w;
    return (camera.inverse_view * view_pos).xyz;
}
```

---

## 9. Pitfalls and Cautions

> [!CAUTION]
> **Critical Issues**

### 9.1. Missing Texture Usage Flags

G-Buffer textures MUST have both flags:
```rust
usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
```

### 9.2. Depth Reconstruction Y-Flip

WebGPU NDC has Y pointing up, but UV has Y pointing down. Flip Y in reconstruction:
```wgsl
let ndc_y = -(uv.y * 2.0 - 1.0);  // Note the negative
```

### 9.3. Compute Shader Bounds Check

Always guard against out-of-bounds pixel access:
```wgsl
if (gid.x >= dims.x || gid.y >= dims.y) { return; }
```

### 9.4. Normal Precision

Use `Rgba16Float` for normals. `Rgba8Unorm` causes visible banding on curved surfaces.

---

## References

- [WebGPU Texture Formats](https://www.w3.org/TR/webgpu/#texture-formats)
- [Learn OpenGL - Deferred Shading](https://learnopengl.com/Advanced-Lighting/Deferred-Shading)
