# Data Layout and Memory Alignment

This document defines the memory layout and alignment rules for data shared between Rust (CPU) and WGSL (GPU). Correct alignment is critical to prevent rendering artifacts, GPU validation errors, and silent data corruption.

---

## Table of Contents

1. [WGSL Alignment Rules Overview](#1-wgsl-alignment-rules-overview)
2. [std140 Layout Rules](#2-std140-layout-rules)
3. [std430 Layout Rules](#3-std430-layout-rules)
4. [Rust to WGSL Struct Mapping](#4-rust-to-wgsl-struct-mapping)
5. [Alignment Quick Reference Table](#5-alignment-quick-reference-table)
6. [Practical Examples](#6-practical-examples)
7. [Pitfalls and Cautions](#7-pitfalls-and-cautions)

---

## 1. WGSL Alignment Rules Overview

WGSL shader language enforces strict memory alignment rules for uniform and storage buffers. These rules are derived from the underlying graphics API requirements (Vulkan, Metal, DirectX 12).

### Buffer Types and Their Layout Rules

| Buffer Type      | WGSL Declaration                    | Layout Rule |
|------------------|-------------------------------------|-------------|
| Uniform Buffer   | `var<uniform>`                      | std140      |
| Storage Buffer   | `var<storage, read>` or `read_write`| std430      |

### Key Terminology

- **Alignment**: The byte boundary where a value must begin
- **Size**: The number of bytes the value occupies
- **Stride**: The distance between consecutive array elements

---

## 2. std140 Layout Rules

Used for **Uniform Buffers** (`var<uniform>`). This layout prioritizes portability over memory efficiency.

### Fundamental Rules

1. **Scalars** (f32, i32, u32): 4-byte alignment, 4-byte size
2. **vec2<T>**: 8-byte alignment, 8-byte size
3. **vec3<T>**: **16-byte alignment**, 12-byte size (padded to 16)
4. **vec4<T>**: 16-byte alignment, 16-byte size
5. **mat4x4<f32>**: Treated as array of 4 vec4, 16-byte alignment, 64-byte size
6. **mat3x3<f32>**: Treated as array of 3 vec4 (padded!), 16-byte alignment, 48-byte size
7. **Structs**: Alignment = largest member alignment, rounded up to 16-byte boundary
8. **Arrays**: Element alignment rounded up to 16-byte boundary

### Critical std140 Quirks

> [!WARNING]
> **vec3 is 16-byte aligned in std140!**
> 
> This is the most common source of alignment bugs. A `vec3<f32>` requires 16-byte alignment and occupies 16 bytes (including 4 bytes of padding).

### std140 Struct Alignment Example

```wgsl
// WGSL
struct LightData {
    position: vec3<f32>,    // offset 0, alignment 16, size 16 (12 + 4 pad)
    intensity: f32,         // offset 16, alignment 4, size 4
    direction: vec3<f32>,   // offset 32, alignment 16, size 16 (12 + 4 pad)
    range: f32,             // offset 48, alignment 4, size 4
}
// Total size: 52 bytes, rounded to 64 (struct alignment = 16)
```

---

## 3. std430 Layout Rules

Used for **Storage Buffers** (`var<storage>`). More memory-efficient than std140.

### Key Differences from std140

1. **vec3<T>**: Still 16-byte aligned (same as std140)
2. **Structs**: Alignment = largest member alignment (NOT rounded to 16)
3. **Arrays**: Element alignment = actual element alignment (NOT rounded to 16)

### std430 Array Advantage

```wgsl
// std140: array<f32, 4> requires 16-byte stride (wasteful)
// std430: array<f32, 4> uses 4-byte stride (efficient)
```

---

## 4. Rust to WGSL Struct Mapping

### Required Rust Attributes

Always use these attributes when defining GPU-shared structs:

```rust
#[repr(C)]           // Use C memory layout (predictable ordering)
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
```

### Using bytemuck for Safe Transmutation

The `bytemuck` crate provides safe casting between Rust types and byte slices:

```toml
# Cargo.toml
[dependencies]
bytemuck = { version = "1", features = ["derive"] }
```

### Explicit Padding Strategy

**Always add explicit padding fields** rather than relying on implicit alignment:

```rust
// Rust struct matching WGSL uniform buffer
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightData {
    position: [f32; 3],    // 12 bytes
    _pad0: f32,            // 4 bytes explicit padding
    direction: [f32; 3],   // 12 bytes
    intensity: f32,        // 4 bytes (fits in vec3 padding slot)
    color: [f32; 3],       // 12 bytes
    range: f32,            // 4 bytes
}
```

---

## 5. Alignment Quick Reference Table

### Scalar and Vector Types

| WGSL Type       | Alignment (bytes) | Size (bytes) | Rust Equivalent           |
|-----------------|-------------------|--------------|---------------------------|
| f32             | 4                 | 4            | f32                       |
| i32             | 4                 | 4            | i32                       |
| u32             | 4                 | 4            | u32                       |
| vec2<f32>       | 8                 | 8            | [f32; 2]                  |
| vec3<f32>       | 16                | 12 (+4 pad)  | [f32; 3] + _pad: f32      |
| vec4<f32>       | 16                | 16           | [f32; 4]                  |

### Matrix Types

| WGSL Type       | Alignment (bytes) | Size (bytes) | Rust Equivalent           |
|-----------------|-------------------|--------------|---------------------------|
| mat2x2<f32>     | 8                 | 16           | [[f32; 2]; 2]             |
| mat3x3<f32>     | 16                | 48           | [[f32; 4]; 3] (padded!)   |
| mat4x4<f32>     | 16                | 64           | [[f32; 4]; 4]             |

> [!IMPORTANT]
> **mat3x3 requires padding!**
> 
> In std140/std430, mat3x3 is stored as 3 vec4 columns, not 3 vec3. Each column has 4 bytes of padding.

---

## 6. Practical Examples

### Example 1: Camera Uniform Buffer

```wgsl
// WGSL
struct CameraUniforms {
    view: mat4x4<f32>,        // offset 0, size 64
    projection: mat4x4<f32>,  // offset 64, size 64
    view_projection: mat4x4<f32>, // offset 128, size 64
    position: vec3<f32>,      // offset 192, size 12
    _padding: f32,            // offset 204, size 4
}
// Total: 208 bytes
```

```rust
// Rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniforms {
    view: [[f32; 4]; 4],           // 64 bytes
    projection: [[f32; 4]; 4],     // 64 bytes
    view_projection: [[f32; 4]; 4], // 64 bytes
    position: [f32; 3],            // 12 bytes
    _padding: f32,                 // 4 bytes
}

// Verify size at compile time
const _: () = assert!(std::mem::size_of::<CameraUniforms>() == 208);
```

### Example 2: Material Storage Buffer

```wgsl
// WGSL
struct Material {
    albedo: vec4<f32>,        // offset 0, size 16
    metallic: f32,            // offset 16, size 4
    roughness: f32,           // offset 20, size 4
    ao: f32,                  // offset 24, size 4
    _padding: f32,            // offset 28, size 4
}
// Total: 32 bytes
```

```rust
// Rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Material {
    albedo: [f32; 4],    // 16 bytes
    metallic: f32,       // 4 bytes
    roughness: f32,      // 4 bytes
    ao: f32,             // 4 bytes
    _padding: f32,       // 4 bytes
}

const _: () = assert!(std::mem::size_of::<Material>() == 32);
```

### Example 3: Using glam Types Directly

When using `glam` with bytemuck, ensure proper feature flags:

```toml
# Cargo.toml
[dependencies]
glam = { version = "0.24", features = ["bytemuck"] }
```

```rust
use glam::{Vec3, Vec4, Mat4};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Transform {
    model: Mat4,        // 64 bytes, 16-byte aligned
    normal: Mat4,       // 64 bytes (use mat4 for normal matrix, not mat3!)
}
```

---

## 7. Pitfalls and Cautions

> [!CAUTION]
> **Critical Alignment Issues to Avoid**

### 7.1. vec3 Alignment Trap

**Problem:** Placing a scalar immediately after vec3 without accounting for padding.

**Symptom:** Data appears shifted or corrupted in shader.

**Wrong:**
```rust
struct Wrong {
    position: [f32; 3],  // 12 bytes
    scale: f32,          // Rust places at offset 12, but WGSL expects offset 16!
}
```

**Correct:**
```rust
struct Correct {
    position: [f32; 3],  // 12 bytes
    _pad: f32,           // 4 bytes padding
    scale: f32,          // Now correctly at offset 16
    // Or better: combine into position_and_scale: [f32; 4]
}
```

---

### 7.2. mat3x3 Column Padding

**Problem:** Using `[[f32; 3]; 3]` for mat3x3.

**Symptom:** Matrix data is completely wrong in shader.

**Wrong:**
```rust
struct Wrong {
    normal_matrix: [[f32; 3]; 3],  // 36 bytes - INCORRECT
}
```

**Correct:**
```rust
struct Correct {
    // Each column is padded to vec4
    normal_matrix: [[f32; 4]; 3],  // 48 bytes - CORRECT
}
```

---

### 7.3. Array Stride in Uniform Buffers

**Problem:** Assuming array elements are tightly packed in uniform buffers.

**Symptom:** Only every 4th array element appears correct.

**Wrong (for uniform buffer):**
```rust
struct Wrong {
    weights: [f32; 8],  // Rust: 32 bytes
    // WGSL std140: 8 * 16 = 128 bytes (each f32 padded to 16!)
}
```

**Correct:**
```rust
struct Correct {
    // Pack 4 floats into vec4 to avoid std140 array padding
    weights: [[f32; 4]; 2],  // 32 bytes, matching WGSL
}
```

---

### 7.4. Buffer Mapping Lifetime

**Problem:** Holding buffer mapping across await points.

**Symptom:** `BufferAsyncError` or GPU validation error.

**Wrong:**
```rust
// WRONG: Mapping may be invalidated
let slice = buffer.slice(..);
slice.map_async(wgpu::MapMode::Write, |_| {});
device.poll(wgpu::Maintain::Wait);
// ... do other async work ...
let data = slice.get_mapped_range_mut(); // May fail!
```

**Correct:**
```rust
// CORRECT: Use mapping immediately
let slice = buffer.slice(..);
slice.map_async(wgpu::MapMode::Write, |_| {});
device.poll(wgpu::Maintain::Wait);
{
    let mut data = slice.get_mapped_range_mut();
    data.copy_from_slice(bytemuck::cast_slice(&cpu_data));
}
buffer.unmap(); // Unmap before any other GPU work
```

---

### 7.5. Forgetting repr(C)

**Problem:** Omitting `#[repr(C)]` from GPU-shared structs.

**Symptom:** Unpredictable layout, data corruption.

**Wrong:**
```rust
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
struct Wrong {  // Missing repr(C)!
    a: f32,
    b: u32,
}
```

**Correct:**
```rust
#[repr(C)]  // REQUIRED for deterministic layout
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Correct {
    a: f32,
    b: u32,
}
```

---

### 7.6. Compile-Time Size Verification

**Best Practice:** Always verify struct sizes at compile time:

```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MyUniform {
    // ... fields ...
}

// This assertion runs at compile time - zero runtime cost
const _: () = assert!(
    std::mem::size_of::<MyUniform>() == 128,
    "MyUniform size mismatch! Check alignment."
);
```

---

## References

- [WGSL Specification - Memory Layout](https://www.w3.org/TR/WGSL/#memory-layouts)
- [WebGPU Buffer Mapping](https://www.w3.org/TR/webgpu/#buffer-mapping)
- [bytemuck Documentation](https://docs.rs/bytemuck/latest/bytemuck/)
- [glam bytemuck Support](https://docs.rs/glam/latest/glam/#bytemuck)
