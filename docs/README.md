# Aether Engine Documentation

> Single Source of Truth for the Aether Engine - A high-performance open-world 3D game engine built with Rust and WebGPU.

---

## Quick Reference

| Aspect              | Specification                                          |
|---------------------|--------------------------------------------------------|
| **Language**        | Rust (Target: `wasm32-unknown-unknown`)                |
| **Graphics API**    | WebGPU via `wgpu` crate                                |
| **Math Library**    | `glam` (SIMD-optimized linear algebra)                 |
| **Coordinate**      | Y-Up, Right-Handed                                     |
| **Rendering**       | Deferred Shading (PBR + IBL)                           |
| **Terrain**         | GPU-Driven Dual Contouring                             |
| **Architecture**    | Multi-threaded ECS (Main / Logic / Render)             |

---

## Documentation Structure

### 01_STANDARDS

Conventions and rules that must be followed throughout the codebase.

| Document                                                | Description                                    |
|---------------------------------------------------------|------------------------------------------------|
| [coordinate_systems.md](./01_STANDARDS/coordinate_systems.md) | World, NDC, UV coordinate conventions    |
| [data_layout.md](./01_STANDARDS/data_layout.md)         | WGSL alignment rules, Rust struct padding      |

---

### 02_ARCHITECTURE

High-level system design and architectural patterns.

| Document                                                | Description                                    |
|---------------------------------------------------------|------------------------------------------------|
| [threading_model.md](./02_ARCHITECTURE/threading_model.md)   | Multi-thread architecture, SharedArrayBuffer |
| [engine_loop.md](./02_ARCHITECTURE/engine_loop.md)      | Fixed/variable timestep, frame interpolation   |

---

### 03_SUBSYSTEMS

Detailed implementation specifications for each engine subsystem.

| Document                                                | Description                                    |
|---------------------------------------------------------|------------------------------------------------|
| [renderer_deferred.md](./03_SUBSYSTEMS/renderer_deferred.md) | G-Buffer format, render passes, PBR lighting |
| [terrain_compute.md](./03_SUBSYSTEMS/terrain_compute.md)     | Dual Contouring compute pipeline, LOD        |

---

### 04_TASKS

Development workflow and task management.

| Document                                                | Description                                    |
|---------------------------------------------------------|------------------------------------------------|
| [task_template.md](./04_TASKS/task_template.md)         | Standard template for development tasks        |

---

## Technology Stack

### Core Dependencies

```toml
[dependencies]
wgpu = "0.18"           # WebGPU implementation
glam = { version = "0.24", features = ["bytemuck"] }  # Math
bytemuck = { version = "1", features = ["derive"] }   # Pod casting
winit = "0.29"          # Window management
pollster = "0.3"        # Async runtime for WASM
```

### Build Targets

| Target                    | Use Case                      |
|---------------------------|-------------------------------|
| `wasm32-unknown-unknown`  | Browser deployment (primary)  |
| `x86_64-pc-windows-msvc`  | Windows native (development)  |
| `x86_64-unknown-linux-gnu`| Linux native (CI/development) |

---

## Architecture Overview

```
+------------------+     +------------------+     +------------------+
|   Main Thread    |     |   Logic Worker   |     |  Render Worker   |
|                  |     |                  |     |                  |
|  - Input         |<--->|  - ECS Systems   |<--->|  - G-Buffer Pass |
|  - UI/DOM        |     |  - Physics       |     |  - Lighting Pass |
|  - Audio         |     |  - AI/Scripts    |     |  - Post-Process  |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         +------------------------+------------------------+
                                  |
                    +-------------+-------------+
                    |    SharedArrayBuffer      |
                    |                           |
                    |  - Transform data         |
                    |  - Input state            |
                    |  - Performance metrics    |
                    +---------------------------+
```

---

## Key Conventions Summary

### Coordinate Systems

- **World Space:** Y-Up, Right-Handed
- **NDC (WebGPU):** X[-1,1], Y[-1,1], Z[0,1]
- **Texture UV:** Origin at Top-Left

### Memory Alignment

- Always use `#[repr(C)]` for GPU-shared structs
- `vec3<f32>` requires **16-byte alignment** in std140
- Use explicit padding fields, never rely on implicit padding

### G-Buffer Format

| Attachment | Format         | Content                           |
|------------|----------------|-----------------------------------|
| GBuffer0   | Rgba8Unorm     | Albedo.RGB + AO                   |
| GBuffer1   | Rgb10a2Unorm   | World Normal (octahedron encoded) |
| GBuffer2   | Rgba8Unorm     | Roughness + Metalness + Emissive  |
| Depth      | Depth32Float   | Depth                             |

---

## Getting Started

1. **Read Standards First:** Start with `01_STANDARDS/` to understand conventions
2. **Understand Architecture:** Review `02_ARCHITECTURE/` for system design
3. **Deep Dive Subsystems:** Consult `03_SUBSYSTEMS/` for implementation details
4. **Create Tasks:** Use `04_TASKS/task_template.md` for new work items

---

## Contributing

1. All code must follow conventions in `01_STANDARDS/`
2. Update relevant documentation when making changes
3. Include Pitfalls section updates for newly discovered issues
4. Run `cargo clippy` and `cargo fmt` before committing
