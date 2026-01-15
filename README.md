# Aether Engine

A GPU-driven procedural terrain engine built with Rust and WebGPU (wgpu), targeting both native platforms and WebAssembly.

## Overview

Aether Engine is an experimental rendering engine focused on GPU-compute-driven procedural generation. The engine demonstrates modern graphics programming techniques including:

- **GPU Compute Shaders** for terrain generation and animation
- **Infinite Terrain System** with seamless chunk streaming
- **WebGPU/Vulkan Backend** via wgpu for cross-platform support
- **WebAssembly Target** for browser-based deployment

## Features

### Implemented

- [x] WebGPU rendering context with Vulkan backend
- [x] Fixed timestep game loop (60 Hz)
- [x] 3D camera with perspective projection
- [x] GPU compute shader pipeline (Compute -> Storage -> Render)
- [x] Procedural terrain mesh (100x100 grid, 58,806 triangles)
- [x] Real-time normal calculation and directional lighting
- [x] Dual build targets: Native (Windows/Linux/macOS) and WASM

### Planned

- [ ] Infinite terrain scrolling with chunk offset
- [ ] Wireframe debug mode (F1 toggle)
- [ ] Double-sided rendering (F2 toggle)
- [ ] WASD camera movement
- [ ] Noise-based terrain generation
- [ ] LOD (Level of Detail) system
- [ ] Deferred rendering pipeline

## Requirements

### Build Dependencies

- **Rust** 1.75+ (stable)
- **wgpu** 23.x
- **winit** 0.30.x
- **glam** (math library)
- **bytemuck** (GPU memory mapping)

### Runtime Requirements

- Vulkan 1.2+ capable GPU (native)
- WebGPU-compatible browser (WASM)

## Building

### Native Build

```bash
cargo build --release
cargo run --release
```

### WebAssembly Build

```bash
# Install wasm target if not already installed
rustup target add wasm32-unknown-unknown

# Build for WASM
cargo build --target wasm32-unknown-unknown --release

# Use wasm-bindgen for browser deployment
wasm-bindgen --out-dir web/pkg --target web target/wasm32-unknown-unknown/release/aether_engine.wasm
```

## Controls

| Key | Action |
|-----|--------|
| W | Move camera forward |
| S | Move camera backward |
| A | Strafe left |
| D | Strafe right |
| F1 | Toggle wireframe mode |
| F2 | Toggle double-sided rendering |
| ESC | Exit |

## Project Structure

```
aether-engine/
├── src/
│   ├── main.rs              # Entry point, event loop
│   ├── core/
│   │   ├── mod.rs
│   │   ├── time.rs          # Platform-agnostic timing
│   │   └── renderer/
│   │       ├── mod.rs       # Main renderer
│   │       ├── context.rs   # WebGPU context setup
│   │       ├── camera.rs    # 3D camera system
│   │       ├── geometry.rs  # Vertex definitions
│   │       └── terrain.rs   # GPU terrain system
│   └── game/
│       ├── mod.rs
│       └── state.rs         # Game state management
├── assets/
│   └── shaders/
│       ├── cube.wgsl        # Cube rendering shader
│       └── terrain.wgsl     # Terrain compute/render shader
├── docs/
│   ├── 01_STANDARDS/        # Coding standards
│   ├── 02_ARCHITECTURE/     # System architecture
│   ├── 03_SUBSYSTEMS/       # Subsystem documentation
│   └── 04_TASKS/            # Task specifications
└── Cargo.toml
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- **Standards**: Coordinate systems, data layout conventions
- **Architecture**: Engine loop, threading model
- **Subsystems**: Renderer, terrain compute system
- **Tasks**: Implementation specifications for each feature

## Technical Details

### Coordinate System

- **Handedness**: Right-handed
- **Up Vector**: Y-up
- **Front Face**: Counter-clockwise (CCW)
- **Depth Range**: [0, 1] (WebGPU/Vulkan convention)

### Terrain System

- **Grid Size**: 100x100 vertices (10,000 total)
- **Index Count**: 58,806 (99x99 quads, 2 triangles each)
- **Vertex Format**: 32 bytes (position vec4 + normal vec4)
- **Compute Workgroup**: 8x8x1

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [wgpu](https://wgpu.rs/) - Safe and portable GPU abstraction
- [winit](https://github.com/rust-windowing/winit) - Cross-platform window handling
- [glam](https://github.com/bitshifter/glam-rs) - Fast math library for games
- [Kenney](https://kenney.nl/) - Free game assets (Nature Kit, Platformer Kit)
