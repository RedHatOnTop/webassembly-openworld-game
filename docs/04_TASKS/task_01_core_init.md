# Task 01: Core Initialization

Initialize the Aether Engine foundation with AAA-ready WebGPU configuration.

---

## Task Header

| Field          | Value                                       |
|----------------|---------------------------------------------|
| **Task ID**    | TASK-0001                                   |
| **Title**      | Core Initialization (Rust + Winit + Wgpu)   |
| **Status**     | [ ] Not Started                             |
| **Priority**   | Critical                                    |
| **Estimate**   | 4 hours                                     |

---

## Goal

Initialize a Rust project with windowing (Winit) and WebGPU context (Wgpu), configured for future AAA rendering features including shadow mapping, deferred rendering, and GPU-driven terrain.

---

## Relevant Documentation

| Document | Purpose |
|----------|---------|
| [coordinate_systems.md](../01_STANDARDS/coordinate_systems.md) | NDC conventions, Y-up right-handed |
| [data_layout.md](../01_STANDARDS/data_layout.md) | Buffer alignment for future uniforms |
| [threading_model.md](../02_ARCHITECTURE/threading_model.md) | Main thread architecture |
| [engine_loop.md](../02_ARCHITECTURE/engine_loop.md) | Render loop timing |

---

## Implementation Steps

### Phase 1: Project Setup

- [ ] Create project directory and initialize Cargo
  ```bash
  cargo init aether-engine
  cd aether-engine
  ```

- [ ] Configure `Cargo.toml` with dependencies:
  ```toml
  [package]
  name = "aether-engine"
  version = "0.1.0"
  edition = "2021"
  
  [dependencies]
  winit = "0.29"
  wgpu = "0.18"
  glam = { version = "0.24", features = ["bytemuck"] }
  bytemuck = { version = "1", features = ["derive"] }
  pollster = "0.3"
  log = "0.4"
  env_logger = "0.10"
  
  [target.'cfg(target_arch = "wasm32")'.dependencies]
  console_error_panic_hook = "0.1"
  console_log = "1"
  wasm-bindgen = "0.2"
  wasm-bindgen-futures = "0.4"
  web-sys = { version = "0.3", features = ["Document", "Window", "Element"] }
  ```

- [ ] Create project structure:
  ```
  src/
    main.rs
    lib.rs
    renderer/
      mod.rs
      context.rs
  ```

---

### Phase 2: Windowing System

- [ ] Implement window creation in `main.rs`:
  ```rust
  use winit::{
      event::{Event, WindowEvent},
      event_loop::{ControlFlow, EventLoop},
      window::WindowBuilder,
  };
  
  fn main() {
      env_logger::init();
      
      let event_loop = EventLoop::new().unwrap();
      let window = WindowBuilder::new()
          .with_title("Aether Engine")
          .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
          .build(&event_loop)
          .unwrap();
      
      let mut renderer = pollster::block_on(Renderer::new(&window));
      
      event_loop.run(move |event, elwt| {
          match event {
              Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                  elwt.exit();
              }
              Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                  renderer.resize(size.width, size.height);
              }
              Event::AboutToWait => {
                  window.request_redraw();
              }
              Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                  renderer.render();
              }
              _ => {}
          }
      }).unwrap();
  }
  ```

- [ ] Structure code for future worker migration (Main Thread only handles window events)

---

### Phase 3: WebGPU Initialization (Critical)

- [ ] Create `renderer/context.rs` with GPU context:

  ```rust
  use wgpu;
  use winit::window::Window;
  
  pub struct RenderContext {
      pub surface: wgpu::Surface,
      pub device: wgpu::Device,
      pub queue: wgpu::Queue,
      pub config: wgpu::SurfaceConfiguration,
      pub depth_texture: wgpu::Texture,
      pub depth_view: wgpu::TextureView,
  }
  
  impl RenderContext {
      pub async fn new(window: &Window) -> Self {
          let size = window.inner_size();
          
          // Instance
          let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
              backends: wgpu::Backends::all(),
              ..Default::default()
          });
          
          // Surface
          let surface = unsafe { instance.create_surface(window) }.unwrap();
          
          // Adapter
          let adapter = instance
              .request_adapter(&wgpu::RequestAdapterOptions {
                  power_preference: wgpu::PowerPreference::HighPerformance,
                  compatible_surface: Some(&surface),
                  force_fallback_adapter: false,
              })
              .await
              .unwrap();
          
          log::info!("Adapter: {:?}", adapter.get_info());
          
          // Device with AAA-ready limits
          let (device, queue) = adapter
              .request_device(
                  &wgpu::DeviceDescriptor {
                      label: Some("Aether Device"),
                      features: wgpu::Features::empty()
                          | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                      limits: wgpu::Limits {
                          max_bind_groups: 8,  // Shadow maps + G-Buffer + Lighting
                          max_storage_buffers_per_shader_stage: 8,
                          max_storage_textures_per_shader_stage: 4,
                          ..wgpu::Limits::default()
                      },
                  },
                  None,
              )
              .await
              .unwrap();
          
          // Surface configuration
          let surface_caps = surface.get_capabilities(&adapter);
          let surface_format = surface_caps
              .formats
              .iter()
              .find(|f| f.is_srgb())
              .copied()
              .unwrap_or(surface_caps.formats[0]);
          
          let present_mode = if surface_caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
              wgpu::PresentMode::Mailbox
          } else {
              wgpu::PresentMode::Fifo
          };
          
          let config = wgpu::SurfaceConfiguration {
              usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
              format: surface_format,
              width: size.width,
              height: size.height,
              present_mode,
              alpha_mode: surface_caps.alpha_modes[0],
              view_formats: vec![],
          };
          surface.configure(&device, &config);
          
          // Depth texture (CRITICAL: TEXTURE_BINDING for depth reconstruction)
          let (depth_texture, depth_view) = Self::create_depth_texture(&device, size.width, size.height);
          
          Self {
              surface,
              device,
              queue,
              config,
              depth_texture,
              depth_view,
          }
      }
      
      fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
          let texture = device.create_texture(&wgpu::TextureDescriptor {
              label: Some("Depth Texture"),
              size: wgpu::Extent3d {
                  width,
                  height,
                  depth_or_array_layers: 1,
              },
              mip_level_count: 1,
              sample_count: 1,
              dimension: wgpu::TextureDimension::D2,
              format: wgpu::TextureFormat::Depth32Float,
              // CRITICAL: Both flags required for depth reconstruction
              usage: wgpu::TextureUsages::RENDER_ATTACHMENT 
                   | wgpu::TextureUsages::TEXTURE_BINDING,
              view_formats: &[],
          });
          
          let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
          
          (texture, view)
      }
      
      pub fn resize(&mut self, width: u32, height: u32) {
          if width > 0 && height > 0 {
              self.config.width = width;
              self.config.height = height;
              self.surface.configure(&self.device, &self.config);
              
              let (depth_texture, depth_view) = Self::create_depth_texture(&self.device, width, height);
              self.depth_texture = depth_texture;
              self.depth_view = depth_view;
          }
      }
  }
  ```

---

### Phase 4: Render Loop

- [ ] Implement basic renderer in `renderer/mod.rs`:

  ```rust
  mod context;
  
  use context::RenderContext;
  use winit::window::Window;
  
  pub struct Renderer {
      ctx: RenderContext,
  }
  
  impl Renderer {
      pub async fn new(window: &Window) -> Self {
          Self {
              ctx: RenderContext::new(window).await,
          }
      }
      
      pub fn resize(&mut self, width: u32, height: u32) {
          self.ctx.resize(width, height);
      }
      
      pub fn render(&mut self) {
          let output = match self.ctx.surface.get_current_texture() {
              Ok(output) => output,
              Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                  self.ctx.surface.configure(&self.ctx.device, &self.ctx.config);
                  return;
              }
              Err(e) => {
                  log::error!("Surface error: {:?}", e);
                  return;
              }
          };
          
          let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
          
          let mut encoder = self.ctx.device.create_command_encoder(
              &wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") }
          );
          
          {
              let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                  label: Some("Clear Pass"),
                  color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                      view: &view,
                      resolve_target: None,
                      ops: wgpu::Operations {
                          load: wgpu::LoadOp::Clear(wgpu::Color {
                              r: 0.1,
                              g: 0.2,
                              b: 0.3,
                              a: 1.0,
                          }),
                          store: wgpu::StoreOp::Store,
                      },
                  })],
                  depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                      view: &self.ctx.depth_view,
                      depth_ops: Some(wgpu::Operations {
                          load: wgpu::LoadOp::Clear(1.0),
                          store: wgpu::StoreOp::Store,
                      }),
                      stencil_ops: None,
                  }),
                  ..Default::default()
              });
          }
          
          self.ctx.queue.submit(std::iter::once(encoder.finish()));
          output.present();
      }
  }
  ```

---

## Future Impact Analysis

| Future Feature | Compatibility Check | Status |
|----------------|---------------------|--------|
| Shadow Mapping | `max_bind_groups: 8` supports shadow + scene | [ ] Verified |
| Depth Reconstruction | `TEXTURE_BINDING` on depth texture | [ ] Verified |
| Deferred Rendering | `Rgba16Float` via TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | [ ] Verified |
| Compute Shaders | Storage buffer limits increased | [ ] Verified |
| G-Buffer | Surface is sRGB, G-Buffer will use linear | [ ] Verified |

### Checklist

- [ ] Render pass has depth attachment (future G-Buffer pass compatible)
- [ ] Depth texture usage includes TEXTURE_BINDING
- [ ] Device limits accommodate 8 bind groups (shadow cascades + scene)
- [ ] Code structured for future worker thread migration
- [ ] Present mode prefers Mailbox for low latency

---

## Verification

### Success Criteria

- [ ] `cargo build` succeeds without errors
- [ ] `cargo run` opens a window
- [ ] Window displays colored background (RGB: 0.1, 0.2, 0.3)
- [ ] Window resizes correctly
- [ ] No validation errors in console (`RUST_LOG=wgpu=warn cargo run`)
- [ ] Close button terminates application

### Test Commands

```bash
# Build
cargo build

# Run with validation
RUST_LOG=wgpu=warn cargo run

# Check for warnings
cargo clippy
```

### Expected Result

A 1280x720 window titled "Aether Engine" with a dark blue-gray background. Console shows adapter info. No validation errors.

---

## Notes

- Depth texture is created even though we only clear - this ensures the infrastructure is in place
- `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` enables Rgba16Float sampling which is needed for G-Buffer
- Mailbox present mode reduces input latency vs Fifo
- Code is structured so `RenderContext` can be moved to a worker thread later

---

## Changelog

| Date       | Author | Change |
|------------|--------|--------|
| 2026-01-13 | Init   | Created task |
