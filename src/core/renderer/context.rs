//! WebGPU Render Context
//!
//! Manages GPU resources: Device, Queue, Surface, and Depth Texture.
//! Configured for AAA-ready features per task_01_core_init.md.

use std::sync::Arc;
use winit::window::Window;

/// Holds all WebGPU context resources.
pub struct RenderContext {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub size: (u32, u32),
}

impl RenderContext {
    /// Creates a new RenderContext with AAA-ready configuration.
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface
        let surface = instance
            .create_surface(window)
            .expect("Failed to create surface");

        // Request high-performance adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find suitable GPU adapter");

        let adapter_info = adapter.get_info();
        log::info!(
            "GPU Adapter: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );
        log::info!("Driver: {}", adapter_info.driver);

        // Request device with AAA-ready limits
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Aether Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_bind_groups: 8,
                        max_storage_buffers_per_shader_stage: 8,
                        max_storage_textures_per_shader_stage: 4,
                        max_sampled_textures_per_shader_stage: 16,
                        ..wgpu::Limits::default()
                    },
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("Failed to create GPU device");

        log::info!("Device created with AAA-ready limits");

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        log::info!("Surface format: {:?}", surface_format);

        let present_mode = if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::Mailbox)
        {
            log::info!("Present mode: Mailbox (low latency)");
            wgpu::PresentMode::Mailbox
        } else {
            log::info!("Present mode: Fifo (vsync)");
            wgpu::PresentMode::Fifo
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let (depth_texture, depth_view) = Self::create_depth_texture(&device, width, height);

        Self {
            surface,
            device,
            queue,
            config,
            depth_texture,
            depth_view,
            size: (width, height),
        }
    }

    /// Creates depth texture with RENDER_ATTACHMENT and TEXTURE_BINDING.
    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        log::debug!("Depth texture created: {}x{} Depth32Float", width, height);

        (texture, view)
    }

    /// Resizes surface and depth texture.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 && (width != self.size.0 || height != self.size.1) {
            self.size = (width, height);
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);

            let (depth_texture, depth_view) =
                Self::create_depth_texture(&self.device, width, height);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;

            log::debug!("Resized to {}x{}", width, height);
        }
    }
}
