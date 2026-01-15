//! Renderer Module
//!
//! Handles frame rendering using WebGPU.
//! Implements the 3D rendering pipeline with:
//! - Free camera (WASD + mouse look)
//! - GPU-driven terrain mesh (Compute -> Storage -> Render pattern)
//! - 5-Channel Climate Model for procedural continental terrain
//! - Directional lighting for 3D depth perception
//! - Infinite terrain scrolling with camera movement
//! - Terrain debug modes: Normal (F1), Continentalness (F2), Erosion (F3), Peaks (F4)
//! - Render debug modes: Wireframe (F5), Double-sided (F6)

mod context;
pub mod geometry;
pub mod camera;
pub mod terrain;
pub mod texture;

use std::sync::Arc;

use context::RenderContext;
use camera::{Camera, CameraController, CameraUniform};
use terrain::{TerrainSystem, INDEX_COUNT};
use wgpu::util::DeviceExt;
use winit::window::Window;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::game::GameState;

// ============================================================================
// Debug State
// ============================================================================

/// Debug rendering state.
#[derive(Debug, Clone, Copy)]
pub struct DebugState {
    /// Wireframe mode enabled (F5 toggle).
    pub wireframe: bool,
    /// Double-sided rendering enabled (F6 toggle).
    pub double_sided: bool,
}

impl Default for DebugState {
    fn default() -> Self {
        Self {
            wireframe: false,
            double_sided: false,
        }
    }
}

// ============================================================================
// Renderer
// ============================================================================

/// Main renderer struct.
///
/// Manages the 3D rendering pipeline with camera, terrain, and GPU resources.
pub struct Renderer {
    /// WebGPU context (device, queue, surface, etc.)
    ctx: RenderContext,
    /// Free camera with position and rotation.
    pub camera: Camera,
    /// Camera controller for input handling.
    pub camera_controller: CameraController,
    /// Camera uniform data for GPU.
    camera_uniform: CameraUniform,
    /// Camera uniform buffer.
    camera_buffer: wgpu::Buffer,
    /// Camera bind group.
    camera_bind_group: wgpu::BindGroup,
    /// GPU-driven terrain mesh system.
    terrain_system: TerrainSystem,
    /// Debug rendering state.
    debug_state: DebugState,
    /// Wireframe terrain render pipeline (optional, requires POLYGON_MODE_LINE).
    terrain_wireframe_pipeline: Option<wgpu::RenderPipeline>,
    /// Double-sided terrain render pipeline (no backface culling).
    terrain_double_sided_pipeline: wgpu::RenderPipeline,
}

impl Renderer {
    /// Creates a new Renderer with initialized WebGPU context and terrain pipeline.
    pub async fn new(window: Arc<Window>) -> Self {
        let ctx = RenderContext::new(window).await;

        // Initialize free camera
        let aspect = ctx.size.0 as f32 / ctx.size.1 as f32;
        let camera = Camera::new(aspect);
        let camera_controller = CameraController::new();
        let camera_uniform = CameraUniform::new();

        // Create camera uniform buffer
        let camera_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout for camera
        let camera_bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create bind group for camera
        let camera_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Create terrain system
        let terrain_system = TerrainSystem::new(
            &ctx.device,
            &ctx.queue,
            ctx.config.format,
            &camera_bind_group_layout,
        );
        log::info!(
            "Terrain system created ({} vertices, {} indices)",
            terrain::VERTEX_COUNT,
            INDEX_COUNT
        );

        // Create wireframe pipeline (optional, requires POLYGON_MODE_LINE)
        let terrain_wireframe_pipeline = if ctx.polygon_mode_supported {
            Some(Self::create_terrain_pipeline(
                &ctx.device,
                ctx.config.format,
                &camera_bind_group_layout,
                wgpu::PolygonMode::Line,
                Some(wgpu::Face::Back),
            ))
        } else {
            log::warn!("Wireframe mode unavailable (POLYGON_MODE_LINE not supported)");
            None
        };

        // Create double-sided pipeline (no backface culling)
        let terrain_double_sided_pipeline = Self::create_terrain_pipeline(
            &ctx.device,
            ctx.config.format,
            &camera_bind_group_layout,
            wgpu::PolygonMode::Fill,
            None, // No culling
        );

        log::info!("Terrain pipelines created (wireframe: {}, double-sided: ready)",
            if ctx.polygon_mode_supported { "ready" } else { "unavailable" });

        Self {
            ctx,
            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            terrain_system,
            debug_state: DebugState::default(),
            terrain_wireframe_pipeline,
            terrain_double_sided_pipeline,
        }
    }

    /// Creates a terrain render pipeline with specified polygon mode and culling.
    fn create_terrain_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        polygon_mode: wgpu::PolygonMode,
        cull_mode: Option<wgpu::Face>,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/terrain.wgsl").into(),
            ),
        });

        // Render storage bind group layout (group 0)
        let render_storage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Render Storage Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Texture bind group layout (group 2)
        let render_texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Texture Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Render uniform bind group layout (group 3)
        let render_uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Render Uniform Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terrain Pipeline Layout"),
            bind_group_layouts: &[&render_storage_layout, camera_bind_group_layout, &render_texture_layout, &render_uniform_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terrain Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                polygon_mode,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
    }

    /// Handles window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.ctx.resize(width, height);
        self.camera.update_aspect(width, height);
    }

    /// Renders a single frame.
    pub fn render(&mut self, state: &GameState, _alpha: f32) -> Result<(), wgpu::SurfaceError> {
        // Acquire the next frame
        let output = match self.ctx.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.ctx.surface.configure(&self.ctx.device, &self.ctx.config);
                return Ok(());
            }
            Err(e) => return Err(e),
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Sky color (gradient based on camera pitch)
        let sky_t = (self.camera.pitch / std::f32::consts::FRAC_PI_2 + 1.0) * 0.5;
        let clear_color = wgpu::Color {
            r: 0.4 + sky_t as f64 * 0.2,
            g: 0.6 + sky_t as f64 * 0.15,
            b: 0.85 + sky_t as f64 * 0.1,
            a: 1.0,
        };

        // Update camera view-projection matrix
        let view_proj = self.camera.build_view_projection_matrix();
        self.camera_uniform.update_view_proj(view_proj);
        self.ctx.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update terrain uniform with camera position for chunk offset
        let (cam_x, cam_z) = self.camera.get_xz_position();
        self.terrain_system.update_camera_position(cam_x, cam_z);
        self.terrain_system.update_time(state.time);
        self.terrain_system.write_uniforms(&self.ctx.queue);

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // =====================================================================
        // Compute Pass: Terrain Generation
        // =====================================================================
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Terrain Compute Pass"),
                timestamp_writes: None,
            });

            // One-time index generation
            if !self.terrain_system.indices_generated {
                compute_pass.set_pipeline(&self.terrain_system.index_gen_pipeline);
                compute_pass.set_bind_group(0, &self.terrain_system.compute_bind_group_0, &[]);
                compute_pass.set_bind_group(1, &self.terrain_system.compute_bind_group_1, &[]);
                let index_workgroups = TerrainSystem::index_workgroup_count();
                compute_pass.dispatch_workgroups(index_workgroups, index_workgroups, 1);
                self.terrain_system.indices_generated = true;
                log::info!("Terrain indices generated ({} indices)", INDEX_COUNT);
            }

            // Per-frame position and normal update
            compute_pass.set_pipeline(&self.terrain_system.terrain_update_pipeline);
            compute_pass.set_bind_group(0, &self.terrain_system.compute_bind_group_0, &[]);
            compute_pass.set_bind_group(1, &self.terrain_system.compute_bind_group_1, &[]);
            let vertex_workgroups = TerrainSystem::vertex_workgroup_count();
            compute_pass.dispatch_workgroups(vertex_workgroups, vertex_workgroups, 1);
        }

        // =====================================================================
        // Render Pass: Draw Terrain
        // =====================================================================
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
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
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Select terrain pipeline based on debug state
            let terrain_pipeline = if self.debug_state.wireframe {
                self.terrain_wireframe_pipeline.as_ref().unwrap_or(&self.terrain_system.render_pipeline)
            } else if self.debug_state.double_sided {
                &self.terrain_double_sided_pipeline
            } else {
                &self.terrain_system.render_pipeline
            };

            render_pass.set_pipeline(terrain_pipeline);
            render_pass.set_bind_group(0, &self.terrain_system.render_bind_group_0, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(2, &self.terrain_system.render_bind_group_2, &[]);
            render_pass.set_bind_group(3, &self.terrain_system.render_bind_group_3, &[]);
            render_pass.set_index_buffer(
                self.terrain_system.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.draw_indexed(0..INDEX_COUNT, 0, 0..1);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Handles keyboard input for debug toggles.
    pub fn handle_key(&mut self, key_code: PhysicalKey, pressed: bool) -> bool {
        match key_code {
            // Terrain visualization modes (F1-F4)
            PhysicalKey::Code(KeyCode::F1) if pressed => {
                self.terrain_system.set_debug_mode(0);
                true
            }
            PhysicalKey::Code(KeyCode::F2) if pressed => {
                self.terrain_system.set_debug_mode(1);
                true
            }
            PhysicalKey::Code(KeyCode::F3) if pressed => {
                self.terrain_system.set_debug_mode(2);
                true
            }
            PhysicalKey::Code(KeyCode::F4) if pressed => {
                self.terrain_system.set_debug_mode(3);
                true
            }
            // Render debug toggles (F5-F6)
            PhysicalKey::Code(KeyCode::F5) if pressed => {
                if self.terrain_wireframe_pipeline.is_some() {
                    self.debug_state.wireframe = !self.debug_state.wireframe;
                    log::info!("Wireframe mode: {}", if self.debug_state.wireframe { "ON" } else { "OFF" });
                }
                true
            }
            PhysicalKey::Code(KeyCode::F6) if pressed => {
                self.debug_state.double_sided = !self.debug_state.double_sided;
                log::info!("Double-sided mode: {}", if self.debug_state.double_sided { "ON" } else { "OFF" });
                true
            }
            // Forward to camera controller for WASD movement
            PhysicalKey::Code(code) => {
                self.camera_controller.process_keyboard(code, pressed)
            }
            _ => false,
        }
    }

    /// Handles mouse button input.
    pub fn handle_mouse_button(&mut self, button: winit::event::MouseButton, pressed: bool) {
        self.camera_controller.process_mouse_button(button, pressed);
    }

    /// Handles mouse motion input.
    pub fn handle_mouse_motion(&mut self, delta: (f64, f64)) {
        self.camera_controller.process_mouse_motion(delta);
    }

    /// Updates camera movement based on controller input.
    pub fn update_movement(&mut self, delta_time: f32) {
        self.camera.update(&mut self.camera_controller, delta_time);
    }
}
