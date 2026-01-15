//! Renderer Module
//!
//! Handles frame rendering using WebGPU.
//! Implements the 3D rendering pipeline with:
//! - Pulsing background color (visual game loop verification)
//! - 3D camera with perspective projection
//! - GPU-driven terrain mesh (Compute -> Storage -> Render pattern)
//! - 5-Channel Climate Model for procedural continental terrain
//! - Directional lighting for 3D depth perception
//! - Infinite terrain scrolling with camera movement (WASD)
//! - Terrain debug modes: Normal (F1), Continentalness (F2), Erosion (F3), Peaks (F4)
//! - Render debug modes: Wireframe (F5), Double-sided (F6)
//! - Triplanar texture mapping with TextureArray

mod context;
pub mod geometry;
pub mod camera;
pub mod terrain;
pub mod texture;

use std::sync::Arc;

use context::RenderContext;
use geometry::{Vertex, CUBE_VERTICES, CUBE_INDICES, CUBE_INDEX_COUNT};
use camera::{Camera, CameraUniform};
use terrain::{TerrainSystem, INDEX_COUNT};
use wgpu::util::DeviceExt;
use winit::window::Window;
use winit::keyboard::{KeyCode, PhysicalKey};
use glam::Mat4;

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
/// Manages the 3D rendering pipeline with camera, geometry, and GPU resources.
/// Also manages the GPU-driven terrain mesh system.
pub struct Renderer {
    /// WebGPU context (device, queue, surface, etc.)
    ctx: RenderContext,
    /// 3D perspective camera.
    camera: Camera,
    /// Camera uniform data for GPU.
    camera_uniform: CameraUniform,
    /// Camera uniform buffer.
    camera_buffer: wgpu::Buffer,
    /// Camera bind group.
    camera_bind_group: wgpu::BindGroup,
    /// Camera bind group layout (shared with terrain system).
    camera_bind_group_layout: wgpu::BindGroupLayout,
    /// Render pipeline for cube drawing.
    pipeline: wgpu::RenderPipeline,
    /// Vertex buffer containing cube geometry.
    vertex_buffer: wgpu::Buffer,
    /// Index buffer for cube geometry.
    index_buffer: wgpu::Buffer,
    /// GPU-driven terrain mesh system.
    terrain_system: TerrainSystem,
    /// Debug rendering state.
    debug_state: DebugState,
    /// Wireframe terrain render pipeline (optional, requires POLYGON_MODE_LINE).
    terrain_wireframe_pipeline: Option<wgpu::RenderPipeline>,
    /// Double-sided terrain render pipeline (no backface culling).
    terrain_double_sided_pipeline: wgpu::RenderPipeline,
    /// Movement input state: (forward, right) - accumulated per frame.
    movement_input: (f32, f32),
}

impl Renderer {
    /// Creates a new Renderer with initialized WebGPU context and 3D pipeline.
    ///
    /// Sets up:
    /// - WebGPU context (device, queue, surface)
    /// - 3D camera with perspective projection
    /// - Cube shader and render pipeline
    /// - Vertex and index buffers for cube geometry
    /// - Terrain mesh system with compute shaders
    pub async fn new(window: Arc<Window>) -> Self {
        let ctx = RenderContext::new(window).await;

        // Initialize camera with correct aspect ratio
        let aspect = ctx.size.0 as f32 / ctx.size.1 as f32;
        let camera = Camera::new(aspect);
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

        // Create render pipeline
        let pipeline = Self::create_pipeline(&ctx.device, ctx.config.format, &camera_bind_group_layout);
        log::info!("3D render pipeline created (CCW front face, back-face culling, depth test)");

        // Create vertex buffer
        let vertex_buffer = Self::create_vertex_buffer(&ctx.device);
        log::info!("Cube vertex buffer created (24 vertices)");

        // Create index buffer
        let index_buffer = Self::create_index_buffer(&ctx.device);
        log::info!("Cube index buffer created (36 indices)");

        // Create terrain system (GPU-driven mesh generation with triplanar texturing)
        let terrain_system = TerrainSystem::new(
            &ctx.device,
            &ctx.queue,
            ctx.config.format,
            &camera_bind_group_layout,
        );
        log::info!(
            "Terrain system created ({} vertices, {} indices, triplanar texturing)",
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
        log::info!("Debug terrain pipelines created (wireframe: {}, double-sided: ready)",
            if ctx.polygon_mode_supported { "ready" } else { "unavailable" });

        Self {
            ctx,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_bind_group_layout,
            pipeline,
            vertex_buffer,
            index_buffer,
            terrain_system,
            debug_state: DebugState::default(),
            terrain_wireframe_pipeline,
            terrain_double_sided_pipeline,
            movement_input: (0.0, 0.0),
        }
    }

    /// Creates the render pipeline for 3D cube rendering.
    ///
    /// Configuration per `coordinate_systems.md`:
    /// - Front face: CCW (Counter-Clockwise)
    /// - Cull mode: Back (back-facing triangles are culled)
    /// - Depth test: Enabled (Less comparison)
    fn create_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        // Load shader from embedded WGSL source
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cube Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/cube.wgsl").into(),
            ),
        });

        // Create pipeline layout with camera bind group
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cube Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cube Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
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
                // CCW winding order is front-facing (per coordinate_systems.md)
                front_face: wgpu::FrontFace::Ccw,
                // Cull back-facing triangles
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                // Closer objects (smaller Z) pass the test
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

    /// Creates the vertex buffer containing cube geometry.
    fn create_vertex_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Vertex Buffer"),
            contents: bytemuck::cast_slice(CUBE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }

    /// Creates the index buffer for cube geometry.
    fn create_index_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Index Buffer"),
            contents: bytemuck::cast_slice(CUBE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        })
    }

    /// Creates a terrain render pipeline with specified polygon mode and culling.
    ///
    /// Used to create the main pipeline, wireframe pipeline, and double-sided pipeline.
    /// Includes bind group 3 for render uniforms (debug mode support).
    fn create_terrain_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        polygon_mode: wgpu::PolygonMode,
        cull_mode: Option<wgpu::Face>,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Shader (Debug)"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/terrain.wgsl").into(),
            ),
        });

        // Render storage bind group layout (group 0)
        let render_storage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Debug Render Storage Layout"),
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
            label: Some("Terrain Debug Texture Layout"),
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

        // Render uniform bind group layout (group 3) - for debug mode access in fragment shader
        let render_uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Debug Render Uniform Layout"),
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
            label: Some("Terrain Debug Pipeline Layout"),
            // Group 0: Vertex storage, Group 1: Camera, Group 2: Textures, Group 3: Render uniforms
            bind_group_layouts: &[&render_storage_layout, camera_bind_group_layout, &render_texture_layout, &render_uniform_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terrain Debug Pipeline"),
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
    ///
    /// Updates surface configuration, depth texture, and camera aspect ratio.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.ctx.resize(width, height);
        // Update camera aspect ratio to prevent distortion
        self.camera.update_aspect(width, height);
    }

    /// Renders a single frame.
    ///
    /// Pipeline:
    /// 1. Update uniforms (camera, terrain time)
    /// 2. Compute Pass: Generate indices (one-time) + Update terrain (per-frame)
    /// 3. Render Pass: Draw terrain mesh + Draw reference cube
    ///
    /// # Arguments
    /// * `state` - The current game state for visual representation.
    /// * `_alpha` - Interpolation factor (0.0-1.0) for smooth rendering between fixed updates.
    ///
    /// # Returns
    /// `Ok(())` on success, or a `wgpu::SurfaceError` on failure.
    pub fn render(&mut self, state: &GameState, _alpha: f32) -> Result<(), wgpu::SurfaceError> {
        // Acquire the next frame
        let output = match self.ctx.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.ctx
                    .surface
                    .configure(&self.ctx.device, &self.ctx.config);
                return Ok(());
            }
            Err(e) => return Err(e),
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Calculate pulsing background color based on game state time.
        // This provides visual verification that the game loop is working.
        let pulse = (state.time.sin() * 0.5 + 0.5) as f64;
        let clear_color = wgpu::Color {
            r: 0.1,
            g: 0.1 + pulse * 0.2,
            b: 0.2 + pulse * 0.3,
            a: 1.0,
        };

        // Create model matrix for cube (rotate over time, elevated above terrain)
        // Position cube at Y=3 to float above the terrain waves
        let translation = Mat4::from_translation(glam::Vec3::new(0.0, 3.0, 0.0));
        let rotation_y = Mat4::from_rotation_y(state.time);
        let rotation_x = Mat4::from_rotation_x(state.time * 0.3);
        let model = translation * rotation_y * rotation_x;

        // Calculate MVP matrix for cube: Projection * View * Model
        let cube_view_proj = self.camera.build_view_projection_matrix() * model;

        // Update camera uniform for cube
        self.camera_uniform.update_view_proj(cube_view_proj);

        // Update terrain system time and write uniforms to both buffers
        self.terrain_system.update_time(state.time);
        self.terrain_system.write_uniforms(&self.ctx.queue);

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

                // Dispatch for (GRID_SIZE - 1) x (GRID_SIZE - 1) cells
                let index_workgroups = TerrainSystem::index_workgroup_count();
                compute_pass.dispatch_workgroups(index_workgroups, index_workgroups, 1);

                self.terrain_system.indices_generated = true;
                log::info!("Terrain indices generated ({} indices)", INDEX_COUNT);
            }

            // Per-frame position and normal update
            compute_pass.set_pipeline(&self.terrain_system.terrain_update_pipeline);
            compute_pass.set_bind_group(0, &self.terrain_system.compute_bind_group_0, &[]);
            compute_pass.set_bind_group(1, &self.terrain_system.compute_bind_group_1, &[]);

            // Dispatch for GRID_SIZE x GRID_SIZE vertices
            let vertex_workgroups = TerrainSystem::vertex_workgroup_count();
            compute_pass.dispatch_workgroups(vertex_workgroups, vertex_workgroups, 1);
        }

        // =====================================================================
        // Render Pass: Draw Scene
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

            // -----------------------------------------------------------------
            // Draw Terrain Mesh (58,806 indices, solid triangles with lighting)
            // -----------------------------------------------------------------
            // Update camera uniform for terrain (no model transform)
            let terrain_view_proj = self.camera.build_view_projection_matrix();
            let mut terrain_camera_uniform = CameraUniform::new();
            terrain_camera_uniform.update_view_proj(terrain_view_proj);
            self.ctx.queue.write_buffer(
                &self.camera_buffer,
                0,
                bytemuck::cast_slice(&[terrain_camera_uniform]),
            );

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

            // -----------------------------------------------------------------
            // Draw Cube (floating above terrain)
            // -----------------------------------------------------------------
            // Restore cube's camera uniform
            self.ctx.queue.write_buffer(
                &self.camera_buffer,
                0,
                bytemuck::cast_slice(&[self.camera_uniform]),
            );

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..CUBE_INDEX_COUNT, 0, 0..1);
        }

        // Submit commands and present
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Handles keyboard input for debug toggles and camera movement.
    ///
    /// # Key Bindings
    /// - F1: Terrain visualization - Normal (textures + lighting)
    /// - F2: Terrain visualization - Continentalness (blue=ocean, green=coast, brown=land)
    /// - F3: Terrain visualization - Erosion (red=jagged, green=flat)
    /// - F4: Terrain visualization - Peaks & Valleys (white=ridges, black=valleys)
    /// - F5: Toggle wireframe mode
    /// - F6: Toggle double-sided rendering
    /// - WASD: Camera movement
    ///
    /// # Arguments
    /// * `key_code` - The physical key that was pressed.
    /// * `pressed` - Whether the key was pressed (true) or released (false).
    ///
    /// # Returns
    /// `true` if the input was handled, `false` otherwise.
    pub fn handle_key(&mut self, key_code: PhysicalKey, pressed: bool) -> bool {
        match key_code {
            // Terrain visualization modes (F1-F4)
            PhysicalKey::Code(KeyCode::F1) if pressed => {
                self.terrain_system.set_debug_mode(0); // Normal
                true
            }
            PhysicalKey::Code(KeyCode::F2) if pressed => {
                self.terrain_system.set_debug_mode(1); // Continentalness
                true
            }
            PhysicalKey::Code(KeyCode::F3) if pressed => {
                self.terrain_system.set_debug_mode(2); // Erosion
                true
            }
            PhysicalKey::Code(KeyCode::F4) if pressed => {
                self.terrain_system.set_debug_mode(3); // Peaks & Valleys
                true
            }
            // Render debug toggles (F5-F6)
            PhysicalKey::Code(KeyCode::F5) if pressed => {
                if self.terrain_wireframe_pipeline.is_some() {
                    self.debug_state.wireframe = !self.debug_state.wireframe;
                    log::info!("Wireframe mode: {}", if self.debug_state.wireframe { "ON" } else { "OFF" });
                } else {
                    log::warn!("Wireframe mode unavailable (POLYGON_MODE_LINE not supported)");
                }
                true
            }
            PhysicalKey::Code(KeyCode::F6) if pressed => {
                self.debug_state.double_sided = !self.debug_state.double_sided;
                log::info!("Double-sided mode: {}", if self.debug_state.double_sided { "ON" } else { "OFF" });
                true
            }
            // WASD movement
            PhysicalKey::Code(KeyCode::KeyW) => {
                self.movement_input.0 = if pressed { 1.0 } else { 0.0 };
                true
            }
            PhysicalKey::Code(KeyCode::KeyS) => {
                self.movement_input.0 = if pressed { -1.0 } else { 0.0 };
                true
            }
            PhysicalKey::Code(KeyCode::KeyA) => {
                self.movement_input.1 = if pressed { -1.0 } else { 0.0 };
                true
            }
            PhysicalKey::Code(KeyCode::KeyD) => {
                self.movement_input.1 = if pressed { 1.0 } else { 0.0 };
                true
            }
            _ => false,
        }
    }

    /// Updates camera movement based on accumulated input.
    ///
    /// Called once per frame to apply smooth camera movement.
    ///
    /// # Arguments
    /// * `delta_time` - Time elapsed since last frame.
    pub fn update_movement(&mut self, delta_time: f32) {
        const MOVE_SPEED: f32 = 10.0; // Units per second

        let (forward, right) = self.movement_input;
        
        if forward != 0.0 {
            self.camera.move_forward(forward * MOVE_SPEED * delta_time);
        }
        if right != 0.0 {
            self.camera.move_right(right * MOVE_SPEED * delta_time);
        }

        // Update terrain chunk offset for infinite scrolling
        let (cam_x, cam_z) = self.camera.get_xz_position();
        self.terrain_system.update_camera_position(cam_x, cam_z);
    }
}
