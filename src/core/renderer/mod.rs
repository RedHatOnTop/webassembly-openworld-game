//! Renderer Module
//!
//! Handles frame rendering using WebGPU.
//! Implements the 3D rendering pipeline with:
//! - Pulsing background color (visual game loop verification)
//! - 3D camera with perspective projection
//! - GPU-driven terrain mesh (Compute -> Storage -> Render pattern)
//! - Directional lighting for 3D depth perception

mod context;
pub mod geometry;
pub mod camera;
pub mod terrain;

use std::sync::Arc;

use context::RenderContext;
use geometry::{Vertex, CUBE_VERTICES, CUBE_INDICES, CUBE_INDEX_COUNT};
use camera::{Camera, CameraUniform};
use terrain::{TerrainSystem, INDEX_COUNT};
use wgpu::util::DeviceExt;
use winit::window::Window;
use glam::Mat4;

use crate::game::GameState;

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

        // Create terrain system (GPU-driven mesh generation)
        let terrain_system = TerrainSystem::new(
            &ctx.device,
            ctx.config.format,
            &camera_bind_group_layout,
        );
        log::info!(
            "Terrain system created ({} vertices, {} indices, compute shaders)",
            terrain::VERTEX_COUNT,
            INDEX_COUNT
        );

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

        // Update terrain system time
        self.terrain_system.update_time(state.time);
        self.ctx.queue.write_buffer(
            &self.terrain_system.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.terrain_system.uniforms]),
        );

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

            render_pass.set_pipeline(&self.terrain_system.render_pipeline);
            render_pass.set_bind_group(0, &self.terrain_system.render_bind_group_0, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
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
}
