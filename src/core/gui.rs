//! GUI System - Developer Debug Tools
//!
//! Provides an immediate mode GUI overlay using egui for:
//! - Real-time performance stats (FPS, coordinates)
//! - Drag-and-drop GLTF testing
//! - Transform controls for debug models
//!
//! Reference: docs/04_TASKS/task_13_5_debug_tools.md

use std::path::PathBuf;
use egui::Context;
use glam::{Vec3, Mat4, Quat};
use winit::window::Window;
use winit::event::WindowEvent;

use crate::core::assets::{load_gltf_debug, LoadedMesh};

// ============================================================================
// Debug Model State
// ============================================================================

/// Transform parameters for the debug model.
#[derive(Clone, Debug)]
pub struct DebugModelParams {
    pub scale: f32,
    pub y_offset: f32,
    pub rotation_y: f32,  // Degrees
}

impl Default for DebugModelParams {
    fn default() -> Self {
        Self {
            scale: 1.0,
            y_offset: 0.0,
            rotation_y: 0.0,
        }
    }
}

/// A debug model loaded via drag-and-drop.
pub struct DebugModel {
    pub mesh: LoadedMesh,
    pub params: DebugModelParams,
    pub spawn_position: Vec3,  // World position where it was spawned
    pub file_name: String,
}

impl DebugModel {
    /// Calculates the world transform matrix from params.
    pub fn get_transform(&self) -> [[f32; 4]; 4] {
        let position = self.spawn_position + Vec3::new(0.0, self.params.y_offset, 0.0);
        let rotation = Quat::from_rotation_y(self.params.rotation_y.to_radians());
        let scale = Vec3::splat(self.params.scale);
        
        Mat4::from_scale_rotation_translation(scale, rotation, position).to_cols_array_2d()
    }
}

// ============================================================================
// GUI System
// ============================================================================

/// Immediate mode GUI system using egui.
pub struct GuiSystem {
    /// egui context.
    ctx: Context,
    /// egui-winit state (handles input).
    state: egui_winit::State,
    /// egui-wgpu renderer.
    renderer: egui_wgpu::Renderer,
    
    // Debug state
    pub fps: f32,
    pub frame_time_ms: f32,
    pub camera_pos: Vec3,
    pub biome_name: String,
    
    /// Currently loaded debug model (via drag-drop).
    pub debug_model: Option<DebugModel>,
    
    /// Editor mode: true = show cursor, GUI active; false = game mode.
    pub editor_mode: bool,
    
    /// Pending file path from drag-drop event.
    pending_file: Option<PathBuf>,
}

impl GuiSystem {
    /// Creates a new GUI system.
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        window: &Window,
    ) -> Self {
        let ctx = Context::default();
        
        // Configure style
        let mut style = egui::Style::default();
        style.visuals.window_fill = egui::Color32::from_rgba_unmultiplied(20, 20, 30, 230);
        style.visuals.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(80));
        ctx.set_style(style);
        
        // Initialize winit state
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        
        // Initialize wgpu renderer
        let renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            None,
            1,
            false,
        );
        
        log::info!("GUI system initialized");
        
        Self {
            ctx,
            state,
            renderer,
            fps: 0.0,
            frame_time_ms: 0.0,
            camera_pos: Vec3::ZERO,
            biome_name: "Unknown".to_string(),
            debug_model: None,
            editor_mode: false,
            pending_file: None,
        }
    }
    
    /// Handles window events for egui input.
    /// Returns true if egui consumed the event.
    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        // Handle file drop
        if let WindowEvent::DroppedFile(path) = event {
            log::info!("File dropped: {:?}", path);
            self.pending_file = Some(path.clone());
            return true;
        }
        
        // Forward to egui
        let response = self.state.on_window_event(window, event);
        response.consumed
    }
    
    /// Toggles editor mode on/off.
    pub fn toggle_editor_mode(&mut self) {
        self.editor_mode = !self.editor_mode;
        log::info!("Editor mode: {}", if self.editor_mode { "ON" } else { "OFF" });
    }
    
    /// Process any pending file drops.
    pub fn process_pending_files(
        &mut self,
        device: &wgpu::Device,
        camera_pos: Vec3,
        camera_forward: Vec3,
    ) {
        if let Some(path) = self.pending_file.take() {
            // Get file name
            let file_name = path.file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            
            // Check extension
            let ext = path.extension()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            
            if ext != "glb" && ext != "gltf" {
                log::warn!("Unsupported file type: {}", ext);
                return;
            }
            
            // Try to load
            let path_str = path.to_string_lossy().to_string();
            let mesh = load_gltf_debug(device, &path_str);
            
            // Spawn 5 meters in front of camera, on ground
            let spawn_pos = camera_pos + camera_forward * 5.0;
            let spawn_pos = Vec3::new(spawn_pos.x, camera_pos.y - 5.0, spawn_pos.z);
            
            log::info!(
                "Loaded debug model '{}' at ({:.1}, {:.1}, {:.1})",
                file_name, spawn_pos.x, spawn_pos.y, spawn_pos.z
            );
            
            self.debug_model = Some(DebugModel {
                mesh,
                params: DebugModelParams::default(),
                spawn_position: spawn_pos,
                file_name,
            });
        }
    }
    
    /// Draws the UI and returns paint jobs.
    pub fn draw_ui(&mut self, window: &Window) -> egui::FullOutput {
        let raw_input = self.state.take_egui_input(window);
        
        self.ctx.run(raw_input, |ctx| {
            // Only show UI in editor mode
            if !self.editor_mode {
                // Minimal stats overlay
                egui::Area::new(egui::Id::new("stats_minimal"))
                    .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
                    .show(ctx, |ui| {
                        ui.label(
                            egui::RichText::new(format!("FPS: {:.0}", self.fps))
                                .color(egui::Color32::GREEN)
                                .size(14.0)
                        );
                        ui.label(
                            egui::RichText::new("[F1] Editor Mode")
                                .color(egui::Color32::GRAY)
                                .size(11.0)
                        );
                    });
                return;
            }
            
            // === Stats Panel (Top-Left) ===
            egui::Window::new("Stats")
                .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
                .resizable(false)
                .collapsible(false)
                .title_bar(true)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("FPS:");
                        ui.colored_label(
                            if self.fps >= 55.0 { egui::Color32::GREEN }
                            else if self.fps >= 30.0 { egui::Color32::YELLOW }
                            else { egui::Color32::RED },
                            format!("{:.0}", self.fps)
                        );
                        ui.label(format!("({:.1}ms)", self.frame_time_ms));
                    });
                    
                    ui.separator();
                    
                    ui.label(format!(
                        "Position: ({:.1}, {:.1}, {:.1})",
                        self.camera_pos.x, self.camera_pos.y, self.camera_pos.z
                    ));
                    
                    ui.label(format!("Biome: {}", self.biome_name));
                    
                    ui.separator();
                    
                    ui.colored_label(egui::Color32::LIGHT_GRAY, "[F1] Exit Editor");
                });
            
            // === Asset Inspector (Top-Right) ===
            egui::Window::new("Asset Inspector")
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 10.0))
                .resizable(false)
                .collapsible(false)
                .title_bar(true)
                .default_width(200.0)
                .show(ctx, |ui| {
                    if let Some(ref mut model) = self.debug_model {
                        ui.label(format!("Model: {}", model.file_name));
                        ui.separator();
                        
                        // Scale slider
                        ui.horizontal(|ui| {
                            ui.label("Scale:");
                            ui.add(egui::Slider::new(&mut model.params.scale, 0.1..=10.0));
                        });
                        
                        // Y-Offset slider
                        ui.horizontal(|ui| {
                            ui.label("Y-Offset:");
                            ui.add(egui::Slider::new(&mut model.params.y_offset, -5.0..=5.0));
                        });
                        
                        // Rotation slider
                        ui.horizontal(|ui| {
                            ui.label("Rotation Y:");
                            ui.add(egui::Slider::new(&mut model.params.rotation_y, 0.0..=360.0).suffix("°"));
                        });
                        
                        ui.separator();
                        
                        // Log to Console button - outputs transform for easy copying
                        if ui.button("📋 Log to Console").clicked() {
                            println!("\n=== Debug Model Transform ===");
                            println!("Model: {}", model.file_name);
                            println!("Position: ({:.2}, {:.2}, {:.2})", 
                                model.spawn_position.x, 
                                model.spawn_position.y + model.params.y_offset, 
                                model.spawn_position.z);
                            println!("Scale: {:.2}", model.params.scale);
                            println!("Y-Offset: {:.2}", model.params.y_offset);
                            println!("Rotation Y: {:.1}°", model.params.rotation_y);
                            println!("==============================\n");
                        }
                        
                        ui.horizontal(|ui| {
                            if ui.button("Reset").clicked() {
                                model.params = DebugModelParams::default();
                            }
                            if ui.button("Clear Model").clicked() {
                                // Will be cleared after this
                            }
                        });
                    } else {
                        ui.vertical_centered(|ui| {
                            ui.add_space(20.0);
                            ui.label(
                                egui::RichText::new("🗁 Drag & Drop")
                                    .size(18.0)
                                    .color(egui::Color32::LIGHT_GRAY)
                            );
                            ui.label(
                                egui::RichText::new(".glb file here")
                                    .size(14.0)
                                    .color(egui::Color32::GRAY)
                            );
                            ui.add_space(20.0);
                        });
                    }
                });
        })
    }
    
    /// Prepares egui render (tessellates).
    pub fn prepare_render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        window: &Window,
        output: &egui::FullOutput,
    ) -> Vec<egui::ClippedPrimitive> {
        // Handle platform output (clipboard, cursor, etc.)
        self.state.handle_platform_output(window, output.platform_output.clone());
        
        // Tessellate
        let primitives = self.ctx.tessellate(output.shapes.clone(), output.pixels_per_point);
        
        // Update textures
        for (id, delta) in &output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }
        
        // Upload buffers
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [
                window.inner_size().width,
                window.inner_size().height,
            ],
            pixels_per_point: window.scale_factor() as f32,
        };
        
        self.renderer.update_buffers(
            device,
            queue,
            encoder,
            &primitives,
            &screen_descriptor,
        );
        
        primitives
    }
    
    /// Renders egui to the given render pass.
    pub fn render(
        &self,
        render_pass: &mut wgpu::RenderPass<'static>,
        primitives: &[egui::ClippedPrimitive],
        screen_width: u32,
        screen_height: u32,
        scale_factor: f32,
    ) {
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [screen_width, screen_height],
            pixels_per_point: scale_factor,
        };
        
        self.renderer.render(render_pass, primitives, &screen_descriptor);
    }
    
    /// Frees textures that are no longer needed.
    pub fn cleanup_textures(&mut self, output: &egui::FullOutput) {
        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}
