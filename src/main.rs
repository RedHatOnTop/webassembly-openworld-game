//! Aether Engine - Main Entry Point
//!
//! Implements a decoupled game loop with:
//! - Fixed timestep for deterministic game logic (60 Hz)
//! - Variable timestep for smooth rendering
//! - Free camera (WASD + mouse look)
//! - Debug GUI overlay (egui)

mod core;
mod game;

use core::{Renderer, Time, GuiSystem};
use game::GameState;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent, ElementState},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId, CursorGrabMode},
    keyboard::{KeyCode, PhysicalKey},
};

/// Fixed timestep for game logic updates (60 Hz).
const FIXED_TIMESTEP: f32 = 1.0 / 60.0;

/// Maximum frame time to prevent spiral of death.
const MAX_FRAME_TIME: f32 = 0.25;

/// Application state handler for winit 0.30.
struct App {
    /// Window handle.
    window: Option<Arc<Window>>,
    /// WebGPU renderer.
    renderer: Option<Renderer>,
    /// GUI system (egui).
    gui_system: Option<GuiSystem>,
    /// Game simulation state.
    game_state: GameState,
    /// Time manager.
    time: Time,
    /// Accumulator for fixed timestep logic.
    accumulator: f32,
    /// FPS tracking.
    fps_counter: FpsCounter,
}

/// Simple FPS counter.
struct FpsCounter {
    frame_count: u32,
    elapsed: f32,
    current_fps: f32,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            frame_count: 0,
            elapsed: 0.0,
            current_fps: 0.0,
        }
    }
    
    fn update(&mut self, delta: f32) -> f32 {
        self.frame_count += 1;
        self.elapsed += delta;
        
        if self.elapsed >= 1.0 {
            self.current_fps = self.frame_count as f32 / self.elapsed;
            self.frame_count = 0;
            self.elapsed = 0.0;
        }
        
        self.current_fps
    }
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            gui_system: None,
            game_state: GameState::new(),
            time: Time::new(),
            accumulator: 0.0,
            fps_counter: FpsCounter::new(),
        }
    }
    
    /// Sets cursor grab mode based on editor mode.
    fn update_cursor_mode(&self) {
        if let (Some(window), Some(gui)) = (&self.window, &self.gui_system) {
            if gui.editor_mode {
                // Editor mode: free cursor
                let _ = window.set_cursor_grab(CursorGrabMode::None);
                window.set_cursor_visible(true);
            } else {
                // Game mode: locked cursor
                let _ = window.set_cursor_grab(CursorGrabMode::Confined)
                    .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked));
                window.set_cursor_visible(false);
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attrs = Window::default_attributes()
                .with_title("Aether Engine")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
                .with_min_inner_size(winit::dpi::LogicalSize::new(320, 240));

            let window = Arc::new(
                event_loop
                    .create_window(window_attrs)
                    .expect("Failed to create window"),
            );

            log::info!("Window created: 1280x720");

            let renderer = pollster::block_on(Renderer::new(Arc::clone(&window)));
            log::info!("Renderer initialized successfully");
            
            // Initialize GUI system
            let gui_system = GuiSystem::new(
                renderer.device(),
                renderer.surface_format(),
                &window,
            );
            log::info!("GUI system initialized");

            self.window = Some(window);
            self.renderer = Some(renderer);
            self.gui_system = Some(gui_system);

            self.time = Time::new();
            log::info!(
                "Game loop initialized | Fixed timestep: {:.4}s ({} Hz)",
                FIXED_TIMESTEP,
                (1.0 / FIXED_TIMESTEP) as u32
            );
            log::info!("Controls: WASD=Move, Space/Shift=Up/Down, Right-Click+Drag=Look, F1=Editor Mode");
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Forward event to GUI first
        if let (Some(window), Some(gui)) = (&self.window, &mut self.gui_system) {
            if gui.on_window_event(window, &event) && gui.editor_mode {
                // GUI consumed the event in editor mode
                return;
            }
        }
        
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested, shutting down...");
                log::info!(
                    "Final stats: {} ticks, {:.2}s elapsed",
                    self.game_state.tick_count,
                    self.game_state.time
                );
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                log::debug!(
                    "Window resized to {}x{}",
                    physical_size.width,
                    physical_size.height
                );
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(physical_size.width, physical_size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // Handle F1 for editor mode toggle
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(KeyCode::F1) = event.physical_key {
                        if let Some(gui) = &mut self.gui_system {
                            gui.toggle_editor_mode();
                            self.update_cursor_mode();
                            return;
                        }
                    }
                }
                
                // Forward to renderer if not in editor mode
                if let (Some(renderer), Some(gui)) = (&mut self.renderer, &self.gui_system) {
                    if !gui.editor_mode {
                        renderer.handle_key(event.physical_key, event.state.is_pressed());
                    }
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                // Only forward to renderer if not in editor mode
                if let (Some(renderer), Some(gui)) = (&mut self.renderer, &self.gui_system) {
                    if !gui.editor_mode {
                        renderer.handle_mouse_button(button, state == ElementState::Pressed);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                // Update time manager
                self.time.update();

                // Get delta time with spiral-of-death protection
                let delta_time = self.time.delta_time().min(MAX_FRAME_TIME);
                
                // Update FPS counter
                let fps = self.fps_counter.update(delta_time);

                // Update camera movement (only if not in editor mode)
                if let (Some(renderer), Some(gui)) = (&mut self.renderer, &self.gui_system) {
                    if !gui.editor_mode {
                        renderer.update_movement(delta_time);
                    }
                    
                    // Update GUI state
                    if let Some(gui) = &mut self.gui_system {
                        gui.fps = fps;
                        gui.frame_time_ms = delta_time * 1000.0;
                        gui.camera_pos = renderer.camera.position;
                        gui.biome_name = "Forest".to_string(); // TODO: Calculate from noise
                        
                        // Process any pending file drops
                        gui.process_pending_files(
                            renderer.device(),
                            renderer.camera.position,
                            renderer.camera.forward(),
                        );
                    }
                }

                // Accumulate time for fixed updates
                self.accumulator += delta_time;

                // Fixed timestep updates
                while self.accumulator >= FIXED_TIMESTEP {
                    self.game_state.update(FIXED_TIMESTEP);
                    self.accumulator -= FIXED_TIMESTEP;
                }

                // Calculate interpolation alpha
                let alpha = self.accumulator / FIXED_TIMESTEP;

                // Render
                if let (Some(renderer), Some(window), Some(gui)) = 
                    (&mut self.renderer, &self.window, &mut self.gui_system) 
                {
                    match renderer.render_with_gui(&self.game_state, alpha, gui, window) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("GPU out of memory!");
                            event_loop.exit();
                        }
                        Err(wgpu::SurfaceError::Timeout) => {
                            log::warn!("Surface timeout, skipping frame");
                        }
                        Err(e) => {
                            log::warn!("Render error: {:?}", e);
                        }
                    }
                }

                // Request next frame
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: winit::event::DeviceId, event: DeviceEvent) {
        // Handle raw mouse motion for camera look (only in game mode)
        if let DeviceEvent::MouseMotion { delta } = event {
            if let (Some(renderer), Some(gui)) = (&mut self.renderer, &self.gui_system) {
                if !gui.editor_mode {
                    renderer.handle_mouse_motion(delta);
                }
            }
        }
    }
}

fn main() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    log::info!("Aether Engine starting...");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}
