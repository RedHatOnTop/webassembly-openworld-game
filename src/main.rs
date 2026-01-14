//! Aether Engine - Main Entry Point
//!
//! Implements a decoupled game loop with:
//! - Fixed timestep for deterministic game logic (60 Hz)
//! - Variable timestep for smooth rendering
//! - Accumulator pattern with spiral-of-death protection

mod core;
mod game;

use core::{Renderer, Time};
use game::GameState;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

/// Fixed timestep for game logic updates (60 Hz).
const FIXED_TIMESTEP: f32 = 1.0 / 60.0;

/// Maximum frame time to prevent spiral of death.
/// If a frame takes longer than this, accumulated time is capped.
const MAX_FRAME_TIME: f32 = 0.25;

/// Application state handler for winit 0.30.
///
/// Manages the game loop with decoupled update and render cycles.
struct App {
    /// Window handle.
    window: Option<Arc<Window>>,
    /// WebGPU renderer.
    renderer: Option<Renderer>,
    /// Game simulation state (updated at fixed timestep).
    game_state: GameState,
    /// Time manager for delta and elapsed time tracking.
    time: Time,
    /// Accumulator for fixed timestep logic.
    accumulator: f32,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            game_state: GameState::new(),
            time: Time::new(),
            accumulator: 0.0,
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

            self.window = Some(window);
            self.renderer = Some(renderer);

            // Reset time manager when window is created
            self.time = Time::new();
            log::info!(
                "Game loop initialized | Fixed timestep: {:.4}s ({} Hz)",
                FIXED_TIMESTEP,
                (1.0 / FIXED_TIMESTEP) as u32
            );
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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
            WindowEvent::RedrawRequested => {
                // Update time manager
                self.time.update();

                // Get delta time with spiral-of-death protection
                let delta_time = self.time.delta_time().min(MAX_FRAME_TIME);

                // Accumulate time for fixed updates
                self.accumulator += delta_time;

                // Fixed timestep updates (deterministic game logic)
                while self.accumulator >= FIXED_TIMESTEP {
                    self.game_state.update(FIXED_TIMESTEP);
                    self.accumulator -= FIXED_TIMESTEP;
                }

                // Calculate interpolation alpha for smooth rendering
                let alpha = self.accumulator / FIXED_TIMESTEP;

                // Render with current state and interpolation factor
                if let Some(renderer) = &mut self.renderer {
                    match renderer.render(&self.game_state, alpha) {
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
