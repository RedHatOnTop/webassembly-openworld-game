//! Core Engine Module
//!
//! Contains fundamental engine systems: renderer, time management, asset loading.

pub mod renderer;
pub mod time;
pub mod assets;

pub use renderer::Renderer;
pub use time::Time;
