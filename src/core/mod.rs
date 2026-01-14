//! Core Engine Module
//!
//! Contains fundamental engine systems: renderer, time management.

pub mod renderer;
pub mod time;

pub use renderer::Renderer;
pub use time::Time;
