//! Core Engine Module
//!
//! Contains fundamental engine systems: renderer, time management, asset loading, GUI.

pub mod renderer;
pub mod time;
pub mod assets;
pub mod gui;

pub use renderer::Renderer;
pub use time::Time;
pub use gui::GuiSystem;
