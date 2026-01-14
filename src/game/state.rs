//! Game State
//!
//! Holds the simulation state that is updated at a fixed timestep.
//! This ensures deterministic game logic regardless of frame rate.

/// The core game state that gets updated at a fixed timestep.
///
/// Contains all simulation data that must be updated deterministically.
/// For visual interpolation, the renderer receives both the current state
/// and an alpha value representing progress toward the next fixed update.
pub struct GameState {
    /// Number of fixed-timestep updates that have occurred.
    pub tick_count: u64,
    /// Accumulated simulation time in seconds.
    /// Used for time-based effects and visual verification.
    pub time: f32,
}

impl GameState {
    /// Creates a new GameState with initial values.
    pub fn new() -> Self {
        Self {
            tick_count: 0,
            time: 0.0,
        }
    }

    /// Updates the game state for one fixed timestep.
    ///
    /// # Arguments
    /// * `dt` - The fixed timestep duration in seconds (typically 1/60).
    pub fn update(&mut self, dt: f32) {
        self.tick_count += 1;
        self.time += dt;

        // Log every 60 ticks (approximately once per second)
        if self.tick_count % 60 == 0 {
            log::info!(
                "Tick {} | Elapsed: {:.2}s",
                self.tick_count,
                self.time
            );
        }
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}
