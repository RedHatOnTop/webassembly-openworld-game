//! Time Manager
//!
//! Platform-agnostic time tracking for the game loop.
//! - Native: Uses std::time::Instant
//! - WASM: Uses web_sys::window().performance().now()

/// Stores the current time as a platform-specific instant.
/// On Native, this is std::time::Instant.
/// On WASM, this is f64 milliseconds from performance.now().
#[cfg(not(target_arch = "wasm32"))]
type Instant = std::time::Instant;

#[cfg(target_arch = "wasm32")]
type Instant = f64;

/// Returns the current time as a platform-specific instant.
#[cfg(not(target_arch = "wasm32"))]
fn now() -> Instant {
    std::time::Instant::now()
}

#[cfg(target_arch = "wasm32")]
fn now() -> Instant {
    // Get performance.now() in milliseconds
    web_sys::window()
        .expect("no window")
        .performance()
        .expect("no performance")
        .now()
}

/// Calculates the duration between two instants in seconds.
#[cfg(not(target_arch = "wasm32"))]
fn duration_since(start: Instant, end: Instant) -> f64 {
    end.duration_since(start).as_secs_f64()
}

#[cfg(target_arch = "wasm32")]
fn duration_since(start: Instant, end: Instant) -> f64 {
    // Convert milliseconds to seconds
    (end - start) / 1000.0
}

/// Time manager for tracking frame timing and elapsed time.
///
/// Provides delta_time for the game loop and total elapsed time
/// since application start.
pub struct Time {
    /// Time when the application started.
    startup: Instant,
    /// Time of the last frame.
    last_frame: Instant,
    /// Time elapsed since the last frame (in seconds).
    delta_time: f32,
    /// Total time elapsed since startup (in seconds).
    elapsed_time: f64,
}

impl Time {
    /// Creates a new Time manager, initializing startup time to now.
    pub fn new() -> Self {
        let current = now();
        Self {
            startup: current,
            last_frame: current,
            delta_time: 0.0,
            elapsed_time: 0.0,
        }
    }

    /// Updates the time manager. Call this once per frame.
    ///
    /// Calculates delta_time since the last call and updates elapsed_time.
    pub fn update(&mut self) {
        let current = now();
        self.delta_time = duration_since(self.last_frame, current) as f32;
        self.elapsed_time = duration_since(self.startup, current);
        self.last_frame = current;
    }

    /// Returns the time elapsed since the last frame in seconds.
    #[inline]
    pub fn delta_time(&self) -> f32 {
        self.delta_time
    }

    /// Returns the total time elapsed since application start in seconds.
    #[inline]
    pub fn elapsed_time(&self) -> f64 {
        self.elapsed_time
    }
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}
