# Task 02: Decoupled Game Loop

| Status | Priority | Depends On |
|--------|----------|------------|
| Pending | High | Task 01 (Window Initialization) |

## Goal

Implement a deterministic fixed timestep loop for logic updates and a variable timestep loop for rendering. This decouples game simulation from frame rate, ensuring consistent physics and gameplay regardless of display refresh rate.

Reference: [engine_loop.md](../02_ARCHITECTURE/engine_loop.md)

---

## Implementation Steps

### Step 1: Time Manager

Create `src/core/time.rs`.

**Requirements:**
- Implement a `Time` struct to track:
  - `delta_time`: Time elapsed since the last frame (in seconds).
  - `elapsed_time`: Total time since application start (in seconds).
- Provide `now()` functionality using platform-appropriate APIs:
  - **WASM**: Use `web_sys::window().performance().now()`.
  - **Native**: Use `std::time::Instant`.

**Struct Signature:**
```rust
pub struct Time {
    startup: Instant, // Platform-specific instant
    last_frame: Instant,
    delta_time: f32,
    elapsed_time: f64,
}

impl Time {
    pub fn new() -> Self;
    pub fn update(&mut self);
    pub fn delta_time(&self) -> f32;
    pub fn elapsed_time(&self) -> f64;
}
```

---

### Step 2: Game State

Create `src/game/state.rs`.

**Requirements:**
- Define a `GameState` struct to hold simulation data.
- For initial verification, include simple fields:
  - `tick_count: u64` - Number of fixed updates executed.
  - `time: f32` - Accumulated simulation time.

**Struct Signature:**
```rust
pub struct GameState {
    pub tick_count: u64,
    pub time: f32,
}

impl GameState {
    pub fn new() -> Self;
    pub fn update(&mut self, dt: f32);
}
```

The `update` method should:
- Increment `tick_count`.
- Accumulate `time += dt`.

---

### Step 3: The Loop (Integration in `main.rs`)

Modify the `App` struct to integrate the game loop.

**Requirements:**
- Add fields to `App`:
  - `game_state: GameState`
  - `time: Time`
  - `accumulator: f32`
- Define constant: `const FIXED_TIMESTEP: f32 = 1.0 / 60.0;`

**Accumulator Pattern Implementation:**
```rust
// In the main loop / frame callback:
self.time.update();
let delta_time = self.time.delta_time();

self.accumulator += delta_time;

while self.accumulator >= FIXED_TIMESTEP {
    self.game_state.update(FIXED_TIMESTEP);
    self.accumulator -= FIXED_TIMESTEP;
}

// Calculate interpolation alpha for rendering
let alpha = self.accumulator / FIXED_TIMESTEP;

// Pass state and alpha to renderer
self.renderer.render(&self.game_state, alpha);
```

**Spiral of Death Protection:**
- Cap the maximum accumulated time to prevent runaway updates:
```rust
const MAX_FRAME_TIME: f32 = 0.25; // 250ms cap
let delta_time = self.time.delta_time().min(MAX_FRAME_TIME);
```

---

### Step 4: Renderer Update

Update `src/core/renderer.rs`.

**Requirements:**
- Modify `Renderer::render()` signature to accept game state:
```rust
pub fn render(&mut self, state: &GameState, alpha: f32) -> Result<(), wgpu::SurfaceError>;
```
- For visual verification, implement a pulsing clear color based on `GameState.time`:
```rust
let pulse = (state.time.sin() * 0.5 + 0.5) as f64;
let clear_color = wgpu::Color {
    r: 0.1,
    g: 0.1 + pulse * 0.2,
    b: 0.2 + pulse * 0.3,
    a: 1.0,
};
```

---

## File Structure After Completion

```
src/
  core/
    mod.rs        # Add: pub mod time;
    time.rs       # NEW
    renderer.rs   # MODIFIED
  game/
    mod.rs        # NEW: pub mod state;
    state.rs      # NEW
  main.rs         # MODIFIED
```

---

## Verification

### Automated Checks
1. **Build**: `cargo build --target wasm32-unknown-unknown`
2. **Native Build**: `cargo build`
3. **No Warnings**: Ensure clean compilation.

### Runtime Verification
1. **Run the engine** (native or WASM).
2. **Console Output**: Add logging to verify:
   - Fixed updates occur at approximately 60 ticks/second.
   - Log `tick_count` periodically (e.g., every 60 ticks).
3. **Visual Proof**: Screen background color should pulse smoothly, demonstrating:
   - Game state updates correctly.
   - Renderer receives and uses game state.

### Expected Console Output (Example)
```
[INFO] Tick 60 | Elapsed: 1.00s
[INFO] Tick 120 | Elapsed: 2.00s
[INFO] Tick 180 | Elapsed: 3.00s
```

---

## Acceptance Criteria

- [ ] `Time` struct correctly tracks delta and elapsed time on both WASM and Native.
- [ ] `GameState` updates deterministically at fixed timestep.
- [ ] Accumulator pattern implemented with spiral-of-death protection.
- [ ] Renderer accepts `GameState` and `alpha` parameters.
- [ ] Visual verification: pulsing background color.
- [ ] Stable ~60 logic updates per second confirmed via logging.
