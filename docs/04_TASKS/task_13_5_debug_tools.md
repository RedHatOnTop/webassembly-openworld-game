# Task 13.5: Developer Debug Tools

## Goal
Implement an **Immediate Mode GUI (egui)** overlay for real-time asset testing and performance monitoring.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        App (main.rs)                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │  Renderer   │◄───│  GuiSystem  │◄───│  DroppedFile    │  │
│  │             │    │             │    │  Events         │  │
│  │ ┌─────────┐ │    │ ┌─────────┐ │    └─────────────────┘  │
│  │ │3D Scene │ │    │ │egui Ctx │ │                         │
│  │ └─────────┘ │    │ └─────────┘ │                         │
│  │ ┌─────────┐ │    │ ┌─────────┐ │                         │
│  │ │Debug Obj│◄├────│ │Sliders  │ │                         │
│  │ └─────────┘ │    │ └─────────┘ │                         │
│  └─────────────┘    └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Steps

### Step 1: Dependencies
- Add `egui = "0.29"`, `egui-wgpu = "0.29"`, `egui-winit = "0.29"` to Cargo.toml
- Note: Using 0.29 for winit 0.30 compatibility

### Step 2: GuiSystem (`src/core/gui.rs`)
- Initialize `egui::Context`, `egui_winit::State`, `egui_wgpu::Renderer`
- Store debug state: FPS, camera pos, loaded debug model
- UI Panels:
  - **Stats Panel (Top-Left)**: FPS, coordinates, biome
  - **Asset Inspector (Top-Right)**: Sliders for scale/rotation/offset

### Step 3: Main.rs Integration
- Forward `WindowEvent` to `gui_system.on_window_event()`
- Handle `WindowEvent::DroppedFile` to load test model
- Toggle cursor lock with `F1` for Game/Editor mode
- Render egui after 3D scene

### Step 4: Debug Model Rendering
- Store temporary `DebugInstance` in GuiSystem
- Apply UI slider transforms to instance matrix
- Render using existing structure pipeline

## UI Layout

```
┌────────────────────────────────────────────────────────────┐
│ ┌─────────────────┐                    ┌─────────────────┐ │
│ │ FPS: 60         │                    │ Asset Inspector │ │
│ │ Pos: 0, 40, 0   │                    │ ─────────────── │ │
│ │ Biome: Forest   │                    │ Scale: [===]1.0 │ │
│ └─────────────────┘                    │ Y-Off: [===]0.0 │ │
│                                        │ Rot Y: [===]0°  │ │
│                                        │ [Reset] [Apply] │ │
│                                        └─────────────────┘ │
│                      [3D VIEW]                             │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Key Bindings
- `F1`: Toggle Editor Mode (show cursor, enable GUI)
- Drag & Drop: Load GLTF file for testing

## Verification
- [ ] FPS counter displays correct value
- [ ] Camera coordinates update in real-time
- [ ] Drag-drop GLTF spawns model in view
- [ ] Sliders control transform in real-time
- [ ] Invalid GLTF doesn't crash engine
