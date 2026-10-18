//! Switch-slider widget — preference toggles (stats / grid / motion).
//!
//! WGSL port of `core/utils/canvas/widgets/switch-slider/shaders.ts`:
//! capsule track, per-context animated waveform (ECG pulse / blueprint
//! grid / motion streaks), 3D knob with shadow + accent dot.
//!
//! Uniform layout (f32s):
//!   [0..1] res px      [2] time s    [3] progress 0..1
//!   [4] knob_x px      [5] context (0 stats, 1 grid, 2 motion)
//!   [6..7] pad
//!   [8..11] widget rect (x, y, w, h) CSS px
//!
//! SYNC CONTRACT: `UNIFORM_F32S` mirrors `GPU_WIDGET_UNIFORM_F32S`
//! (core/tokens/motion/gpu-widgets.ts). CAUTION — all widget math is
//! RAW PIXELS against `res`, not UV: changing the design size (14px
//! track height, 11px knob radius in the shader) requires editing the
//! shader's px constants, not just res. Safe tunables (track colors,
//! wave speeds, pattern density) are commented in switch_slider.wgsl.

/// f32 slots in the widget uniform block.
pub const UNIFORM_F32S: usize = 12;

/// WGSL — ported from SWITCH_SLIDER_FS.
pub const WGSL: &str = include_str!("switch_slider.wgsl");
