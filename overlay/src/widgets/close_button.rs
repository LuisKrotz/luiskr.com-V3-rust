//! Close-button widget — the modal's liquid-fill dismiss control.
//!
//! WGSL port of `core/utils/canvas/widgets/close-button/shaders.ts`:
//! turquoise liquid rising with a glowing foam crest + procedural
//! bubbles, rotating auto-drawn X strokes, click shockwave.
//!
//! Uniform layout (f32s):
//!   [0..1] res px      [2] time s    [3] liquid 0..1
//!   [4] draw 0..1      [5] rot rad   [6] click_time s   [7] pad
//!   [8..11] widget rect (x, y, w, h) CSS px
//!
//! SYNC CONTRACT: `UNIFORM_F32S` mirrors `GPU_WIDGET_UNIFORM_F32S`.
//! `click_time` uses -1.0 for "never clicked"; `rot` is raw radians
//! (JS advances it — the shader consumes, never accumulates, so frame
//! rate can't drift the spin). `liquid`/`draw` are eased on the JS
//! side — the shader treats them as instantaneous state. Tunables
//! (wave amplitude, bubble count×4, shockwave speed/lifetime) are
//! per-line commented in close_button.wgsl.

/// f32 slots in the widget uniform block.
pub const UNIFORM_F32S: usize = 12;

/// WGSL — ported from CLOSE_BUTTON_FS.
pub const WGSL: &str = include_str!("close_button.wgsl");
