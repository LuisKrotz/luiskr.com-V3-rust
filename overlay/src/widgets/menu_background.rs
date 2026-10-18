//! Menu-background widget — the fullscreen nav backdrop.
//!
//! WGSL port of `core/utils/canvas/loaders/menu-background/shaders.ts`:
//! domain-warped fbm contour field (double-warp, IQ's technique) drawn
//! as ~14 isolines with pixel-constant hairline width via `fwidth`,
//! vignette + reveal fade, dual-ink drift.
//!
//! Uniform layout (f32s):
//!   [0] time s      [1..2] res px    [3] reveal 0..1
//!   [4..6] color rgb (ink A)         [7] alpha
//!   [8..10] color2 rgb (ink B)       [11] pad
//!   [12..15] widget rect (x, y, w, h) CSS px
//!
//! SYNC CONTRACT: `UNIFORM_F32S` (16 — the largest block) mirrors
//! `GPU_WIDGET_UNIFORM_F32S`. COST WARNING: this is the most expensive
//! shader per pixel (5 fbm evaluations = 80 hash calls/px) — it should
//! only be live while the menu is open; the bridge's `reveal` gate
//! already zeroes its output when closed, but unregistering when the
//! menu is closed entirely saves the whole pass. Tuning levers
//! (octaves, levels, warp strength) are commented in the .wgsl.

/// f32 slots in the widget uniform block.
pub const UNIFORM_F32S: usize = 16;

/// WGSL — ported from MENU_BG_FS_BODY.
pub const WGSL: &str = include_str!("menu_background.wgsl");
