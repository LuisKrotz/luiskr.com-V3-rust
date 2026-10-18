//! Burger widget — the nav hamburger's three hairline bars.
//!
//! Faithful WGSL port of `core/utils/canvas/widgets/burger-button/shaders.ts`:
//! per-bar phase-offset waves, ±4% length breathing, ~9s ripple period —
//! deliberate slowness keeps hairline edges from crawling at 120Hz+.
//! Uniforms only; geometry is synthesized in-shader per pixel, so the
//! widget costs one scissored fullscreen draw — no buffers per frame.
//!
//! Uniform layout (f32s):
//!   [0] time s      [1] res.x px   [2] res.y px   [3] dark 0..1
//!   [4..7] widget rect (x, y, w, h) in canvas CSS px — scissor fallback.
//!
//! SYNC CONTRACT: `UNIFORM_F32S` must equal this kind's entry in
//! `GPU_WIDGET_UNIFORM_F32S` (core/tokens/motion/gpu-widgets.ts) — the
//! bridge allocates/writes that many f32 slots per frame. All visual
//! tunables live in burger.wgsl; its comments mark safe ranges (wave
//! amplitude, breathing depth, bar geometry — all UV-relative).

/// f32 slots in the widget uniform block.
pub const UNIFORM_F32S: usize = 8;

/// WGSL — ported line-for-line from BURGER_FS_BODY. `fwidth` is core
/// WGSL (no extension dance); everything else maps 1:1.
pub const WGSL: &str = include_str!("burger.wgsl");
