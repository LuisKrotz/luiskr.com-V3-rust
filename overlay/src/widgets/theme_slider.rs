//! Theme-slider widget — the day/system/night slider scene.
//!
//! WGSL port of `core/utils/canvas/widgets/theme-slider/shaders.ts` —
//! the largest procedural scene in the set: gradient sky, twinkling
//! stars, corona-ringed sun, aurora band, crescent moon, two parallax
//! mesa layers with heat shimmer, interactive ripple, moon/sun knob.
//!
//! Uniform layout (f32s):
//!   [0..1] res px      [2] time s    [3] progress 0..2 (night→sys→day)
//!   [4] knob_x px      [5] ripple_time s   [6] ripple_pos px   [7] pad
//!   [8..11] widget rect (x, y, w, h) CSS px
//!
//! SYNC CONTRACT: `UNIFORM_F32S` mirrors `GPU_WIDGET_UNIFORM_F32S`.
//! `progress` is 0..2 (THREE stops — night/system/day), unlike every
//! other widget's 0..1 — pass the eased slider position directly.
//! `ripple_time` uses -1.0 for "never tapped" (elapsed<0 gate keeps
//! the ring inert). Tunables: scene layer alphas + positions are
//! per-layer commented in theme_slider.wgsl — each layer is
//! self-contained and safe to re-tune without touching the rest.

/// f32 slots in the widget uniform block.
pub const UNIFORM_F32S: usize = 12;

/// WGSL — ported from THEME_SLIDER_FS.
pub const WGSL: &str = include_str!("theme_slider.wgsl");
