//! wgsl — the shared shader library linked into every widget module.
//!
//! # Why this exists
//!
//! Every widget crate used to paste its own copy of the SDF
//! primitives, hash helpers, easing curves, quad-corner table, and the
//! fullscreen-triangle vertex stage — under THREE different naming
//! schemes (`sdCapsule`/`distCapsule`/`distSegment` were the same
//! function; `hash`/`hash21`/`hash1`/`hp`/`h` five different hashes).
//! One audited copy now lives here; each crate's `.wgsl` file holds
//! only what is genuinely unique to that widget.
//!
//! # How a widget consumes it
//!
//! ```rust,ignore
//! // Scaffold widgets (uniform block + fullscreen triangle): the
//! // prelude is injected by init_widget — the crate's WGSL carries
//! // no helper fns and no vs_main at all.
//! gpu_mini::init_widget(&canvas, &device, WGSL)?;
//!
//! // Custom-pipeline widgets concat the chunks they need:
//! let source = gpu_mini::wgsl::fullscreen_source(WGSL);
//! gpu_mini::fullscreen_pipeline(&device, format, &source, &layouts)?;
//!
//! // Mesh/instanced widgets (own vs_main) take helpers only:
//! let source = gpu_mini::wgsl::helpers_source(WGSL);
//! device.create_shader_module(&GpuShaderModuleDescriptor::new(&source));
//! ```
//!
//! # What is deliberately NOT shared
//!
//! Per-widget hashes with different constants (a new seed re-rolls the
//! noise field — same quality, different arrangement, which is a pixel
//! change not a dedup), widget-specific glyphs, and each widget's
//! `Uniforms` struct (its field layout IS the JS↔wasm sync contract).

/// The shared widget ink/glow palette: `INK_ON_DARK`, `INK_ON_LIGHT`,
/// `GLOW_ON_DARK`, `GLOW_ON_LIGHT`, `DISC_SLATE`. Values match the GLSL
/// fallback library so WebGL and WebGPU widgets paint identical inks.
pub const PALETTE: &str = include_str!("wgsl/palette.wgsl");

/// Signed-distance primitives: `dist_to_segment`, `sd_capsule`,
/// `sd_round_box`, `smooth_min` — literate param names, one body each.
pub const SDF: &str = include_str!("wgsl/sdf.wgsl");

/// Deterministic sine-hash + value noise: `hash_sin_2d`, `hash_sin_1d`,
/// `value_noise` — the classic 127.1/311.7 lattice constants.
pub const HASH: &str = include_str!("wgsl/hash.wgsl");

/// Shared easing curves: `ease_smooth_step`, `ease_in_out_cubic`.
pub const EASING: &str = include_str!("wgsl/easing.wgsl");

/// `quad_corner` — the 6-vertex unit-square table for vertex-less
/// instanced quads (ring fills, per-glyph cells).
pub const QUAD: &str = include_str!("wgsl/quad.wgsl");

/// The fullscreen-triangle `vs_main` — the single vertex stage every
/// SDF widget used to redefine verbatim.
pub const FULLSCREEN_VS: &str = include_str!("wgsl/fullscreen.wgsl");

/// Every shared chunk except the vertex stage — the "helpers" prelude.
/// `concat!` expands each `include_str!` at compile time, so this is a
/// single &'static str — zero runtime cost, zero allocation.
const HELPERS: &str = concat!(
    include_str!("wgsl/palette.wgsl"),
    include_str!("wgsl/sdf.wgsl"),
    include_str!("wgsl/hash.wgsl"),
    include_str!("wgsl/easing.wgsl"),
    include_str!("wgsl/quad.wgsl"),
);

/// `helpers_source` — prepend the shared helper library to a widget
/// source that keeps its OWN vertex stage (mesh, instanced-quad, and
/// 6-vert-quad widgets). One small String alloc at init; never per
/// frame (the composed source is used once for shader compilation).
///
/// @param widget_wgsl The widget's own WGSL (its Uniforms struct,
///   vertex stage, and fragment body — no shared helper redefinitions).
/// @returns The complete module source ready for create_shader_module.
pub fn helpers_source(widget_wgsl: &str) -> String {
    [HELPERS, widget_wgsl].concat()
}

/// `fullscreen_source` — helpers + the shared fullscreen-triangle
/// `vs_main` + the widget's fragment-only source. This is the shape
/// `init_widget` builds internally; custom-layout widgets on the
/// triangle path call it themselves and hand the result to
/// `fullscreen_pipeline`.
///
/// @param widget_wgsl The widget's fragment-side source — its Uniforms
///   struct, bindings, and `fs_main` (no `vs_main`, no shared helpers).
/// @returns The complete module source ready for create_shader_module.
pub fn fullscreen_source(widget_wgsl: &str) -> String {
    [HELPERS, FULLSCREEN_VS, widget_wgsl].concat()
}
