//! Skeleton widget — the loading shimmer's "data decoding" cells.
//!
//! WGSL port of `core/utils/canvas/loaders/skeleton/shaders.ts`: rounded
//! rect cells whose 0/1 glyphs morph at per-cell cadence, ragged-right
//! text rows, density drift, resolve collapse. Per-rect data (rect,
//! meta, surface ink, sink ink) lives in a **storage buffer** rather
//! than a uniform array — 24 rects × 4 vec4 fields is 1536 B, valid in
//! either class, but storage keeps the door open to larger sheets and
//! lets the bridge update rect data with one write_buffer.
//!
//! Uniform layout (f32s):
//!   [0..1] res px    [2] time s     [3] resolve 0..1
//!   [4] ink_alpha    [5] count      [6..7] pad
//!   [8..11] widget rect (x, y, w, h) CSS px
//!
//! Storage layout (vec4s, ×MAX_RECTS):
//!   rects  [x, y, w, h] device px (y down)
//!   meta   [radius, cell, row height (0 = media), unused]
//!   sbase  [r, g, b, unused] surface
//!   sink   [r, g, b, unused] ink
//!
//! SYNC CONTRACT: `UNIFORM_F32S` + `STORAGE_VEC4S` mirror the bridge
//! constants (`GPU_WIDGET_UNIFORM_F32S` + the 24×16 f32 sheet write
//! size). `count` must never exceed MAX_RECTS — the shader breaks at
//! `i >= count` but the sheet is only `MAX_RECTS*4` vec4s deep, so an
//! over-count reads uninitialized storage.

/// f32 slots in the widget uniform block.
pub const UNIFORM_F32S: usize = 12;

/// MAX_RECTS — mirrors SKELETON_RENDER.MAX_RECTS (24) in
/// `core/tokens/motion/skeleton.ts`; bump both together.
pub const MAX_RECTS: usize = 24;

/// vec4 fields per rect — rects, meta, sbase, sink.
pub const RECT_FIELDS: usize = 4;

/// Storage-buffer size in vec4s (MAX_RECTS × RECT_FIELDS).
pub const STORAGE_VEC4S: usize = MAX_RECTS * RECT_FIELDS;

/// WGSL — ported from SKELETON_FS.
pub const WGSL: &str = include_str!("skeleton.wgsl");
