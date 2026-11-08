// palette.wgsl — the shared widget ink/glow palette, one audited copy.
//
// Every themed widget lerps between a "light theme" ink (dark slate,
// readable on pale surfaces) and a "dark theme" ink (ice white) on the
// `theme_dark` uniform — previously each shader pasted its own copy of
// these literals. The values are the site's canvas-ink tokens: they
// match the GLSL side (core/utils/canvas/glsl/palette.ts) so the WebGL
// fallback and the WebGPU widget paint identical pixels.
//
// Channels are linear 0..1 sRGB — the same space CSS colors land in.

/// Widget ink on a dark theme — near-white with a cool cast.
const INK_ON_DARK = vec3<f32>(0.95, 0.97, 1.0);

/// Widget ink on a light theme — dark slate/charcoal; never pure black
/// so the AA edge keeps a soft gradient to work with.
const INK_ON_LIGHT = vec3<f32>(0.10, 0.10, 0.14);

/// Halo/glow on a dark theme — the ink's hue family desaturated; a
/// saturated halo produces a chromatic fringe that reads as broken AA.
const GLOW_ON_DARK = vec3<f32>(0.55, 0.58, 0.65);

/// Halo/glow on a light theme — same desaturated-ink rule, dark side.
const GLOW_ON_LIGHT = vec3<f32>(0.15, 0.15, 0.22);

/// The close disc's painted slate — the X button's backing circle,
/// shared by the burger→close morph so the disc never pops color.
const DISC_SLATE = vec3<f32>(0.22, 0.23, 0.27);
