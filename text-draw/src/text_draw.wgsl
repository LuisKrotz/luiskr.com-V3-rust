// text_draw.wgsl — the per-character hand-drawn writing animation.
//
// Builds on the `char-draw` CSS keyframe contract (per-char delay +
// duration + translateY(0.12em) entrance) but renders each letter as
// though a pen writes it: the reveal edge does NOT sweep a straight
// line — it follows a per-pixel traversal field baked into the atlas'
// R channel (see draw-text/traversal.ts): a geodesic flood over the
// glyph's ink mask seeded at the pen-entry point, so the frontier hugs
// the strokes, forks at junctions, splits around rings, and dots the
// tittle last like a real hand. A tight dark bead rides the frontier
// (the nib's wet deposit) and the whole quad carries a sub-pixel hand
// tremor that settles to zero when the letter completes — the written
// strokes are as crisp as the DOM glyphs they replace.
//
// Timing is UNCHANGED from the CSS/JS contract: `delay` and `dur`
// still come from the sheet exactly as the keyframes prescribed —
// only the in-glyph reveal mechanism is new.
//
// Per char: 6 synthesized verts (two triangles, no vertex buffer) —
// the vertex shader reads the storage sheet by instance_index,
// computes eased progress, offsets the quad by (1−e)·0.12·em (the
// CSS rise) plus the tremor, and passes the unit-quad coordinate +
// progress + a per-char seed to the fragment stage, which samples the
// atlas through the stroke-direction mask.
//
// Char-sheet layout (12 f32s — lib.rs doc):
//   [0..4) dest_rect (x,y,w,h device px) · [4..8) uv_rect ·
//   [8] delay_ms · [9] duration_ms · [10] em_px · [11] pad
//
// Uniform layout (f32s=16):
//   [0,1] resolution · [2] time_ms (elapsed since trigger) ·
//   [3] char_count · [4..8) ink rgba · [8..16) pad
//
// Shared prelude: quad_corner (the 6-vert unit square) and
// hash_sin_1d (per-char seed) come from gpu-mini's helpers_source —
// this file owns only the text-draw-specific entrance ease, vertex
// stage, and nib-masked fragment stage.

/// translateY(0.12em) — the CSS keyframe's entrance rise, in ems of
/// the char's font size.
const ENTRANCE_RISE_EM = 0.12;

/// Peak tremor displacement in device px — sub-pixel life, not blur
/// (a multi-px tremor smears every stroke).
const TREMOR_PEAK_PX = 0.25;

/// Stroke-frontier overshoot — the tail of the stroke finishes before
/// progress hits 1, giving the visible "lift-off" dwell before the
/// pen hops to the next char.
const FRONTIER_OVERSHOOT = 1.04;

/// Frontier transition width as a fraction of the stroke span —
/// about one texel of anti-alias, so the written stroke is as crisp
/// as the rasterized glyph underneath.
const FRONTIER_SOFTNESS = 0.012;

/// Nib bead half-width as a fraction of the stroke span — a tight
/// ridge straddling the frontier.
const NIB_HALF_WIDTH = 0.008;

/// Coverage overfill under the nib bead — the pen's wet deposit
/// bulges the mask slightly at the writing front.
const NIB_OVERFILL = 0.25;

/// Ink darkening under the nib bead — wet ink reads darker where the
/// pen is pressing.
const NIB_DARKEN = 0.12;

struct Uniforms {
    resolution: vec2<f32>,
    time_ms: f32,
    char_count: f32,
    ink_rgba: vec4<f32>,
    _pad: vec4<f32>,
};

struct CharRecord {
    /// x,y,w,h device px.
    dest_rect: vec4<f32>,
    /// u0,v0,u1,v1 in the atlas.
    uv_rect: vec4<f32>,
    delay_ms: f32,
    duration_ms: f32,
    /// The char's font-size px — translateY(0.12em) unit.
    em_px: f32,
    pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> char_records: array<CharRecord>;
@group(0) @binding(2) var atlas_tex: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;

/// `entrance_ease` — evaluate cubic-bezier(0.22,1,0.36,1) at progress
/// `progress`.
///
/// The CSS timing function is a cubic bezier through P1=(0.22,1),
/// P2=(0.36,1) — parametrically:
///   x(t) = 3(1−t)²t·0.22 + 3(1−t)t²·0.36 + t³
///   y(t) = 3(1−t)²t·1    + 3(1−t)t²·1    + t³   = 3t − 3t² + t³·(…)
///        = 1 − (1−t)³
/// (P1y=P2y=1 collapses the y polynomial to 1−(1−t)³.)
///
/// CSS evaluates y at the t where x(t)=p — an inverse: solve x(t)=p
/// for t by Newton's method. 3 iterations converge to ~1e-5 (the
/// function is smooth and monotonic on [0,1]); 4 for safety margin.
fn entrance_ease(progress: f32) -> f32 {
    var t = progress;

    for (var i = 0; i < 4; i++) {
        let remain = 1.0 - t;
        let bez_x = 3.0 * remain * remain * t * 0.22 + 3.0 * remain * t * t * 0.36 + t * t * t;
        let bez_dx = 3.0 * remain * remain * 0.22 + 6.0 * remain * t * (0.36 - 0.22) + 3.0 * t * t * (1.0 - 0.36);

        if (abs(bez_dx) < 1e-6) { break; }

        t = clamp(t - (bez_x - progress) / bez_dx, 0.0, 1.0);
    }

    // y(t) with P1y=P2y=1 → 1−(1−t)³.
    let remain = 1.0 - t;

    return 1.0 - remain * remain * remain;
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    /// Unit-quad coordinate (0..1 across the char box) — the mask's
    /// working space.
    @location(1) quad_local: vec2<f32>,
    /// Linear stroke progress 0..1 — drives the reveal edge.
    @location(2) progress: f32,
    /// Whole-glyph alpha (the eased CSS entrance component).
    @location(3) alpha: f32,
    /// Per-char seed — flat so every fragment of the quad shares the
    /// same stroke direction/wobble (interpolating it would tear the
    /// mask).
    @location(4) @interpolate(flat) char_seed: f32,
};

/// `vs_main` — synthesize one vertex of one char's quad: read the
/// sheet record, ease the per-char progress, offset the quad down by
/// the eased rise + hand tremor, project px→NDC.
@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOut {
    var out: VertexOut;
    let corner = quad_corner(vertex_index);
    let record = char_records[instance_index];

    // Per-char eased progress — the CSS keyframe's timing curve
    // evaluated at (time − delay)/dur, clamped to the keyframe span.
    // `progress` stays linear: a pen moves at near-constant speed
    // through a letter, so the stroke mask follows `progress` while
    // the entrance (rise + faint alpha ramp) keeps the eased value.
    let progress = clamp(
        (uniforms.time_ms - record.delay_ms) / max(record.duration_ms, 1.0),
        0.0,
        1.0,
    );
    let eased = entrance_ease(progress);

    // translateY(0.12em) → down 12% of the char's font size at
    // progress 0, rising to 0 at progress 1.
    let rise_offset = (1.0 - eased) * ENTRANCE_RISE_EM * record.em_px;

    // Hand tremor — two incommensurate sines (~5Hz, ~4.3Hz at ms
    // resolution ≈ the physiological tremor band) fading to zero as
    // the letter completes: the nib wobbles while the stroke is live
    // and the finished letter sits dead still. Kept at
    // TREMOR_PEAK_PX — sub-pixel life, not blur.
    let char_seed = f32(instance_index);
    let tremor = (1.0 - eased) * TREMOR_PEAK_PX * vec2<f32>(
        sin(uniforms.time_ms * 0.031 + char_seed * 7.13),
        cos(uniforms.time_ms * 0.027 + char_seed * 3.71),
    );

    // Device px → NDC: x' = x/res·2−1, y' = 1−y/res·2 (y-down DOM →
    // y-up clip).
    let pixel_pos = record.dest_rect.xy + tremor + vec2<f32>(
        corner.x * record.dest_rect.z,
        corner.y * record.dest_rect.w + rise_offset,
    );
    let clip_xy = vec2<f32>(
        pixel_pos.x / uniforms.resolution.x * 2.0 - 1.0,
        1.0 - pixel_pos.y / uniforms.resolution.y * 2.0,
    );

    out.position = vec4<f32>(clip_xy, 0.0, 1.0);
    out.uv = record.uv_rect.xy + corner * (record.uv_rect.zw - record.uv_rect.xy);
    out.quad_local = corner;
    out.progress = progress;
    out.alpha = eased;
    out.char_seed = char_seed;

    return out;
}

/// `fs_main` — atlas texel · ink, masked by the baked traversal field.
///
/// texel.r holds the pen-visit order for that pixel (0 = stroke entry,
/// 1 = stroke end / unreachable / paper); texel.a is the glyph
/// coverage. The frontier advances linearly — a nib moves at
/// near-constant speed — and every pixel whose visit-order is behind
/// it is ink; ahead of it is paper. The transition window is
/// FRONTIER_SOFTNESS of the stroke span — about one texel of
/// anti-alias, so the written stroke is as crisp as the rasterized
/// glyph underneath. A tight Gaussian bead centred ON the frontier
/// overfills coverage slightly and darkens — the pen's wet deposit —
/// so the writing front reads as a nib, not a clipping plane.
@fragment
fn fs_main(frag: VertexOut) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas_tex, atlas_sampler, frag.uv);

    let frontier = frag.progress * FRONTIER_OVERSHOOT;
    let written = 1.0 - smoothstep(frontier - FRONTIER_SOFTNESS, frontier, texel.r);

    // Nib bead — a tight ridge straddling the frontier: coverage
    // overfills NIB_OVERFILL of the span around the edge, ink darkens
    // under pressure.
    let nib_distance = (texel.r - frontier) / NIB_HALF_WIDTH;
    let nib_bead = exp(-nib_distance * nib_distance);

    let coverage = min(written + nib_bead * NIB_OVERFILL, 1.0);
    let alpha = texel.a * uniforms.ink_rgba.a * frag.alpha * coverage;
    let ink_rgb = uniforms.ink_rgba.rgb * (1.0 - nib_bead * NIB_DARKEN);

    return vec4<f32>(ink_rgb * alpha, alpha);
}
