// ring_sheen.wgsl — instanced tile border rings with a drifting sheen.
//
// Ports the masked-gradient ring on `.home-mosaic-item` /
// `.related-item` (home-mosaic/_item.scss, internals/_related.scss):
//
//   ring    — hairline band at each tile's rounded-rect edge, drawn
//             as an SDF: |sd_round_box(to_center)| < hairline.
//   sheen   — a narrow brightness band along the tile's 120°
//             diagonal (the CSS gradient stops: transparent 32% →
//             edge 45% → accent 50% → edge 55% → transparent 68%),
//             drifting as a 300%-sized background-position sweep
//             condensed to a single band that slides along the
//             diagonal over the 9s loop.
//   breath  — the ::before cross-fade 0.15↔0.9, ease-in-out
//             alternate over 9s — one alpha term since both masked
//             layers draw the same ring.
//
// This file keeps its OWN instanced-quad vertex stage, so it composes
// with the gpu-mini helpers prelude (sd_round_box, quad_corner,
// ease_smooth_step) — never the shared fullscreen triangle.
//
// Tile-sheet layout (8 f32s — lib.rs doc):
//   [0..4) rect_px (x,y,w,h device px) · [4] corner_radius ·
//   [5] phase_offset · [6,7] reserved
//
// Uniform layout (f32s=16):
//   [0,1] resolution_dev · [2] time_seconds · [3] hairline_half ·
//   [4..8) edge_rgba · [8..12) accent_rgba · [12..16) reserved

struct Uniforms {
    resolution_dev: vec2<f32>,
    time_seconds: f32,
    hairline_half: f32,
    edge_rgba: vec4<f32>,
    accent_rgba: vec4<f32>,
    reserved: vec4<f32>,
};

struct TileRecord {
    rect_px: vec4<f32>,
    corner_radius: f32,
    phase_offset: f32,
    reserved: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> tiles: array<TileRecord>;

// ── Tunables ─────────────────────────────────────────────────────────

// Quad slack — each tile's quad extends one hairline+1px past its rect
// so the ring's outer edge isn't clipped by its own quad bounds.
const SLACK_PAD_PX: f32 = 1.0;

// The ring mask AA — |sdf| < hairline gives the band; smoothstep over
// ±1px so the hairline doesn't shimmer.
const RING_AA_PX: f32 = 1.0;
const LIVE_EPSILON: f32 = 0.001;

// Diagonal sheen — the 120° gradient direction (CSS deg: 0 = upward,
// clockwise — 120° → direction (sin120, −cos120) = (0.866, 0.5) in
// y-down screen space).
const SHEEN_DIR = vec2<f32>(0.8660254, 0.5);
// The band slides along the diagonal over the 9s-alternate loop —
// alternate means 9s out + 9s back = an 18s round trip, so the
// triangle wave's period is 18s, not 9s.
const SWEEP_PERIOD_S: f32 = 18.0;
// Band center travel: starts/ends 0.3 OFF the tile's diagonal on each
// side so the ring fully clears between sweeps.
const TRAVEL_LO: f32 = -0.3;
const TRAVEL_HI: f32 = 1.3;

// Band shape — the CSS stops (transparent 32% → edge 45% → accent 50%
// → edge 55% → transparent 68%) scale to a band of half-width 0.18
// around the band center: full accent inside ±0.05, tint falls to 0
// by ±0.18.
const CORE_HALF_WIDTH: f32 = 0.05;
const TINT_OUTER_WIDTH: f32 = 0.18;

// Breath — ::before cross-fade 0.15↔0.9 ease-in-out alternate on the
// same 18s round trip (the ping phase already carries the alternate
// timing; the shared smoothstep curve ≈ ease-in-out).
const BREATH_LO: f32 = 0.15;
const BREATH_RANGE: f32 = 0.75;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) pixel_px: vec2<f32>,   // device-px position
    @location(1) instance_t: f32,       // tile index (flat — fs reads the same record)
};

/// `vs_main` — place the tile's quad: each instance covers the tile
/// rect (plus one hairline of slack so the ring's outer edge isn't
/// clipped by its own quad bounds); geometry synthesized from the
/// shared quad table, no vertex buffers.
@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOut {
    var output: VertexOut;
    let corner_uv = quad_corner(vertex_index);
    let tile = tiles[instance_index];
    let slack = uniforms.hairline_half + SLACK_PAD_PX;

    let pixel_px = tile.rect_px.xy - slack + corner_uv * (tile.rect_px.zw + slack * 2.0);
    let clip_xy = vec2<f32>(
        pixel_px.x / uniforms.resolution_dev.x * 2.0 - 1.0,
        1.0 - pixel_px.y / uniforms.resolution_dev.y * 2.0,
    );

    output.position = vec4<f32>(clip_xy, 0.0, 1.0);
    output.pixel_px = pixel_px;
    output.instance_t = f32(instance_index);

    return output;
}

/// `fs_main` — ring mask × diagonal sheen × breath.
@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    let tile = tiles[u32(input.instance_t + 0.5)];
    let center_px = tile.rect_px.xy + tile.rect_px.zw * 0.5;
    let half_extents = tile.rect_px.zw * 0.5;

    // Ring mask — 1 inside the hairline band, antialiased at the SDF
    // edges over ~1px so the hairline doesn't shimmer.
    let border_dist = sd_round_box(
        input.pixel_px - center_px,
        half_extents,
        tile.corner_radius,
    );
    let ring_mask = 1.0 - smoothstep(
        uniforms.hairline_half - RING_AA_PX,
        uniforms.hairline_half + RING_AA_PX,
        abs(border_dist),
    );

    if (ring_mask <= LIVE_EPSILON) {
        discard;
    }

    // Diagonal sheen — project the pixel onto SHEEN_DIR and normalize
    // by the tile's diagonal extent → band_t ∈ [0,1].
    let diagonal_span = dot(tile.rect_px.zw, abs(SHEEN_DIR));
    let band_t = dot(input.pixel_px - tile.rect_px.xy, SHEEN_DIR) / max(diagonal_span, 1.0);

    // Band center position — triangle wave (0→1→0 over the 18s round
    // trip) mapped onto the off→through→off travel range.
    let sweep_phase = fract((uniforms.time_seconds / SWEEP_PERIOD_S) + tile.phase_offset);
    let ping = 1.0 - abs(sweep_phase * 2.0 - 1.0);
    let band_center = mix(TRAVEL_LO, TRAVEL_HI, ping);

    // Band masks — full accent inside the core, edge tint out to the
    // band's outer width.
    let band_dist = abs(band_t - band_center);
    let core_mask = 1.0 - smoothstep(0.0, CORE_HALF_WIDTH, band_dist);
    let tint_mask = 1.0 - smoothstep(CORE_HALF_WIDTH, TINT_OUTER_WIDTH, band_dist);

    // Breath — the cross-fade alpha on the same alternate loop.
    let breath = BREATH_LO + BREATH_RANGE * ease_smooth_step(ping);

    let sheen_rgb = mix(uniforms.edge_rgba.rgb, uniforms.accent_rgba.rgb, core_mask);
    let sheen_alpha = mix(uniforms.edge_rgba.a, uniforms.accent_rgba.a, core_mask) * tint_mask;
    let alpha = ring_mask * sheen_alpha * breath;

    return vec4<f32>(sheen_rgb * alpha, alpha);
}
