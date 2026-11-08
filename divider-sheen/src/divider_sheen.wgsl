// divider_sheen.wgsl — footer hairline + traveling accent band.
//
// Replaces the CSS `::after` translateX sheen on
// `.footer-source-divider` / `.internal-footer-items-divider`:
//
//   hairline — horizontal edge-faded line (border color, 12%/88%
//              transparent ends, per the SCSS `background:
//              linear-gradient(90deg, transparent, border 12%,
//              border 88%, transparent)`).
//   band     — 25%-width accent band sweeping left→right over 7s
//              ease-in-out (the keyframe `translateX(-110% → 440%)`
//              with `ease-in-out` timing = a smoothstep phase).
//   sheen    — the band's own gradient (transparent → accent →
//              transparent across its 25% width — the SCSS band
//              gradient is `90deg, transparent, accent 50%, transparent`).
//
// Fragment-side chunk only — ease_smooth_step + vs_main come from
// gpu-mini's wgsl library.
//
// Uniform layout (f32s=16 — lib.rs doc):
//   [0,1] resolution_dev · [2] time_seconds · [3] reserved ·
//   [4..8) border_rgba · [8..12) accent_rgba ·
//   [12..16) clip_rect (0,0,w,h device px)

struct Uniforms {
    resolution_dev: vec2<f32>,
    time_seconds: f32,
    reserved: f32,
    border_rgba: vec4<f32>,
    accent_rgba: vec4<f32>,
    clip_rect: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// ── Tunables ─────────────────────────────────────────────────────────

// Hairline edge fade — ramps in 0..FADE_IN_HI, full to FADE_OUT_LO,
// out to 1 (the SCSS 12%/88% stops).
const FADE_IN_HI: f32 = 0.12;
const FADE_OUT_LO: f32 = 0.88;

// Traveling band — the CSS: width 25%, translateX(-110% → 440%) over
// 7s ease-in-out. In uv terms the band's LEFT edge travels
// TRAVEL_LO → TRAVEL_HI; the center is left + half the band width.
const SWEEP_PERIOD_S: f32 = 7.0;
const TRAVEL_LO: f32 = -0.275;
const TRAVEL_HI: f32 = 1.1;
const BAND_HALF_WIDTH: f32 = 0.125;

/// `fs_main` — one fragment of the divider quad.
@fragment
fn fs_main(@builtin(position) pixel_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // frag→uv in the widget rect — device px to 0..1.
    let widget_uv = (pixel_coord.xy - uniforms.clip_rect.xy) / uniforms.clip_rect.zw;

    // ── Hairline base ──────────────────────────────────────────────
    // Edge fade: linear-gradient(90deg, transparent, border 12%,
    // border 88%, transparent).
    let fade_in = smoothstep(0.0, FADE_IN_HI, widget_uv.x);
    let fade_out = 1.0 - smoothstep(FADE_OUT_LO, 1.0, widget_uv.x);
    let edge_fade = fade_in * fade_out;

    var color = vec4<f32>(uniforms.border_rgba.rgb, uniforms.border_rgba.a * edge_fade);

    // ── Traveling accent band ──────────────────────────────────────
    // smoothstep ≈ the CSS ease-in-out timing (the difference vs
    // cubic-bezier(0.42,0,0.58,1) is sub-frame at this period).
    let phase = fract(uniforms.time_seconds / SWEEP_PERIOD_S);
    let band_left = mix(TRAVEL_LO, TRAVEL_HI, ease_smooth_step(phase));
    let band_center = band_left + BAND_HALF_WIDTH;

    // Band gradient: transparent → accent @50% → transparent across
    // the 25% width — triangular falloff (the CSS linear-gradient
    // is linear, so a triangle, not a bell).
    let band_dist = abs(widget_uv.x - band_center) / BAND_HALF_WIDTH;
    let band = (1.0 - band_dist) * step(band_dist, 1.0);

    color = vec4<f32>(
        mix(color.rgb, uniforms.accent_rgba.rgb, band * uniforms.accent_rgba.a),
        max(color.a, band * uniforms.accent_rgba.a),
    );

    // Premultiplied out — the canvas alpha mode is premultiplied.
    return vec4<f32>(color.rgb * color.a, color.a);
}
