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
// Uniform layout (f32s=16 — lib.rs doc):
//   [0,1] res · [2] time · [3] pad · [4..8) border rgba ·
//   [8..12) accent rgba · [12..16) rect (0,0,w,h device px)

struct U {
    res     : vec2<f32>,
    time    : f32,
    _pad0   : f32,
    border  : vec4<f32>,
    accent  : vec4<f32>,
    rect    : vec4<f32>,
};

@group(0) @binding(0) var<uniform> u : U;

/// `ease` — smoothstep ease-in-out matching the CSS `ease-in-out`
/// timing function closely enough for a 7s ambient sweep (the
/// difference vs the CSS cubic-bezier(0.42,0,0.58,1) is sub-frame at
/// this period — smoothstep is the standard stand-in).
fn ease(t : f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

/// `fs_main` — one fragment of the divider quad.
@fragment
fn fs_main(@builtin(position) frag : vec4<f32>) -> @location(0) vec4<f32> {
    // frag→uv in the widget rect — device px to 0..1.
    let uv = (frag.xy - u.rect.xy) / u.rect.zw;

    // ── Hairline base ──────────────────────────────────────────────
    // Edge fade: linear-gradient(90deg, transparent, border 12%,
    // border 88%, transparent) — ramps in 0..0.12, full 0.12..0.88,
    // out 0.88..1.
    let fade_in = smoothstep(0.0, 0.12, uv.x);
    let fade_out = 1.0 - smoothstep(0.88, 1.0, uv.x);
    let edge = fade_in * fade_out;

    var col = vec4<f32>(u.border.rgb, u.border.a * edge);

    // ── Traveling accent band ──────────────────────────────────────
    // The CSS: band width 25%, translateX(-110% → 440%) over 7s
    // ease-in-out. In uv terms the band's LEFT edge travels
    // -0.275 → 1.1 (band-width fractions of the hairline width).
    // Center = left + 0.125.
    let phase = fract(u.time / 7.0);
    let left = mix(-0.275, 1.1, ease(phase));
    let cx = left + 0.125;

    // Band gradient: transparent → accent @50% → transparent across
    // the 25% width — triangular falloff (the CSS linear-gradient
    // is linear, so a triangle, not a bell).
    let d = abs(uv.x - cx) / 0.125;
    let band = (1.0 - d) * step(d, 1.0);

    col = vec4<f32>(
        mix(col.rgb, u.accent.rgb, band * u.accent.a),
        max(col.a, band * u.accent.a),
    );

    // Premultiplied out — the canvas alpha mode is premultiplied.
    return vec4<f32>(col.rgb * col.a, col.a);
}

/// `vs_main` — the shared fullscreen triangle (gpu-mini contract).
@vertex
fn vs_main(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );

    return vec4<f32>(pos[i], 0.0, 1.0);
}
