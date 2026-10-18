// burger.wgsl — the nav hamburger's three hairline bars.
//
// WGSL port of burger-button's fragment shader
// (core/utils/canvas/widgets/burger-button/shaders.ts) — the port is
// faithful: every constant below matches the GLSL original so the
// rendered pixels are identical when the same uniforms arrive.
//
// ════════════════════════════════════════════════════════════════════
// HOW THIS DIFFERS FROM A TRADITIONAL UI ELEMENT
// ════════════════════════════════════════════════════════════════════
//
// A normal page draws the burger as three <div> bars (or an SVG). The
// browser rasterizes them once; animating means moving DOM nodes —
// layout → paint → composite every frame (~ms each at worst).
//
// Here there is NO geometry at all: a single fullscreen triangle covers
// the widget rect, and this fragment shader answers "is this PIXEL part
// of a bar?" per pixel per frame using signed-distance-field math.
// Cost: one draw call + ~30 ALU ops per pixel — for a 36×14 widget that
// is ~15k ops total, nanoseconds on GPU. No layout pass exists; the
// shader IS the layout.
//
// Uniform layout (f32s, must match burger.rs::UNIFORM_F32S = 8):
//   [0]   time     seconds since widget mount — drives every sine below
//   [1,2] res      widget pixel size (device px)
//   [3]   dark     0.0 = light theme, 1.0 = dark theme (lerp factor)
//   [4..7] rect    widget clip rect in canvas CSS px (x, y, w, h)

struct Uniforms {
    time: f32,
    res: vec2<f32>,
    dark: f32,
    rect: vec4<f32>, // x, y, w, h — canvas CSS px (this widget's clip)
}

@group(0) @binding(0) var<uniform> u: Uniforms;

// ── SDF primitives ───────────────────────────────────────────────────
//
// sdCapsule returns the signed distance from point `p` to a capsule:
// a line segment a→b thickened by radius `r` (a "stadium" — rectangle
// with semicircular caps). Negative = inside, 0 = edge, positive =
// outside. This is the fundamental trick: the BARS are not drawn, their
// distance FIELD is computed, and pixels within the field get ink.
//
// The math, step by step:
//   pa = p - a         → pixel position relative to segment start
//   ba = b - a         → the segment itself as a vector
//   dot(pa,ba)/dot(ba,ba) → parametric position t of the closest point
//     on the INFINITE line (projection scalar; t=0 at a, t=1 at b)
//   clamp(t,0,1)       → restrict to the SEGMENT, giving the closest
//                        point ON the capsule's spine
//   length(pa - ba*h)  → euclidean distance to that closest spine point
//   ... - r            → subtract thickness → signed distance to surface
fn sdCapsule(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// A horizontal bar as a capsule: height h → radius h/2, width w → the
// spine runs from 0.5-halfL to 0.5+halfL (UV space, centered at x=0.5).
// `max(w*0.5 - r, 0.0)` guards the degenerate w<2r case (a capsule whose
// "spine" is negative-length — the clamp keeps it a pure circle).
fn sdHBar(p: vec2<f32>, y: f32, w: f32, h: f32) -> f32 {
    let r = h * 0.5;
    let halfL = max(w * 0.5 - r, 0.0);
    return sdCapsule(p, vec2<f32>(0.5 - halfL, y), vec2<f32>(0.5 + halfL, y), r);
}

// ── Vertex stage: the fullscreen triangle trick ──────────────────────
//
// Three vertices at (-1,-1), (3,-1), (-1,3) form ONE triangle covering
// the entire clip plane: any pixel in [-1,1]² lies inside it (the two
// "3.0" corners just extend the triangle past the edges — everything
// beyond the viewport is clipped free by the rasterizer).
// Why not a 4-vertex quad? A quad is two triangles sharing a diagonal;
// some GPUs rasterize the shared edge twice → the fragment shader could
// run twice on the seam (visible banding in additive blends). One
// triangle = no seam, one fragment per pixel, guaranteed.
// The render-pass scissor rect (set CPU-side in lib.rs::frame) clips
// this to the widget's rect, so we only pay for the widget's pixels.
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

// ── Fragment stage ───────────────────────────────────────────────────

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // `frag` arrives in SURFACE pixels (device px, whole canvas). The
    // original shader ran on its own canvas where frag==widget pixel.
    // Mapping (frag - rect.xy)/rect.wh converts to widget-UV 0..1 so the
    // ported math is byte-identical to the original canvas version.
    // NOTE: rect is CSS px and frag is device px — the uniforms are
    // written pre-scaled by the bridge so this division is exact.
    let uv = vec2<f32>(
        (frag.x - u.rect.x) / u.rect.z,
        (frag.y - u.rect.y) / u.rect.w,
    );
    let t = u.time;

    // Bar geometry in UV units. SAFE RANGES:
    //   barW 0.54  — bar half-extent 0.27 each side of center; >0.9 and
    //                bars reach the widget's edge and look cropped.
    //   barH 0.045 — hairline ~2px on a 44px-tall widget; raise to 0.07
    //                for a chunkier bar, keep < gap or bars merge.
    //   gap  0.16  — center spacing ±0.16 around y=0.5; must stay > barH.
    let barW = 0.54;
    let barH = 0.045;
    let gap = 0.16;

    // ── Wave ripple ──────────────────────────────────────────────────
    // sin(uv.x * 5.0 - t * 0.7 + phase) — a traveling sine:
    //   uv.x * 5.0  → ~0.8 wavelengths across the bar (spatial freq)
    //   t * 0.7     → wave travels right at 0.7/5 ≈ 0.14 uv/s ≈ 9s period
    //   * 0.009     → ±0.9% vertical amplitude — barely-visible shimmer.
    //     Raise toward 0.02 for a livelier wave; >0.03 makes hairlines
    //     visibly crawl (anti-aliasing shimmer) at 120/240Hz.
    //   phase 0/1.3/2.6 → thirds of ~2π (1.256) — bars never crest
    //     together, so the group reads as one continuous surface.
    let wave1 = sin(uv.x * 5.0 - t * 0.7) * 0.009;
    let wave2 = sin(uv.x * 5.0 - t * 0.7 + 1.3) * 0.009;
    let wave3 = sin(uv.x * 5.0 - t * 0.7 + 2.6) * 0.009;

    // ── Bobbing ──────────────────────────────────────────────────────
    // Per-bar vertical bob at 0.55 rad/s (~11s period), amplitude 0.4%
    // of widget height. Phases 1.2/2.4 keep it uncorrelated with the
    // ripple above — two detuned frequencies never visibly "beat".
    let bob1 = sin(t * 0.55) * 0.004;
    let bob2 = sin(t * 0.55 + 1.2) * 0.004;
    let bob3 = sin(t * 0.55 + 2.4) * 0.004;

    // ── Breathing ────────────────────────────────────────────────────
    // grow oscillates 0.96..1.04 at 0.45 rad/s (~14s period). Outer bars
    // scale by `grow`; the middle by `2.0 - grow` — it counter-shrinks
    // so total ink mass is constant (eye reads it as soft pressure, not
    // flicker). ±0.03 base offsets keep the middle bar visually equal.
    let grow = 1.0 + 0.04 * sin(t * 0.45);

    // Each bar's SDF evaluated at the wave-shifted sample point. Moving
    // the SAMPLE by -wave is mathematically identical to moving the BAR
    // by +wave (SDF convention: transform the point, not the shape).
    let d1 = sdHBar(vec2<f32>(uv.x, uv.y + wave1), 0.5 - gap + bob1, barW * grow, barH);
    let d2 = sdHBar(vec2<f32>(uv.x, uv.y + wave2), 0.5 + bob2, (barW + 0.03) * (2.0 - grow), barH);
    let d3 = sdHBar(vec2<f32>(uv.x, uv.y + wave3), 0.5 + gap + bob3, (barW - 0.03) * grow, barH);

    // Union of the three bars = minimum distance (standard SDF union).
    let bar = min(min(d1, d2), d3);

    // ── Anti-aliasing via screen-space derivatives ───────────────────
    // fwidth(bar) = |∂bar/∂x| + |∂bar/∂y| — how fast the distance field
    // changes per pixel. The AA band is sized in FIELD units, not px:
    // smoothstep over ±1.1×fwidth produces a ~2.2px soft edge that is
    // sharp at ANY zoom/DPR — the whole reason SDFs beat alpha sprites.
    // max(...,1e-4) guards the degenerate case where the field is flat
    // (fwidth=0) — a 0-width smoothstep would produce a hard alias.
    let px = max(fwidth(bar), 1e-4);
    let alpha = 1.0 - smoothstep(-px * 1.1, px * 1.1, bar);
    // Soft halo outside the bars: field 0..5.5px-wide → 20% luminance
    // bloom. Keep the coefficient low (<0.3) or hairlines bloom fat.
    let glow = (1.0 - smoothstep(0.0, px * 5.5, bar)) * 0.2;

    // Luminance breathing ±2% at 0.5 rad/s (~12.6s) — slow enough to be
    // sub-perceptual as motion but keeps the icon alive on idle.
    let pulse = 0.98 + 0.02 * sin(t * 0.5);

    // Theme inks — colDark is the ink ON dark themes (near-white with a
    // cool cast), colLight the inverse. `mix(a,b,u.dark)` lerps:
    // dark=1.0 → colDark. Numbers match the original GLSL token-for-token.
    let colDark = vec3<f32>(0.95, 0.97, 1.0) * pulse;
    let colLight = vec3<f32>(0.10, 0.10, 0.14) * pulse;
    let col = mix(colLight, colDark, u.dark);

    // The halo color stays in the ink's hue family but desaturated —
    // using the ink color for glow fringes visibly at the AA edge.
    let glowDark = vec3<f32>(0.55, 0.58, 0.65);
    let glowLight = vec3<f32>(0.15, 0.15, 0.22);
    let glowCol = mix(glowLight, glowDark, u.dark);

    // Final composite: bar ink + separate halo, alpha dominated by the
    // bar with a 25% floor from the halo so pure-halo pixels still show.
    let finalCol = col * alpha + glowCol * glow;
    let finalAlpha = max(alpha, glow * 0.25);

    return vec4<f32>(finalCol, finalAlpha);
}
