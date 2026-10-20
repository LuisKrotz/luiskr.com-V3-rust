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
//   [0..4) dest (x,y,w,h device px) · [4..8) uv (u0,v0,u1,v1) ·
//   [8] delay ms · [9] dur ms · [10,11] pad
//
// Uniform layout (f32s=16):
//   [0,1] res · [2] time_ms (elapsed since trigger) · [3] count ·
//   [4..8) ink rgba · [8..16) pad

struct U {
    res   : vec2<f32>,
    time  : f32,
    count : f32,
    ink   : vec4<f32>,
    _pad  : vec4<f32>,
};

struct CharRec {
    dest  : vec4<f32>,   // x,y,w,h device px
    uv    : vec4<f32>,   // u0,v0,u1,v1 in the atlas
    delay : f32,
    dur   : f32,
    em    : f32,         // the char's font-size px — translateY(0.12em) unit
    pad   : f32,
};

@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var<storage, read> chars : array<CharRec>;
@group(0) @binding(2) var atlas : texture_2d<f32>;
@group(0) @binding(3) var samp : sampler;

/// `bez` — evaluate cubic-bezier(0.22,1,0.36,1) at progress `p`.
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
fn bez(p : f32) -> f32 {
    var t = p;

    for (var i = 0; i < 4; i++) {
        let o = 1.0 - t;
        let x = 3.0 * o * o * t * 0.22 + 3.0 * o * t * t * 0.36 + t * t * t;
        let dx = 3.0 * o * o * 0.22 + 6.0 * o * t * (0.36 - 0.22) + 3.0 * t * t * (1.0 - 0.36);

        if (abs(dx) < 1e-6) { break; }

        t = clamp(t - (x - p) / dx, 0.0, 1.0);
    }

    // y(t) with P1y=P2y=1 → 1−(1−t)³.
    let o = 1.0 - t;

    return 1.0 - o * o * o;
}

/// `hash1` — deterministic per-char pseudo-random in [0,1). Seeded by
/// the instance index so each letter picks a stable (but varied)
/// stroke direction + wobble phase across frames — no per-frame
/// drift, no storage.
fn hash1(n : f32) -> f32 {
    return fract(sin(n * 127.1 + 311.7) * 43758.5453);
}

/// `corner` — vertex_index → unit-quad corner: 0=(-1,-1) 1=(1,-1)
/// 2=(-1,1) 3=(-1,1) 4=(1,-1) 5=(1,1) — two triangles, no index
/// buffer, no vertex buffer.
fn corner(v : u32) -> vec2<f32> {
    let t = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );

    return t[v];
}

struct Out {
    @builtin(position) clip          : vec4<f32>,
    @location(0) uv                  : vec2<f32>,
    /// Unit-quad coordinate (0..1 across the char box) — the mask's
    /// working space.
    @location(1) local               : vec2<f32>,
    /// Linear stroke progress 0..1 — drives the reveal edge.
    @location(2) p                   : f32,
    /// Whole-glyph alpha (the eased CSS entrance component).
    @location(3) alpha               : f32,
    /// Per-char seed — flat so every fragment of the quad shares the
    /// same stroke direction/wobble (interpolating it would tear the
    /// mask).
    @location(4) @interpolate(flat) seed : f32,
};

/// `vs_main` — synthesize one vertex of one char's quad: read the
/// sheet record, ease the per-char progress, offset the quad down by
/// the eased rise + hand tremor, project px→NDC.
@vertex
fn vs_main(
    @builtin(vertex_index) v : u32,
    @builtin(instance_index) inst : u32,
) -> Out {
    var out : Out;
    let c = corner(v);
    let rec = chars[inst];

    // Per-char eased progress — the CSS keyframe's timing curve
    // evaluated at (time − delay)/dur, clamped to the keyframe span.
    // `p` stays linear: a pen moves at near-constant speed through a
    // letter, so the stroke mask follows `p` while the entrance
    // (rise + faint alpha ramp) keeps the eased `e`.
    let p = clamp((u.time - rec.delay) / max(rec.dur, 1.0), 0.0, 1.0);
    let e = bez(p);

    // translateY(0.12em) → down 12% of the char's font size at
    // progress 0, rising to 0 at progress 1.
    let y_off = (1.0 - e) * 0.12 * rec.em;

    // Hand tremor — two incommensurate sines (~5Hz, ~4.3Hz at ms
    // resolution ≈ the physiological tremor band) fading to zero as
    // the letter completes: the nib wobbles while the stroke is live
    // and the finished letter sits dead still. Kept at ~0.25px peak —
    // sub-pixel life, not blur (a multi-px tremor smears every stroke).
    let seed = f32(inst);
    let tremor = (1.0 - e) * 0.25 * vec2<f32>(
        sin(u.time * 0.031 + seed * 7.13),
        cos(u.time * 0.027 + seed * 3.71),
    );

    // Device px → NDC: x' = x/res·2−1, y' = 1−y/res·2 (y-down DOM →
    // y-up clip).
    let px = rec.dest.xy + tremor + vec2<f32>(c.x * rec.dest.z, c.y * rec.dest.w + y_off);
    let ndc = vec2<f32>(px.x / u.res.x * 2.0 - 1.0, 1.0 - px.y / u.res.y * 2.0);

    out.clip  = vec4<f32>(ndc, 0.0, 1.0);
    out.uv    = rec.uv.xy + c * (rec.uv.zw - rec.uv.xy);
    out.local = c;
    out.p     = p;
    out.alpha = e;
    out.seed  = seed;

    return out;
}

/// `fs_main` — atlas texel · ink, masked by the baked traversal field.
///
/// texel.r holds the pen-visit order for that pixel (0 = stroke entry,
/// 1 = stroke end / unreachable / paper); texel.a is the glyph coverage.
/// The frontier `edge` advances linearly — a nib moves at near-constant
/// speed — and every pixel whose visit-order is behind it is ink; ahead
/// of it is paper. The transition window is ~1.2% of the stroke span —
/// about one texel of anti-alias, so the written stroke is as crisp as
/// the rasterized glyph underneath. A tight Gaussian bead centred ON the
/// frontier overfills coverage slightly and darkens — the pen's wet
/// deposit — so the writing front reads as a nib, not a clipping plane.
@fragment
fn fs_main(v : Out) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas, samp, v.uv);

    // 4% overshoot so the tail of the stroke finishes before p hits 1 —
    // the visible "lift-off" dwell before the pen hops to the next char.
    let edge = v.p * 1.04;
    let write = 1.0 - smoothstep(edge - 0.012, edge, texel.r);

    // Nib bead — a tight ridge straddling the frontier: coverage overfills
    // ~0.8% of the span around the edge, ink darkens under pressure.
    let nib_d = (texel.r - edge) / 0.008;
    let nib = exp(-nib_d * nib_d);

    let cov = min(write + nib * 0.25, 1.0);
    let a = texel.a * u.ink.a * v.alpha * cov;
    let rgb = u.ink.rgb * (1.0 - nib * 0.12);

    return vec4<f32>(rgb * a, a);
}
