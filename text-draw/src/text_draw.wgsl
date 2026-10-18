// text_draw.wgsl — the per-character writing animation.
//
// Ports the `char-draw` CSS keyframes (draw-text/_keyframes.scss) to
// one instanced draw over a rasterized atlas:
//
//   char-draw  0%   → opacity 0, translateY(0.12em)
//              100% → opacity 1, transform none
//   easing         → cubic-bezier(0.22, 1, 0.36, 1)  ("sharp pop-in")
//   delay          → offset + i·char-delay + word cascade (folded into
//                    the sheet's per-char `delay` by the JS side, ms)
//   duration       → per-char ms (0.8s desktop / 0.5s related-mosaic)
//
// Per char: 6 synthesized verts (two triangles, no vertex buffer) —
// the vertex shader reads the storage sheet by instance_index,
// computes eased progress, offsets the quad by (1−e)·0.12·dest.h
// (the 0.12em rise — em = the char's own box height in px, exactly
// what translateY(0.12em) resolves to per char), and passes uv +
// alpha to the fragment stage which samples atlas · ink.
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
    @builtin(position) clip : vec4<f32>,
    @location(0) uv         : vec2<f32>,
    @location(1) alpha      : f32,
};

/// `vs_main` — synthesize one vertex of one char's quad: read the
/// sheet record, ease the per-char progress, offset the quad down by
/// the eased rise, project px→NDC.
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
    let p = clamp((u.time - rec.delay) / max(rec.dur, 1.0), 0.0, 1.0);
    let e = bez(p);

    // translateY(0.12em) → down 12% of the char's font size at
    // progress 0, rising to 0 at progress 1 — em is pushed per char
    // because CSS ems resolve against font-size, not the span box.
    let y_off = (1.0 - e) * 0.12 * rec.em;

    // Device px → NDC: x' = x/res·2−1, y' = 1−y/res·2 (y-down DOM →
    // y-up clip). The rect is canvas-relative (dest.xy already in the
    // canvas's device-px space — JS positions the canvas over the
    // block's box and writes canvas-local rects).
    let px = rec.dest.xy + vec2<f32>(c.x * rec.dest.z, c.y * rec.dest.w + y_off);
    let ndc = vec2<f32>(px.x / u.res.x * 2.0 - 1.0, 1.0 - px.y / u.res.y * 2.0);

    out.clip  = vec4<f32>(ndc, 0.0, 1.0);
    out.uv    = rec.uv.xy + c * (rec.uv.zw - rec.uv.xy);
    out.alpha = e;

    return out;
}

/// `fs_main` — atlas texel · ink · eased alpha. The atlas texel IS
/// white-on-transparent ink (JS rasterizes in white so the shader's
/// ink multiply tints live — the CSS animates color via the span's
/// own color, which the ink uniform reproduces).
@fragment
fn fs_main(v : Out) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas, samp, v.uv);
    let a = texel.a * u.ink.a * v.alpha;

    return vec4<f32>(u.ink.rgb * a, a);
}
