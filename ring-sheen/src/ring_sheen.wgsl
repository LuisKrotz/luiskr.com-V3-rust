// ring_sheen.wgsl — instanced tile border rings with a drifting sheen.
//
// Ports the masked-gradient ring on `.home-mosaic-item` /
// `.related-item` (home-mosaic/_item.scss, internals/_related.scss):
//
//   ring    — hairline band at each tile's rounded-rect edge, drawn
//             as an SDF: |dist(frag, rounded_rect)| < hair.
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
// Tile-sheet layout (8 f32s — lib.rs doc):
//   [0..4) rect (x,y,w,h device px) · [4] radius · [5] phase · [6,7] pad
//
// Uniform layout (f32s=16):
//   [0,1] res · [2] time_s · [3] hair half-width ·
//   [4..8) edge rgba · [8..12) accent rgba · [12..16) pad

struct U {
    res    : vec2<f32>,
    time   : f32,
    hair   : f32,
    edge   : vec4<f32>,
    accent : vec4<f32>,
    _pad   : vec4<f32>,
};

struct TileRec {
    rect   : vec4<f32>,
    radius : f32,
    phase  : f32,
    pad    : vec2<f32>,
};

@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var<storage, read> tiles : array<TileRec>;

/// `corner` — vertex_index → unit-quad corner (two tris, no buffers).
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
    @builtin(position) clip  : vec4<f32>,
    @location(0) frag        : vec2<f32>,   // device-px position
    @location(1) inst        : f32,         // tile index (flat — fs reads the same record)
};

/// `vs_main` — place the tile's quad: each instance covers the tile
/// rect (plus one hairline of slack so the ring's outer edge isn't
/// clipped by its own quad bounds).
@vertex
fn vs_main(
    @builtin(vertex_index) v : u32,
    @builtin(instance_index) inst : u32,
) -> Out {
    var out : Out;
    let c = corner(v);
    let rec = tiles[inst];
    let slack = u.hair + 1.0;

    let px = rec.rect.xy - slack
           + c * (rec.rect.zw + slack * 2.0);
    let ndc = vec2<f32>(px.x / u.res.x * 2.0 - 1.0, 1.0 - px.y / u.res.y * 2.0);

    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.frag = px;
    out.inst = f32(inst);

    return out;
}

/// `sd_round_box` — signed distance to a rounded rectangle: negative
/// inside, zero on the edge, positive outside. The classic SDF:
///   d = length(max(|p−c|−(b−r), 0)) + min(max(q.x,q.y), 0) − r
/// where b is the half-size and r the corner radius. The ring mask is
/// |d| < hair — a band straddling the edge.
fn sd_round_box(p : vec2<f32>, center : vec2<f32>, half : vec2<f32>, r : f32) -> f32 {
    let q = abs(p - center) - (half - vec2<f32>(r));

    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

/// `fs_main` — ring mask × diagonal sheen × breath.
@fragment
fn fs_main(v : Out) -> @location(0) vec4<f32> {
    let rec = tiles[u32(v.inst + 0.5)];
    let center = rec.rect.xy + rec.rect.zw * 0.5;
    let half = rec.rect.zw * 0.5;

    // Ring mask — 1 inside the hairline band, antialiased at the SDF
    // edges over ~1px so the hairline doesn't shimmer.
    let d = sd_round_box(v.frag, center, half, rec.radius);
    let ring = 1.0 - smoothstep(u.hair - 1.0, u.hair + 1.0, abs(d));

    if (ring <= 0.001) {
        discard;
    }

    // Diagonal sheen — the 120° gradient direction (CSS deg: 0 =
    // upward, clockwise — 120° → direction (sin120, −cos120) =
    // (0.866, 0.5) in y-down screen space). Project frag onto it and
    // normalize by the tile's diagonal extent → t ∈ [0,1].
    let dir = vec2<f32>(0.8660254, 0.5);
    let span = dot(rec.rect.zw, abs(dir));
    let t = dot(v.frag - rec.rect.xy, dir) / max(span, 1.0);

    // The band slides along the diagonal over the 9s-alternate loop —
    // alternate means 9s out + 9s back = an 18s round trip, so the
    // triangle wave's period is 18s, not 9s.
    let phase = fract((u.time / 18.0) + rec.phase);
    let ping = 1.0 - abs(phase * 2.0 - 1.0);           // 0→1→0
    let pos = mix(-0.3, 1.3, ping);                    // off→through→off

    // Band shape — the CSS stops: transparent 32% → edge 45% →
    // accent 50% → edge 55% → transparent 68%. Scaled to a band of
    // half-width 0.18 around `pos`: inner accent at ±0.05, edge tint
    // at ±0.10, transparent past ±0.18.
    let dd = abs(t - pos);
    let core = 1.0 - smoothstep(0.0, 0.05, dd);
    let tint = 1.0 - smoothstep(0.05, 0.18, dd);

    // Breath — ::before cross-fade 0.15↔0.9 ease-in-out alternate on
    // the same 18s round trip (ping already carries the alternate
    // timing; the cubic smoothstep ≈ ease-in-out).
    let breath = 0.15 + 0.75 * ping * ping * (3.0 - 2.0 * ping);

    let sheen_col = mix(u.edge.rgb, u.accent.rgb, core);
    let sheen_a = mix(u.edge.a, u.accent.a, core) * tint;
    let a = ring * sheen_a * breath;

    return vec4<f32>(sheen_col * a, a);
}
