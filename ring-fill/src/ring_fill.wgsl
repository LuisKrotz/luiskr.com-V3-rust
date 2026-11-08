// ring_fill.wgsl — instanced border-fill sweep for the route-search
// field (rounded-rect) and circular buttons.
//
// Ports the CSS stroke-dashoffset sweep (the image-3 "border draws
// itself" affordance):
//
//   track — faint full hairline outline, always visible.
//   fill  — an arc that travels the perimeter from the top-left
//           clockwise as progress runs 0 → 1; the JS side eases it
//           over ~350 ms, matching the CSS transition it replaces.
//   head  — a brighter dot riding the arc tip.
//
// Tile-sheet layout (8 f32s — lib.rs doc):
//   [0,1] center (device px) · [2] halfW · [3] halfH · [4] cornerR ·
//   [5] progress 0..1 · [6] shape (0=circle, 1=rounded rect) · [7] pad
//
// Uniform layout (f32s=16):
//   [0,1] res · [2] time_s · [3] hair half-width ·
//   [4..8) track rgba · [8..12) fill rgba · [12..16) head rgba

struct U {
    res    : vec2<f32>,
    time   : f32,
    hair   : f32,
    track  : vec4<f32>,
    fill   : vec4<f32>,
    head   : vec4<f32>,
};

struct TileRec {
    center   : vec2<f32>,
    halfw    : f32,
    halfh    : f32,
    cornerr  : f32,
    progress : f32,
    shape    : f32,
    pad      : f32,
};

@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var<storage, read> tiles : array<TileRec>;

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
    @location(0) frag        : vec2<f32>,
    @location(1) inst        : f32,
};

/// `vs_main` — each instance covers the tile's bounding box plus hairline slack.
@vertex
fn vs_main(
    @builtin(vertex_index) v : u32,
    @builtin(instance_index) inst : u32,
) -> Out {
    var out : Out;
    let c = corner(v);
    let rec = tiles[inst];
    let slack = u.hair * 4.0 + 1.0;
    let lo = rec.center - vec2<f32>(rec.halfw, rec.halfh) - vec2<f32>(slack);
    let sz = vec2<f32>(rec.halfw * 2.0 + slack * 2.0, rec.halfh * 2.0 + slack * 2.0);

    let px = lo + c * sz;
    let ndc = vec2<f32>(px.x / u.res.x * 2.0 - 1.0, 1.0 - px.y / u.res.y * 2.0);

    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.frag = px;
    out.inst = f32(inst);

    return out;
}

/// Rounded-rect SDF — signed distance to the box boundary.
fn sd_rrect(p : vec2<f32>, b : vec2<f32>, r : f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r);

    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

/// Perimeter parameter t∈[0,1] of the closest rrect border point —
/// 8 segments walked clockwise from the top edge's left end.
fn rrect_t(p : vec2<f32>, hw : f32, hh : f32, r : f32) -> f32 {
    let tw = 2.0 * (hw - r);
    let th = 2.0 * (hh - r);
    let arc = 1.5707963 * r;
    let total = 2.0 * (tw + th) + 4.0 * arc;

    var best_d = 1e9;
    var best_t = 0.0;
    let off = vec2<f32>(hw - r, hh - r);

    // top edge
    var pr = clamp(p.x, -off.x, off.x);
    var d = abs(p.y + hh) + abs(p.x - pr);
    var l = pr + off.x;

    best_d = d;
    best_t = l;

    // TR arc (θ −π/2 → 0)
    var a = atan2(p.y + off.y, p.x - off.x) + 1.5707963;

    if (a < 0.0) { a = a + 6.2831853; }

    a = clamp(a, 0.0, 1.5707963);
    d = abs(length(p - vec2<f32>(off.x, -off.y)) - r);
    l = tw + a * r;

    if (d < best_d) { best_d = d; best_t = l; }

    // right edge
    pr = clamp(p.y, -off.y, off.y);
    d = abs(p.x - hw) + abs(p.y - pr);
    l = tw + arc + (pr + off.y);

    if (d < best_d) { best_d = d; best_t = l; }

    // BR arc (θ 0 → π/2)
    a = atan2(p.y - off.y, p.x - off.x);

    if (a < 0.0) { a = a + 6.2831853; }

    a = clamp(a, 0.0, 1.5707963);
    d = abs(length(p - off) - r);
    l = tw + arc + th + a * r;

    if (d < best_d) { best_d = d; best_t = l; }

    // bottom edge (right → left)
    pr = clamp(p.x, -off.x, off.x);
    d = abs(p.y - hh) + abs(p.x - pr);
    l = tw + arc + th + arc + (off.x - pr);

    if (d < best_d) { best_d = d; best_t = l; }

    // BL arc (θ π/2 → π)
    a = atan2(p.y - off.y, p.x + off.x);

    if (a < 0.0) { a = a + 6.2831853; }

    a = clamp(a, 1.5707963, 3.1415927);
    d = abs(length(p + vec2<f32>(off.x, -off.y)) - r);
    l = tw + arc + th + arc + tw + (a - 1.5707963) * r;

    if (d < best_d) { best_d = d; best_t = l; }

    // left edge (bottom → top)
    pr = clamp(p.y, -off.y, off.y);
    d = abs(p.x + hw) + abs(p.y - pr);
    l = 2.0 * (tw + arc) + th + (off.y - pr);

    if (d < best_d) { best_d = d; best_t = l; }

    // TL arc (θ π → 3π/2)
    a = atan2(p.y + off.y, p.x + off.x);

    if (a < 0.0) { a = a + 6.2831853; }

    a = clamp(a, 3.1415927, 4.712389);
    d = abs(length(p + off) - r);
    l = 2.0 * (tw + arc + th) + (a - 3.1415927) * r;

    if (d < best_d) { best_d = d; best_t = l; }

    return clamp(best_t / total, 0.0, 1.0);
}

/// Point on the rrect perimeter at arc-length t — mirrors rrect_t's walk.
fn rrect_point(t : f32, hw : f32, hh : f32, r : f32) -> vec2<f32> {
    let tw = 2.0 * (hw - r);
    let th = 2.0 * (hh - r);
    let arc = 1.5707963 * r;
    let off = vec2<f32>(hw - r, hh - r);
    var s = t;

    if (s < tw) { return vec2<f32>(-off.x + s, -hh); }
    s = s - tw;

    if (s < arc) {
        let a = s / r - 1.5707963;

        return vec2<f32>(off.x + r * cos(a), -off.y + r * sin(a));
    }
    s = s - arc;

    if (s < th) { return vec2<f32>(hw, -off.y + s); }
    s = s - th;

    if (s < arc) {
        let a = s / r;

        return vec2<f32>(off.x + r * cos(a), off.y + r * sin(a));
    }
    s = s - arc;

    if (s < tw) { return vec2<f32>(off.x - s, hh); }
    s = s - tw;

    if (s < arc) {
        let a = s / r + 1.5707963;

        return vec2<f32>(-off.x + r * cos(a), off.y + r * sin(a));
    }
    s = s - arc;

    if (s < th) { return vec2<f32>(-hw, off.y - s); }
    s = s - th;

    let a = s / r + 3.1415927;

    return vec2<f32>(-off.x + r * cos(a), -off.y + r * sin(a));
}

/// `fs_main` — track ring + progress arc + head dot, per shape.
@fragment
fn fs_main(v : Out) -> @location(0) vec4<f32> {
    let rec = tiles[u32(v.inst + 0.5)];
    let dvec = v.frag - rec.center;

    var band = 0.0;
    var t = 0.0;
    var tip = rec.center;

    if (rec.shape < 0.5) {
        // circle — radius = halfW; angle sweep from 12 o'clock.
        let dist = length(dvec);

        band = abs(dist - rec.halfw);

        var a = atan2(dvec.x, -dvec.y) / 6.2831853;

        if (a < 0.0) { a = a + 1.0; }

        t = a;

        let ta = rec.progress * 6.2831853;

        tip = rec.center + rec.halfw * vec2<f32>(sin(ta), -cos(ta));
    } else {
        // rounded rect — perimeter sweep from the top edge's left end.
        band = abs(sd_rrect(dvec, vec2<f32>(rec.halfw, rec.halfh), rec.cornerr));
        t = rrect_t(dvec, rec.halfw, rec.halfh, rec.cornerr);

        let tw = 2.0 * (rec.halfw - rec.cornerr);
        let th = 2.0 * (rec.halfh - rec.cornerr);
        let arc = 1.5707963 * rec.cornerr;
        let total = 2.0 * (tw + th) + 4.0 * arc;

        tip = rec.center + rrect_point(rec.progress * total, rec.halfw, rec.halfh, rec.cornerr);
    }

    let ring = 1.0 - smoothstep(u.hair - 1.0, u.hair + 1.0, band);

    let head_r = u.hair * 3.0;
    let head = 1.0 - smoothstep(head_r - 1.0, head_r + 1.0, length(v.frag - tip));

    if (ring <= 0.001 && head <= 0.001) {
        discard;
    }

    // ── Light physics ─────────────────────────────────────────────────────
    // The fill is a light beam sweeping the border: chromatic refraction
    // fringes at the arc edge, an inverse-square halo cast onto the surface
    // (reflection), and spark particles riding the head.

    // Wrap-around distance along the perimeter to the beam tail.
    let wrapped = abs(t - rec.progress);
    let arc_d = min(wrapped, 1.0 - wrapped);

    // The light tip draws the border: already-drawn stays lit (stroke
    // remains), and a bright trail decays behind the head like a comet
    // tail — reads as light traveling, not a filling gauge.
    let drawn = 1.0 - smoothstep(rec.progress - 0.003, rec.progress + 0.003, t);

    let trail = drawn * exp(-arc_d * 14.0);

    var col = u.track * ring;

    // Persistent drawn stroke — dimmer than the track so the tail leads.
    col = col + u.fill * ring * drawn * 0.55;

    // Comet tail — the hot part just behind the tip.
    col = col + u.fill * ring * trail;

    // Refraction fringes — chromatic split at the trail's live edge.
    let fringe_e = 0.02;
    let edge_r = trail * exp(-arc_d * 8.0);
    let edge_b = drawn * (1.0 - exp(-arc_d * 30.0));

    col.r = col.r + u.fill.r * ring * edge_r * fringe_e * 30.0;
    col.b = col.b + u.fill.b * ring * edge_b * fringe_e * 6.0;

    // Reflection — light cast off the ring onto the surface, falls off
    // with distance-to-beam (perimeter arc) × distance-to-border.
    let cast = exp(-arc_d * 26.0) * exp(-band * 0.55) * u.fill.a;

    col.rgb = col.rgb + u.fill.rgb * cast * 0.35;
    col.a = max(col.a, cast * 0.22);

    // Head — hot core + inverse-square halo + sparks.
    let head_d = length(v.frag - tip);
    let core = 1.0 - smoothstep(head_r * 0.5, head_r, head_d);
    let halo = u.head.a * head_r * 2.5 / max(head_d, head_r * 0.5) - u.head.a * 0.4;

    if (rec.progress > 0.001 && rec.progress < 0.999) {
        col = mix(col, u.head, core);
        col.rgb = col.rgb + u.head.rgb * clamp(halo, 0.0, 1.0) * 0.5;
        col.a = max(col.a, clamp(halo, 0.0, 1.0) * 0.5);

        // 3 sparks — deterministic orbiters flickering with time.
        for (var s = 0u; s < 3u; s = s + 1u) {
            let sf = f32(s) * 2.0943951;
            let ang = sf + u.time * (1.7 + f32(s) * 0.9);
            let rad = head_r * (1.6 + 0.5 * sin(u.time * 3.0 + sf * 7.0));
            let sp = tip + rad * vec2<f32>(cos(ang), sin(ang));
            let sd = 1.0 - smoothstep(1.0, 2.2, length(v.frag - sp));
            let tw = 0.5 + 0.5 * sin(u.time * 9.0 + sf * 13.0);

            col = mix(col, u.head, sd * tw);
        }
    }

    let alpha = max(col.a, ring * u.track.a * (1.0 - drawn));

    return vec4<f32>(clamp(col.rgb, vec3<f32>(0.0), vec3<f32>(1.5)), clamp(alpha, 0.0, 1.0));
}
