// corner_brackets.wgsl — L-brackets at a button's TOP-LEFT and
// BOTTOM-RIGHT corners.
//
// Ports the ::before/::after corner-bracket expansion on
// `.pref-done-btn` / `.awards-selected-action` /
// `.awards-explore-action` (preferences/_footer.scss,
// awards-carousel/_selected.scss, _explore.scss):
//
//   rest   — each corner is a square box ($space-md), so each arm's
//            length = arm0 (the stub).
//   hover  — the box grows to calc(100% − inset) in BOTH axes, so the
//            TL bracket's top arm spans the whole top edge and its
//            left arm the whole left edge (the BR bracket mirrors);
//            arm1 carries the resolved device-px spans pushed by JS,
//            over 0.25s — the JS side drives the eased `hover` scalar
//            so the shader stays stateless.
//   tint   — the border color is uniform per surface (edge color);
//            `accent` carries the hover tint for surfaces that use it
//            (the CSS variants animate border-color on some buttons).
//
// The L shape: each bracket owns a horizontal arm (the box's top or
// bottom border) and a vertical arm (its left or right border). A
// fragment is lit inside ANY arm band — the mask is a union of 4
// axis-aligned slabs, min-distance, antialiased ~1px.
//
// Uniform layout (f32s=16 — lib.rs doc):
//   [0,1] res · [2] hover · [3] hair · [4,5] arm0 (w,h) ·
//   [6,7] arm1 (w,h) · [8..12) edge rgba · [12..16) accent rgba

struct U {
    res    : vec2<f32>,
    hover  : f32,
    hair   : f32,
    arm0   : vec2<f32>,
    arm1   : vec2<f32>,
    edge   : vec4<f32>,
    accent : vec4<f32>,
};

@group(0) @binding(0) var<uniform> u : U;

struct Out {
    @builtin(position) clip : vec4<f32>,
    @location(0) frag       : vec2<f32>,   // device-px position
};

/// `vs_main` — fullscreen triangle: 3 verts → clip positions
/// (−1,−1) (3,−1) (−1,3); frag carries device-px for the SDF.
@vertex
fn vs_main(@builtin(vertex_index) v : u32) -> Out {
    var out : Out;
    var pos = vec2<f32>(-1.0, -1.0);

    if (v == 1u) {
        pos = vec2<f32>(3.0, -1.0);
    } else if (v == 2u) {
        pos = vec2<f32>(-1.0, 3.0);
    }

    out.clip = vec4<f32>(pos, 0.0, 1.0);
    out.frag = vec2<f32>(
        (pos.x + 1.0) * 0.5 * u.res.x,
        (1.0 - pos.y) * 0.5 * u.res.y,
    );

    return out;
}

/// `slab` — distance outside an axis-aligned band: how far `p` is
/// from the rectangle [lo.x,hi.x]×[lo.y,hi.y]. `max`-of-axis-excess
/// collapses to ≤0 inside — the arm union is `min` over the 8 slabs.
fn slab(p : vec2<f32>, lo : vec2<f32>, hi : vec2<f32>) -> f32 {
    let e = max(lo - p, p - hi);

    return max(e.x, e.y);
}

/// `fs_main` — the 4-slab union × tint, ~1px-antialiased.
@fragment
fn fs_main(v : Out) -> @location(0) vec4<f32> {
    let p = v.frag;
    let aw = mix(u.arm0.x, u.arm1.x, u.hover);   // box width  (arm x)
    let ah = mix(u.arm0.y, u.arm1.y, u.hover);   // box height (arm y)
    let hw = u.hair * 2.0;                        // band thickness

    // The 4 arms — TL bracket (border-top + border-left) and BR
    // bracket (border-bottom + border-right), flush with the rect
    // edges exactly like the pseudo-elements' `top/left/bottom/right:0`:
    //
    //   TL: top band   [0,aw]·[0,hw]    +  left band   [0,hw]·[0,ah]
    //   BR: bottom band [w−aw,w]·[h−hw,h] + right band [w−hw,w]·[h−ah,h]
    let w = u.res.x;
    let h = u.res.y;

    var d =    slab(p, vec2<f32>(0.0,     0.0),     vec2<f32>(aw, hw));
    d = min(d, slab(p, vec2<f32>(0.0,     0.0),     vec2<f32>(hw, ah)));
    d = min(d, slab(p, vec2<f32>(w - aw,  h - hw),  vec2<f32>(w,  h)));
    d = min(d, slab(p, vec2<f32>(w - hw,  h - ah),  vec2<f32>(w,  h)));

    let mask = 1.0 - smoothstep(-1.0, 1.0, d);

    if (mask <= 0.001) {
        discard;
    }

    let col = mix(u.edge, u.accent, u.hover);
    let a = col.a * mask;

    return vec4<f32>(col.rgb * a, a);
}
