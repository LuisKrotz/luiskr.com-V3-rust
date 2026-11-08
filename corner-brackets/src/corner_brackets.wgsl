// corner_brackets.wgsl — L-brackets at a button's TOP-LEFT and
// BOTTOM-RIGHT corners.
//
// Ports the ::before/::after corner-bracket expansion on
// `.pref-done-btn` / `.awards-selected-action` /
// `.awards-explore-action` (preferences/_footer.scss,
// awards-carousel/_selected.scss, _explore.scss):
//
//   rest   — each corner is a square box ($space-md), so each arm's
//            length = arm_rest (the stub).
//   hover  — the box grows to calc(100% − inset) in BOTH axes, so the
//            TL bracket's top arm spans the whole top edge and its
//            left arm the whole left edge (the BR bracket mirrors);
//            arm_hover carries the resolved device-px spans pushed by
//            JS, over 0.25s — the JS side drives the eased `hover`
//            scalar so the shader stays stateless.
//   tint   — the border color is uniform per surface (edge color);
//            `accent` carries the hover tint for surfaces that use it
//            (the CSS variants animate border-color on some buttons).
//
// The L shape: each bracket owns a horizontal arm (the box's top or
// bottom border) and a vertical arm (its left or right border). A
// fragment is lit inside ANY arm band — the mask is a union of 4
// axis-aligned slabs, min-distance, antialiased ~1px.
//
// The vertex stage is a CUSTOM fullscreen triangle that also projects
// device-px frag coords (the arm math is pixel-native), so this file
// composes with the gpu-mini helpers prelude, not the shared vs_main.
//
// Uniform layout (f32s=16 — lib.rs doc):
//   [0,1] resolution_dev · [2] hover_amount · [3] hairline_half ·
//   [4,5] arm_rest (w,h) · [6,7] arm_hover (w,h) ·
//   [8..12) edge_rgba · [12..16) accent_rgba

struct Uniforms {
    resolution_dev: vec2<f32>,
    hover_amount: f32,
    hairline_half: f32,
    arm_rest: vec2<f32>,
    arm_hover: vec2<f32>,
    edge_rgba: vec4<f32>,
    accent_rgba: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// ── Tunables ─────────────────────────────────────────────────────────

// Band thickness = hairline half-width × 2 (both arms).
const BAND_THICKNESS_MULT: f32 = 2.0;
// The arm-mask AA — a ~1px soft edge so the brackets don't shimmer.
const ARM_AA_PX: f32 = 1.0;
// Pixels this faint are culled before the tint mix.
const LIVE_EPSILON: f32 = 0.001;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) pixel_px: vec2<f32>,   // device-px position
};

/// `vs_main` — fullscreen triangle: 3 verts → clip positions
/// (−1,−1) (3,−1) (−1,3); `pixel_px` carries the device-px position
/// (the triangle's clip coords un-projected into the canvas domain —
/// off-canvas verts land outside 0..res and are clipped by the
/// rasterizer, which is exactly what the fullscreen triangle wants).
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var output: VertexOut;
    var clip_xy = vec2<f32>(-1.0, -1.0);

    if (vertex_index == 1u) {
        clip_xy = vec2<f32>(3.0, -1.0);
    } else if (vertex_index == 2u) {
        clip_xy = vec2<f32>(-1.0, 3.0);
    }

    output.position = vec4<f32>(clip_xy, 0.0, 1.0);
    output.pixel_px = vec2<f32>(
        (clip_xy.x + 1.0) * 0.5 * uniforms.resolution_dev.x,
        (1.0 - clip_xy.y) * 0.5 * uniforms.resolution_dev.y,
    );

    return output;
}

/// `slab_distance` — distance outside an axis-aligned band: how far
/// `point` is from the rectangle [lo.x,hi.x]×[lo.y,hi.y]. `max`-of-
/// axis-excess collapses to ≤0 inside — the arm union is `min` over
/// the 4 slabs.
fn slab_distance(point: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>) -> f32 {
    let excess = max(lo - point, point - hi);

    return max(excess.x, excess.y);
}

/// `fs_main` — the 4-slab union × tint, ~1px-antialiased.
@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    let pixel = input.pixel_px;
    let arm_w = mix(uniforms.arm_rest.x, uniforms.arm_hover.x, uniforms.hover_amount); // box width (arm x)
    let arm_h = mix(uniforms.arm_rest.y, uniforms.arm_hover.y, uniforms.hover_amount); // box height (arm y)
    let band = uniforms.hairline_half * BAND_THICKNESS_MULT;                            // band thickness

    // The 4 arms — TL bracket (border-top + border-left) and BR
    // bracket (border-bottom + border-right), flush with the rect
    // edges exactly like the pseudo-elements' `top/left/bottom/right:0`:
    //
    //   TL: top band    [0,arm_w]·[0,band]  +  left band  [0,band]·[0,arm_h]
    //   BR: bottom band [w−arm_w,w]·[h−band,h] + right band [w−band,w]·[h−arm_h,h]
    let width = uniforms.resolution_dev.x;
    let height = uniforms.resolution_dev.y;

    var dist =    slab_distance(pixel, vec2<f32>(0.0,        0.0),          vec2<f32>(arm_w, band));
    dist = min(dist, slab_distance(pixel, vec2<f32>(0.0,        0.0),          vec2<f32>(band, arm_h)));
    dist = min(dist, slab_distance(pixel, vec2<f32>(width - arm_w, height - band), vec2<f32>(width, height)));
    dist = min(dist, slab_distance(pixel, vec2<f32>(width - band,  height - arm_h), vec2<f32>(width, height)));

    let arm_mask = 1.0 - smoothstep(-ARM_AA_PX, ARM_AA_PX, dist);

    if (arm_mask <= LIVE_EPSILON) {
        discard;
    }

    let tint = mix(uniforms.edge_rgba, uniforms.accent_rgba, uniforms.hover_amount);
    let alpha = tint.a * arm_mask;

    return vec4<f32>(tint.rgb * alpha, alpha);
}
