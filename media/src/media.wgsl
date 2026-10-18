// media.wgsl — textured-quad shader for decoded images + video frames.
//
// The simplest widget: a 6-vertex quad (two triangles) whose corners
// are the unit square — the vertex shader places it over the widget's
// rect, the fragment shader samples the texture. Frame swaps are
// `write_texture` calls CPU-side — this shader never changes.
//
// Uniform layout (f32s = 12 — 3×vec4):
//   [0..3]  rect      x, y, w, h — widget rect, canvas DEVICE px
//   [4..7]  uv_rect   u, v, du, dv — cover-fit crop window (JS computes
//                     the crop so the image fills the rect preserving
//                     aspect — "cover" = scale to cover, crop overflow)
//   [8,9]   viewport  full canvas CSS px — for the NDC projection below
//   [10,11] pad       16B alignment pad
//
// Bindings (must match media.rs::make_slot):
//   0 uniform, 1 texture (RGBA8), 2 sampler (linear, clamp-to-edge).

struct Uniforms {
    rect: vec4<f32>,      // x, y, w, h — canvas CSS px
    uv_rect: vec4<f32>,   // u, v, du, dv — cover-fit crop
    viewport: vec2<f32>,  // canvas CSS px
    pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var media_tex: texture_2d<f32>;
@group(0) @binding(2) var media_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// ── The CSS-px → clip-space projection ──────────────────────────────
//
// Every OTHER widget uses a fullscreen triangle + scissor (their math
// is rect-relative). Media content maps vertex→texture directly, so a
// real quad is more natural. The projection math:
//   px = rect.xy + corner × rect.wh   → CSS-px position of this vertex
//   ndc.x = px.x/vp.x × 2 − 1         → [0,vp] → [-1,+1] (clip x)
//   ndc.y = 1 − px.y/vp.y × 2         → same but FLIPPED: CSS y is down,
//                                       NDC y is up — the flip is why
//                                       screen y=0 lands at ndc +1.
// No matrices needed — it's an orthographic 2D projection done by hand,
// one multiply-add per axis. (A traditional app would build an ortho
// matrix; a 2D UI never needs the generality.)
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Two triangles: (0,0)(1,0)(0,1) + (1,0)(1,1)(0,1) — the unit quad.
    // Six indices = 2 draws' worth of corners; vertex_index picks.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );

    let c = corners[vid];
    let px = u.rect.xy + c * u.rect.zw;
    let ndc = vec2<f32>(
        px.x / u.viewport.x * 2.0 - 1.0,
        1.0 - px.y / u.viewport.y * 2.0,
    );

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    // UV corner in the crop window — c∈{0,1}² picks uv or uv+du/dv.
    // uv_rect is precomputed JS-side for cover-fit; this shader stays
    // ignorant of aspect math (single responsibility).
    out.uv = u.uv_rect.xy + c * u.uv_rect.zw;
    return out;
}

// One bilinear sample — the linear sampler gives free AA at fractional
// scales; a 200px widget drawing a 4K source aliases badly without
// mips (see media.rs's note on why we accept that trade).
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(media_tex, media_samp, in.uv);
}
