// media.wgsl — textured-quad shader for decoded images + video frames.
//
// The simplest widget: a 6-vertex quad (two triangles) whose corners
// are the unit square — the vertex shader places it over the widget's
// rect, the fragment shader samples the texture. Frame swaps are
// `write_texture` calls CPU-side — this shader never changes.
//
// Keeps its OWN vertex stage (a real quad, not the fullscreen
// triangle), so it composes with the gpu-mini helpers prelude —
// `quad_corner` supplies the two-triangle corner table.
//
// Uniform layout (f32s = 12 — 3×vec4):
//   [0..3]  rect_css     x, y, w, h — widget rect, canvas CSS px
//                        (the projection is a ratio — any consistent
//                        unit works; CSS px keeps the pipeline in DOM
//                        units with zero DPR math)
//   [4..7]  uv_rect      u, v, du, dv — cover-fit crop window (JS
//                        computes the crop so the image fills the
//                        rect preserving aspect — "cover" = scale to
//                        cover, crop overflow)
//   [8,9]   viewport_css full canvas CSS px — for the NDC projection
//   [10,11] reserved     16B alignment pad
//
// Bindings (must match media.rs::make_slot):
//   0 uniform, 1 media_tex (RGBA8), 2 media_samp (linear, clamp-edge).

struct Uniforms {
    rect_css: vec4<f32>,     // x, y, w, h — canvas CSS px
    uv_rect: vec4<f32>,      // u, v, du, dv — cover-fit crop
    viewport_css: vec2<f32>, // canvas CSS px
    reserved: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var media_tex: texture_2d<f32>;
@group(0) @binding(2) var media_samp: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_uv: vec2<f32>,
}

// ── The device-px → clip-space projection ───────────────────────────
//
// Every OTHER widget uses a fullscreen triangle + scissor (their math
// is rect-relative). Media content maps vertex→texture directly, so a
// real quad is more natural. The projection math:
//   px = rect.xy + corner × rect.wh   → device-px position of this vertex
//   ndc.x = px.x/vp.x × 2 − 1         → [0,vp] → [-1,+1] (clip x)
//   ndc.y = 1 − px.y/vp.y × 2         → same but FLIPPED: screen y is
//                                       down, NDC y is up — the flip
//                                       is why screen y=0 lands at ndc +1.
// No matrices needed — it's an orthographic 2D projection done by hand,
// one multiply-add per axis. (A traditional app would build an ortho
// matrix; a 2D UI never needs the generality.)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let corner_uv = quad_corner(vertex_index);
    let pixel_px = uniforms.rect_css.xy + corner_uv * uniforms.rect_css.zw;
    let clip_xy = vec2<f32>(
        pixel_px.x / uniforms.viewport_css.x * 2.0 - 1.0,
        1.0 - pixel_px.y / uniforms.viewport_css.y * 2.0,
    );

    var output: VertexOut;
    output.position = vec4<f32>(clip_xy, 0.0, 1.0);
    // UV corner in the crop window — corner_uv ∈ {0,1}² picks uv or
    // uv+du/dv. uv_rect is precomputed JS-side for cover-fit; this
    // shader stays ignorant of aspect math (single responsibility).
    output.tex_uv = uniforms.uv_rect.xy + corner_uv * uniforms.uv_rect.zw;
    return output;
}

// One bilinear sample — the linear sampler gives free AA at fractional
// scales; a 200px widget drawing a 4K source aliases badly without
// mips (see media.rs's note on why we accept that trade).
@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(media_tex, media_samp, input.tex_uv);
}
