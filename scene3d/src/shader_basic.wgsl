// shader_basic.wgsl — unlit mesh pipeline (three.js MeshBasicMaterial).
//
// Vertex: pos → model → view_proj, passing normal/uv/world-pos through
// for the fragment stage. Fragment: mat_color · texture + emissive —
// no lighting terms, matching MeshBasicMaterial semantics (sky domes,
// glow shells, flat-colored geometry).
//
// # Uniform block layout (mirrors renderer.rs FRAME_F32S, byte-exact)
//
//   f32 0-15   view_proj  — proj·view, multiplies on the left.
//   f32 16-18  cam_pos    — world-space eye (unused here; kept so the
//              block is ONE layout across all 5 shaders — a single
//              bind group serves every pipeline).
//   f32 19     time       — seconds since mount (shader clocks).
//   f32 20-23  _pad
//   f32 24-26  light_pos  — the scene point light (unused in basic).
//   f32 27     _pad
//   f32 28-30  light_rgb  — light color (unused in basic).
//   f32 31     _pad
//   f32 32     light_int  — intensity (unused in basic).
//   f32 36-39  canvas     — (w, h, 1/h, dpr) in device px.
//   f32 48-63  model      — the node's world matrix.
//   f32 64-67  mat_color  — material base color rgba.
//   f32 68-71  mat_emit   — emissive additive tint.
//   f32 72-75  mat_params — kind-specific (unused in basic).
//   f32 76-79  mat_flags  — (tex_id, side, blending, transparent).
//
// The struct is declared identically in every scene3d shader — the
// shared `@group(0)` bind group is interchangeable across pipelines
// only when layouts match field-for-field.

struct Frame {
    view_proj : mat4x4<f32>,
    cam_pos   : vec3<f32>,
    time      : f32,
    _pad0     : vec4<f32>,
    light_pos : vec3<f32>,
    _pad1     : f32,
    light_rgb : vec3<f32>,
    _pad2     : f32,
    light_int : f32,
    _pad3     : vec3<f32>,
    canvas    : vec4<f32>,
    model     : mat4x4<f32>,
    mat_color : vec4<f32>,
    mat_emit  : vec4<f32>,
    mat_params: vec4<f32>,
    mat_flags : vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame : Frame;
@group(1) @binding(0) var tex : texture_2d<f32>;
@group(1) @binding(1) var smp : sampler;

/// `VertexIn` — the 32B interleaved stream from renderer.rs:
/// position(0) | normal(1) | uv(2).
struct VertexIn {
    @location(0) pos : vec3<f32>,
    @location(1) nrm : vec3<f32>,
    @location(2) uv  : vec2<f32>,
};

/// `VertexOut` — the varying payload the rasterizer interpolates:
/// clip position plus the fragment inputs.
struct VertexOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) uv         : vec2<f32>,
    @location(1) nrm        : vec3<f32>,
};

/// `vs_main` — one matrix chain: world = model·pos, clip =
/// view_proj·world. The normal transforms by model's upper 3×3 (valid
//  for the uniform/rigid scales these scenes use — non-uniform scale
//  would need the inverse-transpose; star-field/earth scales are
//  uniform, documented in scene.rs).
@vertex
fn vs_main(v : VertexIn) -> VertexOut {
    var out : VertexOut;

    let world = frame.model * vec4<f32>(v.pos, 1.0);

    out.clip = frame.view_proj * world;
    out.uv   = v.uv;
    out.nrm  = normalize((frame.model * vec4<f32>(v.nrm, 0.0)).xyz);

    return out;
}

/// `fs_main` — unlit composite: texel · color + emissive. Untextured
/// materials bind the 1×1 white texel so the multiply is unconditional
/// (no pipeline fork by texture presence — the registry guarantees a
/// valid view at slot 0 for every material).
@fragment
fn fs_main(v : VertexOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, smp, v.uv);

    return texel * frame.mat_color + frame.mat_emit;
}
