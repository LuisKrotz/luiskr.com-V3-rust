// frame_uniform.wgsl — the shared scene3d uniform block, material
// bindings, and the helpers every scene3d pipeline reuses.
//
// This file is NOT a standalone shader: renderer.rs prepends it to
// each shader_*.wgsl at compile time (`concat!` + `include_str!`), so
// every pipeline sees ONE FrameUniforms declaration. That is what
// makes the group(0) bind group interchangeable across pipelines —
// the layouts match field-for-field by construction, not by
// convention.
//
// # Uniform block layout (mirrors renderer.rs FRAME_F32S, byte-exact)
//
//   f32 0-15   view_proj  — proj·view, multiplies on the left.
//   f32 16-18  camera_pos — world-space eye.
//   f32 19     time       — seconds since mount (shader clocks).
//   f32 20-23  _pad0
//   f32 24-26  light_pos  — the scene point light.
//   f32 27     _pad1
//   f32 28-30  light_rgb  — light color.
//   f32 31     _pad2
//   f32 32     light_int  — intensity.
//   f32 33-35  _pad3      — three scalars, NOT a vec3: a vec3 here
//              would 16-align to byte 144 and shift every field after
//              it 4 f32s late (canvas→40, model→44 — the renderer
//              writes model at byte 192). Scalar pads sit at f32
//              33/34/35 exactly, keeping canvas at 36 and model at 48.
//   f32 36-39  canvas     — (w, h, 1/h, dpr) in device px.
//   f32 40-47  _pad4/_pad5 — model must start at f32 48 (renderer.rs
//              writes the draw block at byte offset 192).
//   f32 48-63  model      — the node's world matrix.
//   f32 64-67  mat_color  — material base color rgba.
//   f32 68-71  mat_emit   — emissive additive tint.
//   f32 72-75  mat_params — kind-specific (roughness/metalness,
//              point half-size, …; each shader documents its usage).
//   f32 76-79  mat_flags  — (tex_id, side, blending, transparent).

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    _pad0: vec4<f32>,
    light_pos: vec3<f32>,
    _pad1: f32,
    light_rgb: vec3<f32>,
    _pad2: f32,
    light_int: f32,
    _pad3a: f32,
    _pad3b: f32,
    _pad3c: f32,
    canvas: vec4<f32>,
    _pad4: vec4<f32>,
    _pad5: vec4<f32>,
    model: mat4x4<f32>,
    mat_color: vec4<f32>,
    mat_emit: vec4<f32>,
    mat_params: vec4<f32>,
    mat_flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame_uniforms: FrameUniforms;
@group(1) @binding(0) var material_tex: texture_2d<f32>;
@group(1) @binding(1) var material_sampler: sampler;

/// `CameraBasis` — the screen-facing orthonormal frame rebuilt from
/// (camera → billboard center): `right`/`up` in world space. Both the
/// sprite and points pipelines billboard in world space with this
/// basis instead of touching clip space (see shader_sprite.wgsl's doc
/// for why clip-space tricks lose to a rebuilt basis).
struct CameraBasis {
    right: vec3<f32>,
    up: vec3<f32>,
};

/// `camera_facing_basis` — fwd = normalize(camera_pos − center), then
/// right ⊥ (world_up, fwd) and up completes the frame. The gimbal
/// guard substitutes +Z as the up hint when fwd ∥ world_up (cross
/// product → 0). Cost: one normalize + two crosses — trivial at the
/// 4-verts-per-billboard scale these pipelines run.
fn camera_facing_basis(center: vec3<f32>) -> CameraBasis {
    let forward = normalize(frame_uniforms.camera_pos - center);
    var up_hint = vec3<f32>(0.0, 1.0, 0.0);

    // Gimbal guard: forward ∥ up_hint → cross → 0 → rebase on +Z.
    if (abs(dot(forward, up_hint)) > 0.99) {
        up_hint = vec3<f32>(0.0, 0.0, 1.0);
    }

    var basis: CameraBasis;
    basis.right = normalize(cross(up_hint, forward));
    basis.up = cross(forward, basis.right);

    return basis;
}

/// `unlit_composite` — the MeshBasicMaterial fragment composite:
/// texel·tint + emissive, alpha preserved. basic, instanced, and
/// sprite share it verbatim — one definition here.
fn unlit_composite(texel: vec4<f32>) -> vec4<f32> {
    return texel * frame_uniforms.mat_color + frame_uniforms.mat_emit;
}
