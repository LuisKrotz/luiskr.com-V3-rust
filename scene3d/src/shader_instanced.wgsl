// shader_instanced.wgsl — instanced mesh pipeline (three.js
// InstancedMesh with MeshBasicMaterial — the star-field use case:
// thousands of identical star quads with per-instance transforms).
//
// Vertex buffer slot 1 carries a per-instance mat4x4 as four
// step=instance float32x4 attributes at locations 3-6 (the web-sys
// pipeline builder in renderer.rs declares them as four vec4 params —
// WGSL validates a mat4x4 vertex input inconsistently across drivers,
// so the mat is rebuilt in-shader from its column vecs).
//
// Transform order (three.js semantics):
//   world = model · instance · pos
// `model` is the InstancedMesh node's own transform (usually identity
// — instances are already world-space in star-field), `instance` is
// the per-instance TRS written by scene.set_instance().
//
// Fragment: same unlit composite as shader_basic — instanced glows
// are additive-blended, which the pipeline key already encodes.

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

struct VertexIn {
    @location(0) pos : vec3<f32>,
    @location(1) nrm : vec3<f32>,
    @location(2) uv  : vec2<f32>,
    // Instance matrix columns — location 3-6, step=instance.
    @location(3) i0 : vec4<f32>,
    @location(4) i1 : vec4<f32>,
    @location(5) i2 : vec4<f32>,
    @location(6) i3 : vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) uv         : vec2<f32>,
};

/// `vs_main` — rebuild the instance matrix from its four column
/// attributes and apply the standard chain: model·instance·pos →
/// view_proj. Mat4 storage is column-major (math.rs's .m layout —
/// m[0..4] is column 0), matching WGSL's mat4x4 constructor order.
@vertex
fn vs_main(v : VertexIn) -> VertexOut {
    var out : VertexOut;

    let inst = mat4x4<f32>(v.i0, v.i1, v.i2, v.i3);
    let world = frame.model * inst * vec4<f32>(v.pos, 1.0);

    out.clip = frame.view_proj * world;
    out.uv   = v.uv;

    return out;
}

/// `fs_main` — unlit texel·color + emissive, identical to basic:
/// instancing is a vertex-stage concern only.
@fragment
fn fs_main(v : VertexOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, smp, v.uv);

    return texel * frame.mat_color + frame.mat_emit;
}
