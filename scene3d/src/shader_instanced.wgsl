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
//   world = model · instance · position
// `model` is the InstancedMesh node's own transform (usually identity
// — instances are already world-space in star-field), `instance` is
// the per-instance TRS written by scene.set_instance().
//
// Fragment: same unlit composite as shader_basic — instanced glows
// are additive-blended, which the pipeline key already encodes.
//
// Shared prelude: FrameUniforms, material bindings, and
// unlit_composite come from frame_uniform.wgsl.

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_uv: vec2<f32>,
    // Instance matrix columns — location 3-6, step=instance.
    @location(3) instance_col_0: vec4<f32>,
    @location(4) instance_col_1: vec4<f32>,
    @location(5) instance_col_2: vec4<f32>,
    @location(6) instance_col_3: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_uv: vec2<f32>,
};

/// `vs_main` — rebuild the instance matrix from its four column
/// attributes and apply the standard chain: model·instance·position →
/// view_proj. Mat4 storage is column-major (math.rs's .m layout —
/// m[0..4] is column 0), matching WGSL's mat4x4 constructor order.
@vertex
fn vs_main(vert: VertexIn) -> VertexOut {
    var out: VertexOut;

    // Rebuild the per-instance world matrix from its four column
    // attributes (see header for why it's not a mat4x4 input).
    let instance_mat = mat4x4<f32>(
        vert.instance_col_0,
        vert.instance_col_1,
        vert.instance_col_2,
        vert.instance_col_3,
    );

    // object → instance space → node space → world: model applies
    // the InstancedMesh node's own transform around the instances.
    let world_pos = frame_uniforms.model * instance_mat * vec4<f32>(vert.position, 1.0);

    // world → clip through the shared camera block.
    out.clip_position = frame_uniforms.view_proj * world_pos;

    // uv passes through — the fragment samples the shared atlas.
    out.tex_uv = vert.tex_uv;

    return out;
}

/// `fs_main` — unlit texel·color + emissive, identical to basic:
/// instancing is a vertex-stage concern only.
@fragment
fn fs_main(frag: VertexOut) -> @location(0) vec4<f32> {
    // Sample the instance atlas texel.
    let texel = textureSample(material_tex, material_sampler, frag.tex_uv);

    return unlit_composite(texel);
}
