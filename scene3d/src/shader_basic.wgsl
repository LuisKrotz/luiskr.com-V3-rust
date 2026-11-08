// shader_basic.wgsl — unlit mesh pipeline (three.js MeshBasicMaterial).
//
// Vertex: position → model → view_proj, passing normal/uv through for
// the fragment stage. Fragment: mat_color · texture + emissive — no
// lighting terms, matching MeshBasicMaterial semantics (sky domes,
// glow shells, flat-colored geometry).
//
// The FrameUniforms block, material bindings, and unlit_composite
// come from frame_uniform.wgsl — renderer.rs concatenates that shared
// prelude ahead of this source so the group(0) bind group is
// interchangeable across all five scene3d pipelines by construction.

/// `VertexIn` — the 32B interleaved stream from renderer.rs:
/// position(0) | normal(1) | uv(2).
struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_uv: vec2<f32>,
};

/// `VertexOut` — the varying payload the rasterizer interpolates:
/// clip position plus the fragment inputs.
struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
};

/// `vs_main` — one matrix chain: world = model·position, clip =
/// view_proj·world. The normal transforms by model's upper 3×3 (valid
//  for the uniform/rigid scales these scenes use — non-uniform scale
//  would need the inverse-transpose; star-field/earth scales are
//  uniform, documented in scene.rs).
@vertex
fn vs_main(vert: VertexIn) -> VertexOut {
    var out: VertexOut;

    // Object → world: w=1 marks a point (translation applies).
    let world_pos = frame_uniforms.model * vec4<f32>(vert.position, 1.0);

    // world → clip: the camera's composed proj·view.
    out.clip_position = frame_uniforms.view_proj * world_pos;

    // uv passes through untouched — no per-vertex math needed.
    out.tex_uv = vert.tex_uv;

    // Normal → world: w=0 marks a direction (translation skipped),
    // renormalized because scale denormalizes the attribute.
    out.normal = normalize((frame_uniforms.model * vec4<f32>(vert.normal, 0.0)).xyz);

    return out;
}

/// `fs_main` — unlit composite: texel · color + emissive. Untextured
/// materials bind the 1×1 white texel so the multiply is unconditional
/// (no pipeline fork by texture presence — the registry guarantees a
/// valid view at slot 0 for every material).
@fragment
fn fs_main(frag: VertexOut) -> @location(0) vec4<f32> {
    // Sample the material's texture (or the 1×1 white fallback).
    let texel = textureSample(material_tex, material_sampler, frag.tex_uv);

    return unlit_composite(texel);
}
