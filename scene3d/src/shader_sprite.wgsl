// shader_sprite.wgsl — camera-facing billboard (three.js Sprite).
//
// A sprite is a quad that always faces the camera — the classic
// implementation expands the quad in VIEW space, which is exactly
// what this shader does, with the basis rebuilt in world space
// instead of touching clip space:
//
//   forward = normalize(camera_pos − center)
//   right   = normalize(cross(world_up, forward))
//   up      = cross(forward, right)
//   world_pos = center + (right·corner.x + up·corner.y) · size
//
// This is the CPU-three.js formulation moved to the vertex shader —
// one cross-product basis per corner (4×/sprite, trivial), exact for
// any projection, no matrix inversion, no clip-space fudge factor.
// Degenerate case (forward ∥ world_up): cross product → 0 — the
// shared `camera_facing_basis` helper substitutes +Z as the up hint,
// the standard gimbal guard.
//
// `size` comes from the model matrix's scale (three.js Sprite
// semantics — scale.x/y are the sprite's world-space width/height).
// We extract it as the column lengths of model's upper 3×3, which is
// exact for uniform scale and a documented approximation otherwise.
//
// Shared prelude: FrameUniforms, material bindings,
// camera_facing_basis, and unlit_composite come from
// frame_uniform.wgsl.

struct VertexIn {
    /// Unit-quad corner (−0.5..0.5).
    @location(0) position: vec3<f32>,
    /// Unused — sprites are unlit.
    @location(1) normal: vec3<f32>,
    @location(2) tex_uv: vec2<f32>,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_uv: vec2<f32>,
};

/// `vs_main` — world-space billboard via the shared rebuilt camera
/// basis (see the module doc for why this beats clip-space tricks).
/// The model matrix's translation is the sprite center; its column
/// lengths are the world-space quad size (half-extents: the quad
/// spans ±0.5).
@vertex
fn vs_main(vert: VertexIn) -> VertexOut {
    var out: VertexOut;

    let center = frame_uniforms.model[3].xyz;

    // Model scale → world-space quad size. Column lengths of the
    // upper 3×3: |m[:,0]| = scale.x, |m[:,1]| = scale.y.
    let scale_x = length(frame_uniforms.model[0].xyz);
    let scale_y = length(frame_uniforms.model[1].xyz);

    // Screen-facing basis — rebuilt from (camera → center), gimbal-
    // guarded inside the shared helper.
    let basis = camera_facing_basis(center);

    // Expand the unit-quad corner in the camera basis, scaled by the
    // model's extracted half-extents.
    let world_pos = center + (basis.right * vert.position.x * scale_x) + (basis.up * vert.position.y * scale_y);

    // Project the billboard corner.
    out.clip_position = frame_uniforms.view_proj * vec4<f32>(world_pos, 1.0);

    // uv passes through for the sprite texture sample.
    out.tex_uv = vert.tex_uv;

    return out;
}

/// `fs_main` — sprite composite: texel·color + emissive, same as
/// basic (sprites are unlit quads — glows, labels, flares).
@fragment
fn fs_main(frag: VertexOut) -> @location(0) vec4<f32> {
    // Sample the sprite texture.
    let texel = textureSample(material_tex, material_sampler, frag.tex_uv);

    return unlit_composite(texel);
}
