// shader_sprite.wgsl — camera-facing billboard (three.js Sprite).
//
// A sprite is a quad that always faces the camera — the classic
// implementation expands the quad in VIEW space, which is exactly
// what this shader does:
//
//   center_view = view · model · vec4(0,0,0,1)   — the sprite's world
//                 position viewed from the eye.
//   view_pos    = center_view + vec4(corner.xy · size, 0, 0)
//               — push each corner outward in VIEW space: the quad
//               can never rotate because view-space x/y is the screen
//               plane by definition.
//   clip        = proj · view_pos
//
// `size` comes from the model matrix's scale (three.js Sprite
// semantics — scale.x/y are the sprite's world-space width/height).
// We extract it as the column lengths of model's upper 3×3, which is
// exact for uniform scale and a documented approximation otherwise.
//
// `view` is the inverse of the camera transform — but the uniform
// carries only view_proj. We recover the view matrix's translation-
// removed rotation by noting the sprite center in view space is
// `view·world_pos`; rather than store a second matrix, we compute
// center_view = view_proj·world and unproject? NO — simpler: view_pos
// needs view-space coordinates, and view = proj⁻¹·view_proj. Since
// the renderer also stores cam_pos, the sprite shader uses the
// standard trick instead: world-space billboard using the camera
// right/up vectors, extracted from view_proj inverse is overkill —
// the view basis is orthonormal, so:
//
//   right_view = (1,0,0), up_view = (0,1,0) — view-space billboarding
//   needs NO basis vectors at all. We still need `view` itself to get
//   the center into view space, and view = inverse(proj)·view_proj
//   only if proj is invertible — it is for finite far planes, but
//   inverse() in WGSL is hand-rolled and 16 divisions per vertex is
//   the wrong cost.
//
// The chosen formulation: the renderer writes view_proj AND we
// bill-board in CLIP space instead — the standard clip-space trick:
//
//   clip = view_proj · world_center, then
//   clip.xy += corner.xy · size_view · clip.w
//
//   ·clip.w keeps the expansion proportional to distance (perspective
//   — a far sprite shrinks exactly like a near one at equal world
//   size). size_view = world_size / (2·tan(fov/2)·dist)·… — instead of
//   that chain we pass the world-space half-size and divide by view
//   depth: corner·size·(clip.w/dist) — and clip.w IS -view_z for a
//   perspective matrix, so:
//
//   clip.xy += corner.xy · size_world · proj_scale
//   where proj_scale = proj[0][0], proj[1][1] extracted from
//   view_proj is not separable (proj·view mixes them) — BUT view has
//   orthonormal columns, so the [0][0] and [1][1] scale survives: NO,
//   it doesn't.
//
// FINAL DESIGN (the honest one): the camera block carries cam_pos —
// the sprite builds its own view basis from (cam_pos → sprite_center):
//   fwd   = normalize(cam_pos − center)
//   right = normalize(cross(world_up, fwd))
//   up    = cross(fwd, right)
//   world_pos = center + (right·corner.x + up·corner.y) · size
// This is the CPU-three.js formulation moved to the vertex shader —
// one cross-product basis per corner (4×/sprite, trivial), exact for
// any projection, no matrix inversion, no clip-space fudge factor.
// Degenerate case (fwd ∥ world_up): cross product → 0 — the fallback
// substitutes (0,0,1) as the up hint, the standard gimbal guard.

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
    @location(0) pos : vec3<f32>,   // unit-quad corner (−0.5..0.5)
    @location(1) nrm : vec3<f32>,   // unused — sprites are unlit
    @location(2) uv  : vec2<f32>,
};

struct VertexOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) uv         : vec2<f32>,
};

/// `vs_main` — world-space billboard via a rebuilt camera basis (see
/// the module doc for why this beats clip-space tricks). The model
/// matrix's translation is the sprite center; its column lengths are
//  the world-space quad size (half-extents: the quad spans ±0.5).
@vertex
fn vs_main(v : VertexIn) -> VertexOut {
    var out : VertexOut;

    let center = frame.model[3].xyz;

    // Model scale → world-space quad size. Column lengths of the
    // upper 3×3: |m[:,0]| = scale.x, |m[:,1]| = scale.y.
    let sx = length(frame.model[0].xyz);
    let sy = length(frame.model[1].xyz);

    let fwd = normalize(frame.cam_pos - center);
    var up_hint = vec3<f32>(0.0, 1.0, 0.0);

    // Gimbal guard: fwd ∥ up_hint → cross → 0 → rebase on z.
    if (abs(dot(fwd, up_hint)) > 0.99) {
        up_hint = vec3<f32>(0.0, 0.0, 1.0);
    }

    let right = normalize(cross(up_hint, fwd));
    let up    = cross(fwd, right);

    let world = center + (right * v.pos.x * sx) + (up * v.pos.y * sy);

    out.clip = frame.view_proj * vec4<f32>(world, 1.0);
    out.uv   = v.uv;

    return out;
}

/// `fs_main` — sprite composite: texel·color + emissive, same as
/// basic (sprites are unlit quads — glows, labels, flares).
@fragment
fn fs_main(v : VertexOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, smp, v.uv);

    return texel * frame.mat_color + frame.mat_emit;
}
