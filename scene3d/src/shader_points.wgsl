// shader_points.wgsl — point-sprite pipeline (three.js PointsMaterial).
//
// WebGPU has no gl_PointSize — point primitives rasterize as 1px.
// The renderer instead expands every point into a 4-vertex billboard
// quad at upload time (`expand_points` in renderer.rs): the vertex
// stream's `position` slot carries the shared point CENTER and `corner`
// carries the quad CORNER (−1..1).
//
// Expansion happens in world space via the shared camera-facing basis
// (frame_uniform.wgsl's `camera_facing_basis` — the same rebuilt-basis
// formulation the sprite shader uses): center → world → basis-offset
// by corner·half_size → project. Per-vertex cost is tiny (4
// verts/point, no fragment overdraw beyond the quad).
//
// Point size contract (mat_params.x): world-space half-size — the
// quad's world radius. dpr compensation (canvas.w field) is for
// CSS-specified sizes; world-space sizes scale correctly through the
// projection automatically — no dpr term needed.
//
// Circular falloff: the fragment stage discards fragments outside the
// unit circle (corner length > 1) — square points look wrong in space
// scenes — and applies a soft edge (smoothstep) for anti-aliasing
// without MSAA-dependent alpha-to-coverage.
//
// Shared prelude: FrameUniforms, material bindings, and
// camera_facing_basis come from frame_uniform.wgsl.

/// Edge-AA start radius — alpha ramps 1→0 from 80% to 100% of the
/// quad radius: wide enough to hide the quad edge at point sizes
/// ≥4px, narrow enough that small points stay solid.
const EDGE_FALLOFF_START = 0.8;

struct VertexIn {
    /// Point CENTER (shared by all 4 quad verts).
    @location(0) position: vec3<f32>,
    /// Unused — points are unlit.
    @location(1) normal: vec3<f32>,
    /// Quad corner (−1..1), packed into the uv attribute slot.
    @location(2) corner: vec2<f32>,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) corner: vec2<f32>,
};

/// `vs_main` — billboard expansion via the shared rebuilt-basis trick
/// (see shader_sprite.wgsl's doc for why world-space beats clip-space
/// tricks): center → world → basis-offset by corner·half_size →
/// project.
@vertex
fn vs_main(vert: VertexIn) -> VertexOut {
    var out: VertexOut;

    // The shared point center in world space — all 4 corners of the
    // quad start from the same position.
    let center = (frame_uniforms.model * vec4<f32>(vert.position, 1.0)).xyz;

    // World-space half-size from the material params row.
    let half_size = frame_uniforms.mat_params.x;

    // Screen-facing orthonormal basis — rebuilt from (camera →
    // center) so the quad stares back at the eye (gimbal-guarded
    // inside the shared helper).
    let basis = camera_facing_basis(center);

    // Push this corner outward in the camera basis — the quad is
    // screen-aligned by construction, sized by `half_size` world units.
    let world_pos = center + (basis.right * vert.corner.x + basis.up * vert.corner.y) * half_size;

    // Project the expanded corner to clip space.
    out.clip_position = frame_uniforms.view_proj * vec4<f32>(world_pos, 1.0);

    // Corner passes through — the fragment uses it for the circle
    // falloff and uv reconstruction.
    out.corner = vert.corner;

    return out;
}

/// `fs_main` — circular point sprite: discard outside the unit
/// circle, smoothstep edge AA, texel·color + emissive composite.
/// The discard is a real `discard` (not alpha 0) — additive point
/// clouds must not write transparent depth.
@fragment
fn fs_main(frag: VertexOut) -> @location(0) vec4<f32> {
    // Sample BEFORE the discard — implicit-derivative calls must sit
    // in unconditionally-uniform control flow; a post-discard call is
    // legal on most validators but demote-after-sample ordering is
    // universally safe and identical in output (the discarded
    // fragment never presents).
    let radius = length(frag.corner);
    let texel = textureSample(material_tex, material_sampler, frag.corner * 0.5 + vec2<f32>(0.5));

    if (radius > 1.0) {
        discard;
    }

    // Edge AA: alpha ramps 1→0 over the outer EDGE_FALLOFF_START..1
    // of the radius.
    let alpha = 1.0 - smoothstep(EDGE_FALLOFF_START, 1.0, radius);

    return unlit_composite(texel) * alpha;
}
