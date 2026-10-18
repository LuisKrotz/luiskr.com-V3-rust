// shader_points.wgsl — point-sprite pipeline (three.js PointsMaterial).
//
// WebGPU has no gl_PointSize — point primitives rasterize as 1px.
// The renderer instead expands every point into a 4-vertex billboard
// quad at upload time (`expand_points` in renderer.rs): the vertex
// stream's `pos` slot carries the shared point CENTER and `uv`
// carries the quad CORNER (−1..1).
//
// Expansion happens in view space like the sprite shader, but cheaper:
// the point is a camera-facing screen-aligned quad, so we can add the
// corner offset AFTER view_proj by using the camera basis trick only
// once — actually simpler still: we build the same world-basis the
// sprite shader uses, since per-vertex cost is tiny (4 verts/point,
// no fragment overdraw beyond the quad).
//
// Point size contract (mat_params.x): world-space half-size — the
// quad's world radius. dpr compensation (canvas.w/dpr fields) is for
// CSS-specified sizes; world-space sizes scale correctly through the
// projection automatically — no dpr term needed.
//
// Circular falloff: the fragment stage discards fragments outside the
// unit circle (corner length > 1) — square points look wrong in space
// scenes — and applies a soft edge (smoothstep) for anti-aliasing
// without MSAA-dependent alpha-to-coverage.

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
    mat_params: vec4<f32>,   // [0] = world-space half-size
    mat_flags : vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame : Frame;
@group(1) @binding(0) var tex : texture_2d<f32>;
@group(1) @binding(1) var smp : sampler;

struct VertexIn {
    @location(0) pos    : vec3<f32>,   // point CENTER (shared by 4 verts)
    @location(1) nrm    : vec3<f32>,   // unused — points are unlit
    @location(2) corner : vec2<f32>,   // quad corner (−1..1), packed in uv
};

struct VertexOut {
    @builtin(position) clip   : vec4<f32>,
    @location(0) corner       : vec2<f32>,
};

/// `vs_main` — billboard expansion via the same rebuilt-basis trick
/// as the sprite shader (see shader_sprite.wgsl's doc): center →
/// world → basis-offset by corner·half_size → project.
@vertex
fn vs_main(v : VertexIn) -> VertexOut {
    var out : VertexOut;

    let center = (frame.model * vec4<f32>(v.pos, 1.0)).xyz;
    let size   = frame.mat_params.x;
    let fwd    = normalize(frame.cam_pos - center);
    var up_hint = vec3<f32>(0.0, 1.0, 0.0);

    if (abs(dot(fwd, up_hint)) > 0.99) {
        up_hint = vec3<f32>(0.0, 0.0, 1.0);
    }

    let right = normalize(cross(up_hint, fwd));
    let up    = cross(fwd, right);
    let world = center + (right * v.corner.x + up * v.corner.y) * size;

    out.clip   = frame.view_proj * vec4<f32>(world, 1.0);
    out.corner = v.corner;

    return out;
}

/// `fs_main` — circular point sprite: discard outside the unit
/// circle, smoothstep edge AA, texel·color + emissive composite.
/// The discard is a real `discard` (not alpha 0) — additive point
/// clouds must not write transparent depth.
@fragment
fn fs_main(v : VertexOut) -> @location(0) vec4<f32> {
    let r = length(v.corner);

    if (r > 1.0) {
        discard;
    }

    // Edge AA: alpha ramps 1→0 over the outer 20% of the radius —
    // wide enough to hide the quad edge at point sizes ≥4px, narrow
    // enough that small points stay solid.
    let alpha = 1.0 - smoothstep(0.8, 1.0, r);
    let texel = textureSample(tex, smp, v.corner * 0.5 + vec2<f32>(0.5));

    return (texel * frame.mat_color + frame.mat_emit) * alpha;
}
