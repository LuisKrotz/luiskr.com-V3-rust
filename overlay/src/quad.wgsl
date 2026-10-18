// overlay quads — one instanced draw paints every quad in the frame.
// Instance data is flat CSS-px space; the vertex stage synthesizes a
// fullscreen UV-space quad (6 verts) per instance and projects CSS px →
// clip space against the viewport uniform, so no vertex buffer exists.

struct Instance {
    @location(0) pos: vec2<f32>,
    @location(1) size: vec2<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) uv_size: vec2<f32>,
    @location(4) tint: vec4<f32>,
    @location(5) z: f32,
    @location(6) group: u32,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
}

// Canvas size in CSS px — set once per resize; the projection divides by
// it, so the uniform is hoisted out of the per-vertex math.
@group(0) @binding(0) var<uniform> viewport_px: vec2<f32>;

// The shared texture atlas — quads sample their uv rect from it. One
// binding for the whole frame keeps draw state trivially small.
@group(0) @binding(1) var atlas_tex: texture_2d<f32>;
@group(0) @binding(2) var atlas_samp: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, inst: Instance) -> VsOut {
    // Unit quad corners in [0,1]² — 6 verts, triangle list.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );

    let c = corners[vid];
    let px = inst.pos + c * inst.size;

    // CSS px → NDC: [0,w] → [-1,1] x, [0,h] → [1,-1] y (clip y is up).
    let ndc = vec2<f32>(
        px.x / viewport_px.x * 2.0 - 1.0,
        1.0 - px.y / viewport_px.y * 2.0,
    );

    var out: VsOut;
    out.clip = vec4<f32>(ndc, inst.z, 1.0);
    out.uv = inst.uv + c * inst.uv_size;
    out.tint = inst.tint;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Straight-alpha tint over the atlas sample; the surface is premultiplied
    // so the pipeline blends ALPHA_BLENDING and we output straight color.
    return textureSample(atlas_tex, atlas_samp, in.uv) * in.tint;
}
