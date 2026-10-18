// burn.wgsl — instanced tinted quads for the mosaic letter-burn sim.
//
// One draw call paints every live particle. Instances arrive in a
// STORAGE BUFFER (not a vertex buffer): the vertex stage indexes them
// by @builtin(instance_index) — this keeps the Rust side at a single
// `write_buffer` per frame with no vertex-layout descriptors to keep
// in sync across the JS/WGSL/Rust boundary.
//
// Instance record — 4 × vec4 = 64 B per particle:
//   [0] pos.xy, size.xy    CSS px (origin top-left, +y down)
//   [1] uv.xy, uv_size.xy  reserved (atlas-less today — all particles
//                          are flat tints; kept for future glyph/
//                          deposit quads)
//   [2] tint.rgba          straight-alpha multiplier
//   [3] z, pad, pad, pad   painter's z (CPU-sorted already; unused in
//                          shader, kept for record-shape parity)
//
// Bindings: 0 uniform viewport (vec2 css px) · 1 storage instances.

struct Uniforms {
    viewport_px: vec2<f32>,
    pad: vec2<f32>,
}

struct Instances {
    data: array<vec4<f32>>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> inst: Instances;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) tint: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    // Unit quad corners in [0,1]² — 6 verts, triangle list, no VBO.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );

    let c = corners[vid];
    let r0 = inst.data[iid * 4u];
    let pos = r0.xy;
    let size = r0.zw;
    let px = pos + c * size;

    // CSS px → NDC: [0,w] → [-1,1] x, [0,h] → [1,-1] y (clip y is up).
    let ndc = vec2<f32>(
        px.x / u.viewport_px.x * 2.0 - 1.0,
        1.0 - px.y / u.viewport_px.y * 2.0,
    );

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.tint = inst.data[iid * 4u + 2u];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Flat tinted quad — particles have no texture. Premultiplied
    // output (the pipeline blends premultiplied-over).
    return vec4<f32>(in.tint.rgb * in.tint.a, in.tint.a);
}
