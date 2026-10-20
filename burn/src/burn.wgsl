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
//   [1] shape             FOR EMBER/FLECK particles: all zero → hard
//                          quad. FOR SOFT particles (sim.rs kind 2/3):
//                          x = 1.0 (billow flag), y = rotation rad,
//                          z = noise seed, w = variant —
//                          0 = smoke puff (wide feather, low-freq
//                          billow), 1 = flame lick (tight hot core,
//                          high-frequency licking erosion).
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
    /// Unit-quad coordinate (0..1) — the soft path's radial space.
    @location(1) local: vec2<f32>,
    /// Shape record [1]: x=billow flag, y=rotation, z=noise seed,
    /// w=variant (0 smoke / 1 flame lick).
    @location(2) shape: vec4<f32>,
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
    out.local = c;
    out.shape = inst.data[iid * 4u + 1u];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Flat-tinted hard quad — embers and char flecks (shape.x = 0).
    if (in.shape.x < 0.5) {
        return vec4<f32>(in.tint.rgb * in.tint.a, in.tint.a);
    }

    // ── Soft particle — smoke puff (shape.w=0) or flame lick (w=1) ──
    // A soft radial disc eroded by layered sine noise — cheap stand-in
    // for a turbulence texture: detuned spatial frequencies cut the
    // smooth rim into billows, and the sim's spin rotates the sampling
    // frame so the erosion crawls instead of sitting frozen.
    // Centered coords scaled to ±1:
    var q = (in.local - vec2<f32>(0.5, 0.5)) * 2.0;

    // Rotate the sample frame by the puff's spin angle — the noise
    // field turns with the plume, so billows drift around the rim.
    let cs = cos(in.shape.y);
    let sn = sin(in.shape.y);

    q = vec2<f32>(q.x * cs - q.y * sn, q.x * sn + q.y * cs);

    let seed = in.shape.z;
    let flame = in.shape.w > 0.5;

    // Rim erosion — three sines at incommensurate frequencies.
    // Flame doubles the spatial frequency and amplitudes: a lick's
    // edge flickers hard (turbulent mixing at the flame sheet), smoke
    // only billows.
    let freq = select(1.0, 2.1, flame);
    let amp = select(1.0, 1.9, flame);
    let n = (sin(q.x * 5.0 * freq + seed * 7.0) * 0.12
          + sin(q.y * 4.0 * freq - seed * 3.0) * 0.10
          + sin((q.x + q.y) * 8.0 * freq + seed * 13.0) * 0.06) * amp;
    let r = length(q) + n;

    // Body falloff — smoke is a wide feathered gas (0.45→1.0R); a
    // flame lick keeps a tight hot core (0.2→0.8R) so the tongue reads
    // dense at its root and necks off toward the tip.
    let puff = 1.0 - smoothstep(
        select(0.45, 0.20, flame),
        select(1.0, 0.82, flame),
        r,
    );

    // Inner density variation — real plumes and flames aren't uniform
    // puffs: crossed low-frequency waves break the body into drifting
    // lobes. Flames get a wider amplitude swing — the lick's interior
    // visibly roils.
    let body = select(0.72, 0.58, flame)
        + select(0.28, 0.42, flame)
            * sin(q.x * 3.0 * freq + seed * 5.0)
            * sin(q.y * 2.5 * freq - seed * 9.0);

    let a = in.tint.a * puff * body;

    // Premultiplied output (the pipeline blends premultiplied-over).
    return vec4<f32>(in.tint.rgb * a, a);
}
