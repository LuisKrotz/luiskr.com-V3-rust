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
//
// Shared prelude: quad_corner (the 6-vert unit square) comes from
// gpu-mini's helpers_source — this file owns only the burn-specific
// instanced vertex stage and the smoke/flame fragment path.

/// Soft-particle flag threshold — shape.x ≥ SOFT_FLAG reads as a
/// soft particle (smoke/flame); below it the quad is a hard ember.
const SOFT_FLAG = 0.5;

/// Flame-lick variant threshold — shape.w ≥ FLAME_VARIANT picks the
/// flame path; below it the particle is a smoke puff.
const FLAME_VARIANT = 0.5;

/// Rim-erosion spatial frequency multiplier — a flame lick's edge
/// flickers at ~2× the smoke billow frequency (turbulent mixing at
/// the flame sheet).
const FLAME_FREQ_MULT = 2.1;

/// Rim-erosion amplitude multiplier for flames — the lick's edge
/// erosion bites ~1.9× deeper than a smoke billow.
const FLAME_AMP_MULT = 1.9;

/// Smoke body feather — the puff's smoothstep window 0.45→1.0R.
const SMOKE_FALLOFF_START = 0.45;
const SMOKE_FALLOFF_END = 1.0;

/// Flame body falloff — a lick keeps a tight hot core 0.2→0.82R so
/// the tongue reads dense at its root and necks off toward the tip.
const FLAME_FALLOFF_START = 0.20;
const FLAME_FALLOFF_END = 0.82;

/// Interior density wave — base + swing per variant: real plumes and
/// flames aren't uniform puffs; crossed low-frequency waves break the
/// body into drifting lobes. Flames get the wider swing — the lick's
/// interior visibly roils.
const SMOKE_BODY_BASE = 0.72;
const FLAME_BODY_BASE = 0.58;
const SMOKE_BODY_SWING = 0.28;
const FLAME_BODY_SWING = 0.42;

struct Uniforms {
    viewport_css: vec2<f32>,
    pad: vec2<f32>,
}

struct Instances {
    records: array<vec4<f32>>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> instances: Instances;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) tint_rgba: vec4<f32>,
    /// Unit-quad coordinate (0..1) — the soft path's radial space.
    @location(1) quad_local: vec2<f32>,
    /// Shape record [1]: x=billow flag, y=rotation, z=noise seed,
    /// w=variant (0 smoke / 1 flame lick).
    @location(2) shape: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOut {
    // Unit quad corner in [0,1]² — 6 verts, triangle list, no VBO.
    let corner = quad_corner(vertex_index);
    let record = instances.records[instance_index * 4u];
    let pixel_pos = record.xy + corner * record.zw;

    // CSS px → NDC: [0,w] → [-1,1] x, [0,h] → [1,-1] y (clip y is up).
    let clip_xy = vec2<f32>(
        pixel_pos.x / uniforms.viewport_css.x * 2.0 - 1.0,
        1.0 - pixel_pos.y / uniforms.viewport_css.y * 2.0,
    );

    var out: VertexOut;
    out.position = vec4<f32>(clip_xy, 0.0, 1.0);
    out.tint_rgba = instances.records[instance_index * 4u + 2u];
    out.quad_local = corner;
    out.shape = instances.records[instance_index * 4u + 1u];
    return out;
}

@fragment
fn fs_main(frag: VertexOut) -> @location(0) vec4<f32> {
    // Flat-tinted hard quad — embers and char flecks (shape.x = 0).
    if (frag.shape.x < SOFT_FLAG) {
        return vec4<f32>(frag.tint_rgba.rgb * frag.tint_rgba.a, frag.tint_rgba.a);
    }

    // ── Soft particle — smoke puff (shape.w=0) or flame lick (w=1) ──
    // A soft radial disc eroded by layered sine noise — cheap stand-in
    // for a turbulence texture: detuned spatial frequencies cut the
    // smooth rim into billows, and the sim's spin rotates the sampling
    // frame so the erosion crawls instead of sitting frozen.
    // Centered coords scaled to ±1:
    var radial = (frag.quad_local - vec2<f32>(0.5, 0.5)) * 2.0;

    // Rotate the sample frame by the puff's spin angle — the noise
    // field turns with the plume, so billows drift around the rim.
    let spin_cos = cos(frag.shape.y);
    let spin_sin = sin(frag.shape.y);

    radial = vec2<f32>(
        radial.x * spin_cos - radial.y * spin_sin,
        radial.x * spin_sin + radial.y * spin_cos,
    );

    let noise_seed = frag.shape.z;
    let is_flame = frag.shape.w > FLAME_VARIANT;

    // Rim erosion — three sines at incommensurate frequencies.
    // Flame doubles the spatial frequency and amplitudes: a lick's
    // edge flickers hard (turbulent mixing at the flame sheet), smoke
    // only billows.
    let freq_mult = select(1.0, FLAME_FREQ_MULT, is_flame);
    let amp_mult = select(1.0, FLAME_AMP_MULT, is_flame);
    let erosion = (sin(radial.x * 5.0 * freq_mult + noise_seed * 7.0) * 0.12
        + sin(radial.y * 4.0 * freq_mult - noise_seed * 3.0) * 0.10
        + sin((radial.x + radial.y) * 8.0 * freq_mult + noise_seed * 13.0) * 0.06)
        * amp_mult;
    let radius = length(radial) + erosion;

    // Body falloff — smoke is a wide feathered gas; a flame lick keeps
    // a tight hot core so the tongue reads dense at its root and
    // necks off toward the tip.
    let body_falloff = 1.0 - smoothstep(
        select(SMOKE_FALLOFF_START, FLAME_FALLOFF_START, is_flame),
        select(SMOKE_FALLOFF_END, FLAME_FALLOFF_END, is_flame),
        radius,
    );

    // Inner density variation — crossed low-frequency waves break the
    // body into drifting lobes (see SMOKE/FLAME_BODY_* above).
    let body_density = select(SMOKE_BODY_BASE, FLAME_BODY_BASE, is_flame)
        + select(SMOKE_BODY_SWING, FLAME_BODY_SWING, is_flame)
            * sin(radial.x * 3.0 * freq_mult + noise_seed * 5.0)
            * sin(radial.y * 2.5 * freq_mult - noise_seed * 9.0);

    let alpha = frag.tint_rgba.a * body_falloff * body_density;

    // Premultiplied output (the pipeline blends premultiplied-over).
    return vec4<f32>(frag.tint_rgba.rgb * alpha, alpha);
}
