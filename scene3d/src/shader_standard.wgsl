// shader_standard.wgsl — lit mesh pipeline (three.js
// MeshStandardMaterial reduced to Blinn-Phong: the scenes use one
// point light + roughness/metalness, so a full PBR BRDF would buy
// nothing measurable — the approximation is the standard one:
//
//   diffuse  = color · (N·L)⁺            — Lambert term.
//   specular = light · (N·H)⁺^shininess  — Blinn half-vector term.
//   shininess = (1−roughness)·128 + 2    — maps roughness to exponent:
//             roughness 1 → ~2 (broad, matte), 0 → ~130 (tight glint).
//   metalness scales the specular tint toward the base color (metals
//   reflect colored light; dielectrics reflect white).
//
// Ambient is a flat 8% floor — enough to keep unlit faces readable in
// the dark-space scenes without a second light source or IBL probe.
//
// Uniform layout: identical to shader_basic.wgsl (see that file's doc
// for the byte map). light_* fields are populated here for real.

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
    mat_params: vec4<f32>,   // [0]=unused, [1]=roughness, [2]=metalness
    mat_flags : vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame : Frame;
@group(1) @binding(0) var tex : texture_2d<f32>;
@group(1) @binding(1) var smp : sampler;

struct VertexIn {
    @location(0) pos : vec3<f32>,
    @location(1) nrm : vec3<f32>,
    @location(2) uv  : vec2<f32>,
};

struct VertexOut {
    @builtin(position) clip  : vec4<f32>,
    @location(0) uv          : vec2<f32>,
    @location(1) nrm         : vec3<f32>,
    @location(2) world       : vec3<f32>,
};

/// `vs_main` — same matrix chain as basic plus a world-pos varying:
/// the fragment stage needs it for the light vector L.
@vertex
fn vs_main(v : VertexIn) -> VertexOut {
    var out : VertexOut;

    let world = frame.model * vec4<f32>(v.pos, 1.0);

    out.clip  = frame.view_proj * world;
    out.uv    = v.uv;
    out.nrm   = normalize((frame.model * vec4<f32>(v.nrm, 0.0)).xyz);
    out.world = world.xyz;

    return out;
}

/// `fs_main` — Blinn-Phong with the flat ambient floor. All vector
/// math is normalized first (interpolation denormalizes normals —
/// the classic per-fragment shading bug when skipped).
@fragment
fn fs_main(v : VertexOut) -> @location(0) vec4<f32> {
    let texel   = textureSample(tex, smp, v.uv);
    let base    = texel * frame.mat_color;
    let n       = normalize(v.nrm);
    let l       = normalize(frame.light_pos - v.world);
    let viewdir = normalize(frame.cam_pos - v.world);
    let h       = normalize(l + viewdir);

    let diff = max(dot(n, l), 0.0);
    let shin = (1.0 - frame.mat_params.y) * 128.0 + 2.0;
    let spec = pow(max(dot(n, h), 0.0), shin);

    // Metalness lerps specular tint: white (dielectric) → base (metal).
    let spec_tint = mix(vec3<f32>(1.0), base.rgb, frame.mat_params.z);

    let lit = base.rgb * (0.08 + diff * frame.light_int * frame.light_rgb)
            + spec_tint * spec * frame.light_int;

    return vec4<f32>(lit + frame.mat_emit.rgb, base.a);
}
