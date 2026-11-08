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
// Uniform layout: frame_uniform.wgsl (the shared prelude — see that
// file for the byte map). light_* fields are populated here for real;
// material_params = [0]=unused, [1]=roughness, [2]=metalness.

/// Ambient floor — keeps unlit faces readable without a second light
/// source or IBL probe.
const AMBIENT_FLOOR = 0.08;

/// Roughness → Blinn-Phong exponent range: roughness 1 → ~2 (broad,
/// matte), 0 → ~130 (tight glint).
const SHININESS_RANGE = 128.0;
const SHININESS_MIN = 2.0;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_uv: vec2<f32>,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
};

/// `vs_main` — same matrix chain as basic plus a world-pos varying:
/// the fragment stage needs it for the light vector L.
@vertex
fn vs_main(vert: VertexIn) -> VertexOut {
    var out: VertexOut;

    // Object → world position (w=1 — a point, translation applies).
    let world_pos = frame_uniforms.model * vec4<f32>(vert.position, 1.0);

    // world → clip through the camera's proj·view.
    out.clip_position = frame_uniforms.view_proj * world_pos;

    // uv passes through for the fragment texture sample.
    out.tex_uv = vert.tex_uv;

    // Normal → world (w=0 — a direction, translation skipped),
    // renormalized against scale distortion.
    out.normal = normalize((frame_uniforms.model * vec4<f32>(vert.normal, 0.0)).xyz);

    // World position is a real varying here — the fragment builds
    // the light/view vectors from it.
    out.world_pos = world_pos.xyz;

    return out;
}

/// `fs_main` — Blinn-Phong with the flat ambient floor. All vector
/// math is normalized first (interpolation denormalizes normals —
/// the classic per-fragment shading bug when skipped).
@fragment
fn fs_main(frag: VertexOut) -> @location(0) vec4<f32> {
    // Texel × base tint — the diffuse albedo term.
    let texel = textureSample(material_tex, material_sampler, frag.tex_uv);

    let albedo = texel * frame_uniforms.mat_color;

    // Interpolated normal renormalized (interpolation denormalizes).
    let surface_normal = normalize(frag.normal);

    // L — fragment → light, normalized.
    let light_dir = normalize(frame_uniforms.light_pos - frag.world_pos);

    // V — fragment → camera, normalized.
    let view_dir = normalize(frame_uniforms.camera_pos - frag.world_pos);

    // H — Blinn half-vector (cheaper than Phong's reflect).
    let half_dir = normalize(light_dir + view_dir);

    // Lambert diffuse — clamped at 0 (back faces get no direct light).
    let diffuse_term = max(dot(surface_normal, light_dir), 0.0);

    // Roughness → specular exponent (see header map).
    let specular_exponent = (1.0 - frame_uniforms.mat_params.y) * SHININESS_RANGE + SHININESS_MIN;

    // Blinn specular lobe.
    let specular_lobe = pow(max(dot(surface_normal, half_dir), 0.0), specular_exponent);

    // Metalness lerps specular tint: white (dielectric) → albedo (metal).
    let specular_tint = mix(vec3<f32>(1.0), albedo.rgb, frame_uniforms.mat_params.z);

    // ambient floor + colored diffuse·intensity + tinted specular.
    let lit_rgb = albedo.rgb * (AMBIENT_FLOOR + diffuse_term * frame_uniforms.light_int * frame_uniforms.light_rgb)
        + specular_tint * specular_lobe * frame_uniforms.light_int;

    // Emissive adds after lighting; alpha comes straight from albedo.
    return vec4<f32>(lit_rgb + frame_uniforms.mat_emit.rgb, albedo.a);
}
