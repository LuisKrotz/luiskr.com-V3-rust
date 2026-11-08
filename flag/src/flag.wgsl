// flag.wgsl — the locale flag's aerodynamic wind warp.
//
// WGSL port of core/utils/canvas/widgets/flag/shaders.ts — constants
// preserved 1:1 so pixels are identical under the same uniforms.
//
// Fragment-side chunk only — the fullscreen-triangle vs_main arrives
// via gpu-mini's wgsl library.
//
// ════ THE WIND MODEL (read this before touching constants) ════
//
// A real flag is constrained at the pole and free at the fly edge.
// The math mirrors that:
//   wind_phase = uv.x * WAVE_CYCLES − time * WAVE_SPEED  → a
//     traveling wave: 9.5 ≈ 1.5 wavelengths across the flag's width;
//     speed 3.2/9.5 ≈ 0.34 uv/s → ~3s per full ripple.
//   wind_amp = (POLE_MOTION + FREE_SPAN * uv.x) *
//              (AMP_IDLE + AMP_HOVER * hover)
//     (0.20 + 0.80x) = the "constraint envelope": 20% motion at the
//     pole (it's never truly rigid — cloth), 100% at the fly edge.
//     (0.016 + 0.010h) = base amplitude 1.6% of flag height, rising
//     to 2.6% on hover — hover is additive excitement, not a switch.
//   Warp applies uv.y += sin(phase)·amp  (vertical ripple) and
//   uv.x += cos(phase)·amp·FORESHORTEN (fabric slides horizontally
//   35% as much as it lifts — what makes it read as cloth, not a
//   static wave image).
//
// Per-locale personality (anim_kind 0..11) adds a second detuned
// `flutter` term + a `specular` sheen term per flag — each country's
// fabric weight reads differently. Specular is a luminance adder on
// the SAMPLED texture, scaled (SPECULAR_BASE + SPECULAR_GAIN·hover)
// so sheen grows on hover.
//
// Uniform layout (f32s, flag.rs::UNIFORM_F32S = 12):
//   [0,1] resolution_css canvas CSS px (contract parity — the shader
//         works in clip-rect device px and never reads this field)
//   [2]   time_seconds   seconds
//   [3]   hover_amount   0..1 eased hover amount
//   [4]   anim_kind      0..11 (flag.rs::anims — EN/PT/ES/DE/HRK/CAS/…)
//   [5]   split_flag     >0.5 → split-flag mode (paired-locale flags)
//   [6]   split_x        split seam position 0..1 (only when split)
//   [7]   reserved       unused — keeps the block 16B-aligned per
//                        WGSL uniform rules (vec4 must start at a 16B
//                        boundary)
//   [8..11] clip_rect    widget clip rect, canvas DEVICE px
//
// Bindings (must match lib.rs's flag bind-group construction):
//   0 uniform, 1 flag_a, 2 flag_b, 3 flag_sampler — split flags
//   sample flag_a for x<split_x and flag_b for x≥split_x; single
//   flags bind flag_a twice.

struct Uniforms {
    resolution_css: vec2<f32>,
    time_seconds: f32,
    hover_amount: f32,
    anim_kind: f32,
    split_flag: f32,
    split_x: f32,
    reserved: f32,
    clip_rect: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var flag_a: texture_2d<f32>;
@group(0) @binding(2) var flag_b: texture_2d<f32>;
@group(0) @binding(3) var flag_sampler: sampler;

// ── Tunables ─────────────────────────────────────────────────────────

// The wind model — see the model doc above.
const WAVE_CYCLES: f32 = 9.5;
const WAVE_SPEED: f32 = 3.2;
const POLE_MOTION: f32 = 0.20;
const FREE_SPAN: f32 = 0.80;
const AMP_IDLE: f32 = 0.016;
const AMP_HOVER: f32 = 0.010;
const FORESHORTEN: f32 = 0.35;

// Specular sheen — 40% strength at rest scaling to full on hover.
const SPECULAR_BASE: f32 = 0.4;
const SPECULAR_HOVER_GAIN: f32 = 0.6;

// Split mode — each output half samples the FULL texture's left/right
// half in uv space (the 0.5s are the half-atlas packing); the clamp
// is an edge guard — warp can push samples outside [0,1] and
// unclamped wrapping would smear the opposite edge's pixels.
const HALF_ATLAS: f32 = 0.5;
const UV_GUARD_LO: f32 = 0.005;
const UV_GUARD_HI: f32 = 0.995;
const SPLIT_GATE: f32 = 0.5;

// The seam — a 1.5%-wide darkened dividing line so the two flag
// halves read as stitched cloth, not a hard cut.
const SEAM_WIDTH: f32 = 0.015;
const SEAM_DARKEN: f32 = 0.4;
const SEAM_INK = vec4<f32>(0.1, 0.1, 0.1, 1.0);

// Rounded corners — a 3px flag-cloth radius (a physical flag detail,
// not a theme dimension — intentionally not a token).
const CORNER_RADIUS_PX: f32 = 3.0;
const CORNER_AA_PX: f32 = 1.0;

@fragment
fn fs_main(@builtin(position) pixel_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // Normalize into the widget rect; y flips so top-left matches the
    // original image-coordinate convention.
    let widget_uv = vec2<f32>(
        (pixel_coord.x - uniforms.clip_rect.x) / uniforms.clip_rect.z,
        (pixel_coord.y - uniforms.clip_rect.y) / uniforms.clip_rect.w,
    );

    var warped_uv = widget_uv;
    var specular = 0.0;

    // Wind model: fixed at flagpole (x=0) → free flutter at edge (x=1).
    let wind_phase = widget_uv.x * WAVE_CYCLES - uniforms.time_seconds * WAVE_SPEED;
    let wind_amp = (POLE_MOTION + FREE_SPAN * widget_uv.x) * (AMP_IDLE + AMP_HOVER * uniforms.hover_amount);
    let base_wave = sin(wind_phase);

    var flutter = 0.0;

    // Per-locale personality — the detune frequencies and specular
    // coefficients are per-flag fabric weights (lookup-table data,
    // not tunables); each branch is its country's cloth.
    if (uniforms.anim_kind < 0.5) {
        // EN — stars shimmer
        flutter = sin(wind_phase * 1.5) * 0.005 + cos(widget_uv.y * 7.0 - uniforms.time_seconds * 1.8) * 0.003;
        specular = (0.5 + 0.5 * sin(wind_phase)) * 0.28;
    } else if (uniforms.anim_kind < 1.5) {
        // PT — rhombus solar glow
        flutter = sin(wind_phase * 1.3) * 0.006 + sin(widget_uv.y * 6.0 - uniforms.time_seconds * 1.5) * 0.003;
        specular = (0.5 + 0.5 * cos(wind_phase)) * 0.30;
    } else if (uniforms.anim_kind < 2.5) {
        // ES — flamenco silk
        flutter = sin(wind_phase + widget_uv.y * 4.0) * 0.007;
        specular = (0.5 + 0.5 * sin(wind_phase)) * 0.32;
    } else if (uniforms.anim_kind < 3.5) {
        // DE — cross-ribbon
        flutter = cos(wind_phase * 1.2) * 0.005 + sin(widget_uv.y * 5.0 - uniforms.time_seconds * 1.6) * 0.003;
        specular = (0.5 + 0.5 * sin(wind_phase)) * 0.25;
    } else if (uniforms.anim_kind < 4.5) {
        // HRK — harmonic wave
        flutter = sin(wind_phase * 1.4) * 0.006;
        specular = (0.5 + 0.5 * sin(wind_phase)) * 0.28;
    } else if (uniforms.anim_kind < 5.5) {
        // CAS — Sol de Mayo rays
        flutter = sin(wind_phase * 1.2) * 0.005 + cos(widget_uv.y * 6.0 - uniforms.time_seconds * 2.0) * 0.003;
        specular = (0.5 + 0.5 * cos(wind_phase)) * 0.35;
    } else if (uniforms.anim_kind < 6.5) {
        // RIV — river reflection
        flutter = sin(wind_phase * 1.3) * 0.007;
        specular = (0.5 + 0.5 * sin(wind_phase)) * 0.30;
    } else if (uniforms.anim_kind < 7.5) {
        // GN — horizontal surge
        flutter = cos(wind_phase * 1.1) * 0.006;
        specular = (0.5 + 0.5 * sin(wind_phase)) * 0.28;
    } else if (uniforms.anim_kind < 8.5) {
        // IT — mediterranean silk
        flutter = sin(wind_phase * 1.2) * 0.006 + sin(widget_uv.y * 5.0 - uniforms.time_seconds * 1.5) * 0.003;
        specular = (0.5 + 0.5 * sin(wind_phase)) * 0.30;
    } else if (uniforms.anim_kind < 9.5) {
        // RU — aurora shimmer
        flutter = sin(wind_phase * 1.1) * 0.007 + cos(widget_uv.y * 7.0 - uniforms.time_seconds * 1.8) * 0.003;
        specular = (0.5 + 0.5 * cos(wind_phase)) * 0.35;
    } else if (uniforms.anim_kind < 10.5) {
        // FR — satin ripple
        flutter = sin(wind_phase * 1.3) * 0.006;
        specular = (0.5 + 0.5 * sin(wind_phase)) * 0.28;
    } else {
        // TLN — venetian pulse
        flutter = sin(wind_phase * 1.2) * 0.006 + cos(widget_uv.y * 6.0 - uniforms.time_seconds * 1.9) * 0.003;
        specular = (0.5 + 0.5 * sin(wind_phase)) * 0.30;
    }

    warped_uv.y = warped_uv.y + (base_wave * wind_amp) + (flutter * wind_amp);
    warped_uv.x = warped_uv.x + cos(wind_phase) * (wind_amp * FORESHORTEN);

    var color = vec4<f32>(0.0);

    if (uniforms.split_flag > SPLIT_GATE) {
        // Split mode: the flag's left half shows flag_a remapped into
        // [0,0.5] and the right half flag_b into [0.5,1.0] — each half
        // of the OUTPUT samples the FULL texture but only its
        // left/right half in uv space. Both halves sample
        // unconditionally: `warped_uv.x` is fragment-varying so a
        // textureSample inside `if (warped_uv.x < split_x)` is
        // non-uniform control flow — illegal for implicit-derivative
        // sampling (Dawn: "'textureSample' must only be called from
        // uniform control flow"). Sample both, select.
        var uv_a = vec2<f32>(warped_uv.x / uniforms.split_x * HALF_ATLAS, warped_uv.y);
        uv_a = clamp(uv_a, vec2<f32>(UV_GUARD_LO), vec2<f32>(UV_GUARD_HI));

        var uv_b = vec2<f32>(HALF_ATLAS + (warped_uv.x - uniforms.split_x) / (1.0 - uniforms.split_x) * HALF_ATLAS, warped_uv.y);
        uv_b = clamp(uv_b, vec2<f32>(UV_GUARD_LO), vec2<f32>(UV_GUARD_HI));

        let sample_a = textureSample(flag_a, flag_sampler, uv_a);
        let sample_b = textureSample(flag_b, flag_sampler, uv_b);

        color = select(sample_b, sample_a, warped_uv.x < uniforms.split_x);

        // The seam — computed on un-warped uv so the line stays
        // straight while the fabric ripples around it (a warped seam
        // would wiggle).
        let seam_dist = abs(widget_uv.x - uniforms.split_x);
        let seam_mask = 1.0 - smoothstep(0.0, SEAM_WIDTH, seam_dist);
        color = mix(color, SEAM_INK, seam_mask * SEAM_DARKEN);
    } else {
        let uv_single = clamp(warped_uv, vec2<f32>(UV_GUARD_LO), vec2<f32>(UV_GUARD_HI));
        color = textureSample(flag_a, flag_sampler, uv_single);
    }

    // Specular sheen — ADDED luminance (not mixed): a moving highlight
    // across the fabric. Values >1 are clamped by the output format —
    // fine, the specular coefficient ≤0.35 keeps rgb ≤ ~1.35 worst-case.
    color = vec4<f32>(color.rgb + vec3<f32>(specular * (SPECULAR_BASE + SPECULAR_HOVER_GAIN * uniforms.hover_amount)), color.a);

    // Rounded-corner mask — the classic rounded-box distance outside
    // the corner box; smoothstep over the last 1px gives a soft edge.
    let local_px = widget_uv * vec2<f32>(uniforms.clip_rect.z, uniforms.clip_rect.w);
    let half_size = vec2<f32>(uniforms.clip_rect.z, uniforms.clip_rect.w) * 0.5;
    let corner_radius = vec2<f32>(CORNER_RADIUS_PX, CORNER_RADIUS_PX);
    let corner_out = max(abs(local_px - half_size) - (half_size - corner_radius), vec2<f32>(0.0));
    let corner_dist = length(corner_out);
    let alpha = 1.0 - smoothstep(CORNER_RADIUS_PX - CORNER_AA_PX, CORNER_RADIUS_PX, corner_dist);

    // Premultiplied output: rgb × a matches the pipeline's alpha-blend
    // state (src×1 + dst×(1-a)) — writing non-premultiplied colors with
    // premultiplied blending would double-darken semi-transparent texels.
    return vec4<f32>(color.rgb * (color.a * alpha), color.a * alpha);
}
