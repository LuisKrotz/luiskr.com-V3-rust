// theme_slider.wgsl — the day/system/night preference slider.
//
// WGSL port of core/utils/canvas/widgets/theme-slider/shaders.ts.
// This is the LARGEST widget shader — a full procedural scene
// (sky gradient, starfield, sun with corona, aurora, crescent moon,
// two parallax mesa layers, 3D knob) where `progress` 0..2 morphs
// night → system-twilight → day. All coordinates are in ASPECT-
// CORRECTED track space `point` (x ∈ 0..aspect, y ∈ 0..1, track is a
// capsule of radius 0.5 centered at y=0.5).
//
// Fragment-side chunk only — dist_to_segment + vs_main come from
// gpu-mini's wgsl library. The star hash stays local: it uses the
// 12.9898/78.233 seed (not the shared 127.1/311.7 field) — a different
// seed re-rolls the star placement, which is a pixel change.
//
// Uniform layout (f32s, theme_slider.rs::UNIFORM_F32S = 12):
//   [0,1] resolution_css design size px
//   [2]   time_seconds   seconds — star twinkle, aurora, corona, shimmer
//   [3]   theme_progress 0..2 — 0..1: night→system, 1..2: system→day
//   [4]   knob_center_x  knob center X, px (JS spring)
//   [5]   ripple_time    seconds-stamp of last tap (−1 = none)
//   [6]   ripple_pos     tap X in px → expanding ring origin
//   [7]   reserved       16B alignment pad
//   [8..11] clip_rect    widget clip rect, canvas DEVICE px
//
// Layer order (painter's, back→front — do not reorder):
//   sky → stars → sun → aurora → moon → back mesa → front mesa →
//   bezel → click ripple → knob shadow → knob → knob halos.

struct Uniforms {
    resolution_css: vec2<f32>,
    time_seconds: f32,
    theme_progress: f32,
    knob_center_x: f32,
    ripple_time: f32,
    ripple_pos: f32,
    reserved: f32,
    clip_rect: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// ── Tunables ─────────────────────────────────────────────────────────

// The track capsule — radius 0.5 (half the design height), centers
// half a radius in from each end, centered on the midline.
const TRACK_RADIUS: f32 = 0.5;
const TRACK_Y: f32 = 0.5;
const TRACK_INSET: f32 = 0.5;
const BORDER_AA_LO: f32 = 0.485;

// Sky gradients — each lerps horizon→zenith with height. The palette
// is a sunset-family wash (night violet → twilight rose → day coral).
const NIGHT_SKY_LO = vec3<f32>(0.38, 0.28, 0.60);
const NIGHT_SKY_HI = vec3<f32>(0.46, 0.35, 0.70);
const SYSTEM_SKY_LO = vec3<f32>(0.60, 0.38, 0.52);
const SYSTEM_SKY_HI = vec3<f32>(0.72, 0.48, 0.62);
const DAY_SKY_LO = vec3<f32>(0.92, 0.48, 0.40);
const DAY_SKY_HI = vec3<f32>(0.98, 0.64, 0.50);

// Starfield — grid cells of 1/STAR_GRID scene-height; ~28% of cells
// host a star (rnd > STAR_DENSITY). Each star sits at its cell CENTER
// (0.5 offset) and twinkles at time×STAR_TWINKLE_SPEED with a per-cell
// phase so they pulse out of sync.
const STAR_GRID: f32 = 20.0;
const STAR_DENSITY: f32 = 0.72;
const STAR_RADIUS: f32 = 0.022;
const STAR_TWINKLE_SPEED: f32 = 3.5;
const STAR_PHASE_SPAN: f32 = 6.28; // ≈ 2π
const STAR_FADE_LO: f32 = 0.2;
const STAR_FADE_HI: f32 = 1.1;
const STAR_INK = vec3<f32>(1.0);

// Rising sun — fades in across the progress window; a pulsing core
// disc + three concentric glow rings at decreasing alpha + ten
// angular rays rotating slowly.
const SUN_FADE_LO: f32 = 0.3;
const SUN_FADE_HI: f32 = 1.8;
const SUN_POS = vec2<f32>(0.80, 0.62);
const SUN_PULSE_AMP: f32 = 0.012;
const SUN_PULSE_SPEED: f32 = 2.8;
const SUN_CORE_LO: f32 = 0.135;
const SUN_CORE_HI: f32 = 0.165;
const SUN_RING1_LO: f32 = 0.22;
const SUN_RING1_HI: f32 = 0.26;
const SUN_RING1_ALPHA: f32 = 0.45;
const SUN_RING2_LO: f32 = 0.30;
const SUN_RING2_HI: f32 = 0.35;
const SUN_RING2_ALPHA: f32 = 0.28;
const SUN_RING3_LO: f32 = 0.39;
const SUN_RING3_HI: f32 = 0.45;
const SUN_RING3_ALPHA: f32 = 0.16;
const SUN_RAY_COUNT: f32 = 10.0;
const SUN_RAY_SPEED: f32 = 0.85;
const SUN_RAY_LO: f32 = 0.14;
const SUN_RAY_HI: f32 = 0.44;
const SUN_RAY_ALPHA: f32 = 0.28;
const SUN_RING_SPAN: f32 = 0.45;
const SUN_CREAM = vec3<f32>(0.99, 0.90, 0.75);
const SUN_RING_INNER = vec3<f32>(0.98, 0.76, 0.60);
const SUN_RING_OUTER = vec3<f32>(0.96, 0.64, 0.52);

// Twilight aurora — `system_factor` peaks at progress=1.0 and decays
// over a ±0.625 window, so the aurora only exists around "system".
// The wave is a phase-modulated sine (inner sin bends it vertically —
// the curtain effect); the band clips to a y window; the color lerps
// cyan→magenta across x with time drift.
const AURORA_WINDOW: f32 = 1.6;      // 1/window → decay over ±0.625
const AURORA_FREQ_X: f32 = 2.6;
const AURORA_BEND_FREQ: f32 = 3.8;
const AURORA_BEND_SPEED: f32 = 1.4;
const AURORA_DRIFT_SPEED: f32 = 1.6;
const AURORA_RISE_LO: f32 = 0.35;
const AURORA_RISE_HI: f32 = 0.75;
const AURORA_FALL_LO: f32 = 0.70;
const AURORA_FALL_HI: f32 = 0.95;
const AURORA_COLOR_FREQ: f32 = 1.8;
const AURORA_COLOR_DRIFT: f32 = 0.9;
const AURORA_ALPHA: f32 = 0.38;
const AURORA_CYAN = vec3<f32>(0.38, 0.78, 0.92);
const AURORA_MAGENTA = vec3<f32>(0.85, 0.45, 0.80);

// Crescent moon — SDF subtraction: a full disc minus an offset inner
// disc (the offset aims the crescent's horns down-right — waxing
// evening moon).
const MOON_POS = vec2<f32>(2.58, 0.60);
const MOON_BITE = vec2<f32>(-0.052, 0.042);
const MOON_OUTER_LO: f32 = 0.16;
const MOON_OUTER_HI: f32 = 0.18;
const MOON_INNER_LO: f32 = 0.138;
const MOON_INNER_HI: f32 = 0.174;
const MOON_FADE_LO: f32 = 0.1;
const MOON_FADE_HI: f32 = 1.2;
const MOON_ALPHA: f32 = 0.95;
const MOON_INK = vec3<f32>(1.0);

// Canyon mesas — terrain silhouettes as height functions: base +
// slow sine swell + faster cosine detail + two triangular plateaus
// (flat tops = the mesa look). `parallax` slides the back layer
// while the front layer moves half as much in the opposite
// direction — differential motion sells depth. `heat_shimmer` adds a
// high-frequency wobble only during day.
const PARALLAX_RANGE: f32 = 0.12;
const FRONT_PARALLAX: f32 = 0.5;
const DAY_FADE_LO: f32 = 1.0;
const DAY_FADE_HI: f32 = 1.9;
const SHIMMER_FREQ: f32 = 14.0;
const SHIMMER_SPEED: f32 = 2.8;
const SHIMMER_AMP: f32 = 0.005;
const BACK_BASE: f32 = 0.33;
const BACK_SWELL_AMP: f32 = 0.14;
const BACK_SWELL_FREQ: f32 = 2.6;
const BACK_DETAIL_AMP: f32 = 0.09;
const BACK_DETAIL_FREQ: f32 = 5.4;
const PLATEAU1_X: f32 = 1.72;
const PLATEAU1_WIDTH: f32 = 0.3;
const PLATEAU1_AMP: f32 = 0.16;
const PLATEAU2_X: f32 = 2.68;
const PLATEAU2_WIDTH: f32 = 0.24;
const PLATEAU2_AMP: f32 = 0.18;
const BACK_SHIMMER_SHARE: f32 = 0.5;
const FRONT_BASE: f32 = 0.21;
const FRONT_SWELL_AMP: f32 = 0.09;
const FRONT_SWELL_FREQ: f32 = 2.4;
const FRONT_SWELL_PHASE: f32 = 0.8;
const FRONT_DETAIL_AMP: f32 = 0.05;
const FRONT_DETAIL_FREQ: f32 = 3.8;
const MESA_AA_LO: f32 = 0.015;
const MESA_AA_HI: f32 = 0.005;
const MESA_BACK_NIGHT = vec3<f32>(0.34, 0.16, 0.49);
const MESA_BACK_SYSTEM = vec3<f32>(0.52, 0.24, 0.45);
const MESA_BACK_DAY = vec3<f32>(0.76, 0.30, 0.40);
const MESA_FRONT_NIGHT = vec3<f32>(0.42, 0.22, 0.58);
const MESA_FRONT_SYSTEM = vec3<f32>(0.62, 0.30, 0.50);
const MESA_FRONT_DAY = vec3<f32>(0.85, 0.40, 0.48);

// Inset bezel — the outer band darkens toward the edge so the track
// reads as inset glass.
const BEZEL_LO: f32 = 0.46;
const BEZEL_HI: f32 = 0.50;
const BEZEL_DARKEN: f32 = 0.65;
const BEZEL_STRENGTH: f32 = 0.55;

// Tap ripple — an expanding ring, radius = elapsed×1.0 (one
// scene-height/s), alpha decaying linearly to 0 over its life. The
// `>= 0` guard keeps ripple_time=−1 (never tapped) inert.
const RIPPLE_LIFE: f32 = 0.6;
const RIPPLE_SPEED: f32 = 1.0;
const RIPPLE_WIDTH: f32 = 0.08;
const RIPPLE_STRENGTH: f32 = 0.55;
const RIPPLE_INK = vec3<f32>(0.35, 0.25, 0.45);

// The knob — a fake-3D sphere tinted moon→system→sun, with lunar
// craters fading out as it morphs and a colored halo per mode.
const KNOB_RADIUS: f32 = 0.38;
const KNOB_SHADOW_WIDTH: f32 = 0.07;
const KNOB_SHADOW_STRENGTH: f32 = 0.38;
const KNOB_SHADOW_INK = vec3<f32>(0.12, 0.08, 0.20);
const KNOB_NIGHT = vec3<f32>(0.98, 0.98, 1.0);
const KNOB_SYSTEM = vec3<f32>(0.98, 0.94, 0.90);
const KNOB_DAY = vec3<f32>(0.99, 0.89, 0.73);
const CRATER_FADE_LO: f32 = 0.1;
const CRATER_FADE_HI: f32 = 0.9;
const CRATER_STRENGTH: f32 = 0.75;
const CRATER_INK = vec3<f32>(0.88, 0.89, 0.92);
const KNOB_RIM_DARKEN: f32 = 0.14;
const KNOB_AA_LO: f32 = 0.02;
const KNOB_AA_HI: f32 = 0.005;
const SUN_HALO_WIDTH: f32 = 0.08;
const SUN_HALO_PULSE_AMP: f32 = 0.02;
const SUN_HALO_PULSE_SPEED: f32 = 3.0;
const SUN_HALO_ALPHA: f32 = 0.42;
const SUN_HALO_INK = vec3<f32>(1.0, 0.82, 0.50);
const SYSTEM_HALO_WIDTH: f32 = 0.07;
const SYSTEM_HALO_PULSE_AMP: f32 = 0.015;
const SYSTEM_HALO_PULSE_SPEED: f32 = 2.4;
const SYSTEM_HALO_ALPHA: f32 = 0.35;
const SYSTEM_HALO_INK = vec3<f32>(0.85, 0.72, 0.98);

// Cheap visibility gate — layers below this alpha skip their pass.
const LIVE_EPSILON: f32 = 0.01;

/// `hash_star_cell` → [0,1) — the classic sin-fract hash for
/// deterministic per-cell randomness, tuned to THIS scene's seed
/// (12.9898/78.233, ×437.585453 — not the shared library's field;
/// switching would re-roll every star's placement). dot() decorrelates
/// x/y. NOT cryptographic — it only needs cells to look random for
/// star placement. Deterministic = same stars every frame (essential —
/// a time-varying hash would boil).
fn hash_star_cell(cell: vec2<f32>) -> f32 {
    return fract(sin(dot(cell, vec2<f32>(12.9898, 78.233))) * 437.585453);
}

@fragment
fn fs_main(@builtin(position) pixel_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // WGSL `pixel_coord` is framebuffer-origin (y=0 at TOP) while the
    // ported GLSL math assumes v_uv semantics (y=0 at BOTTOM) — without
    // the flip the whole scene renders upside-down: stars at the
    // bottom, mesas at the top, sun rising from the floor.
    let widget_uv = vec2<f32>(
        (pixel_coord.x - uniforms.clip_rect.x) / uniforms.clip_rect.z,
        1.0 - (pixel_coord.y - uniforms.clip_rect.y) / uniforms.clip_rect.w,
    );

    let aspect = uniforms.resolution_css.x / uniforms.resolution_css.y;
    let point = vec2<f32>(widget_uv.x * aspect, widget_uv.y);

    let track_start = vec2<f32>(TRACK_INSET, TRACK_Y);
    let track_end = vec2<f32>(aspect - TRACK_INSET, TRACK_Y);
    let track_dist = dist_to_segment(point, track_start, track_end);

    if (track_dist > TRACK_RADIUS) {
        discard;
    }

    let border_alpha = 1.0 - smoothstep(BORDER_AA_LO, TRACK_RADIUS, track_dist);

    // Sky gradient by height.
    let height = point.y;
    let night_sky = mix(NIGHT_SKY_LO, NIGHT_SKY_HI, height);
    let system_sky = mix(SYSTEM_SKY_LO, SYSTEM_SKY_HI, height);
    let day_sky = mix(DAY_SKY_LO, DAY_SKY_HI, height);

    var sky: vec3<f32>;
    if (uniforms.theme_progress <= 1.0) {
        sky = mix(night_sky, system_sky, uniforms.theme_progress);
    } else {
        sky = mix(system_sky, day_sky, uniforms.theme_progress - 1.0);
    }

    var color = sky;

    // Twinkling stars — only for progress < ~1.1 (night + twilight).
    let star_alpha = 1.0 - smoothstep(STAR_FADE_LO, STAR_FADE_HI, uniforms.theme_progress);
    if (star_alpha > LIVE_EPSILON) {
        let star_cell = floor(point * STAR_GRID);
        let star_seed = hash_star_cell(star_cell);
        if (star_seed > STAR_DENSITY) {
            let star_pos = (star_cell + vec2<f32>(0.5)) / STAR_GRID;
            let star_dist = length(point - star_pos);
            let twinkle = 0.5 + 0.5 * sin(uniforms.time_seconds * STAR_TWINKLE_SPEED + star_seed * STAR_PHASE_SPAN);
            let star_glow = (1.0 - smoothstep(0.0, STAR_RADIUS, star_dist)) * twinkle * star_alpha;
            color = color + STAR_INK * star_glow;
        }
    }

    // Rising sun — `corona_pulse` breathes the core radius ±1.2%;
    // `total_sun` is clamped because ring alphas overlap and would
    // sum >1.
    let sun_alpha = smoothstep(SUN_FADE_LO, SUN_FADE_HI, uniforms.theme_progress);
    if (sun_alpha > LIVE_EPSILON) {
        let sun_dist = length(point - SUN_POS);
        let corona_pulse = SUN_PULSE_AMP * sin(uniforms.time_seconds * SUN_PULSE_SPEED);
        let core_mask = 1.0 - smoothstep(SUN_CORE_LO + corona_pulse, SUN_CORE_HI + corona_pulse, sun_dist);
        let ring1_mask = (1.0 - smoothstep(SUN_RING1_LO, SUN_RING1_HI, sun_dist)) * SUN_RING1_ALPHA;
        let ring2_mask = (1.0 - smoothstep(SUN_RING2_LO, SUN_RING2_HI, sun_dist)) * SUN_RING2_ALPHA;
        let ring3_mask = (1.0 - smoothstep(SUN_RING3_LO, SUN_RING3_HI, sun_dist)) * SUN_RING3_ALPHA;

        let sun_angle = atan2(point.y - SUN_POS.y, point.x - SUN_POS.x);
        let sun_rays = 0.5 + 0.5 * sin(sun_angle * SUN_RAY_COUNT + uniforms.time_seconds * SUN_RAY_SPEED);
        let ray_glow = (1.0 - smoothstep(SUN_RAY_LO, SUN_RAY_HI, sun_dist)) * sun_rays * SUN_RAY_ALPHA;

        let ring_ink = mix(SUN_RING_INNER, SUN_RING_OUTER, clamp(sun_dist / SUN_RING_SPAN, 0.0, 1.0));
        let total_sun = clamp(core_mask + ring1_mask + ring2_mask + ring3_mask + ray_glow, 0.0, 1.0);
        color = mix(color, mix(ring_ink, SUN_CREAM, core_mask), total_sun * sun_alpha);
    }

    // Twilight aurora — phase-modulated sine wave clipped to a y
    // window, color lerping cyan→magenta across x with time drift.
    let system_factor = 1.0 - min(1.0, abs(uniforms.theme_progress - 1.0) * AURORA_WINDOW);
    if (system_factor > LIVE_EPSILON) {
        let aurora_wave = sin(
            point.x * AURORA_FREQ_X
            + sin(point.y * AURORA_BEND_FREQ + uniforms.time_seconds * AURORA_BEND_SPEED)
            + uniforms.time_seconds * AURORA_DRIFT_SPEED,
        ) * 0.5 + 0.5;
        let aurora_band = smoothstep(AURORA_RISE_LO, AURORA_RISE_HI, point.y)
            * (1.0 - smoothstep(AURORA_FALL_LO, AURORA_FALL_HI, point.y));
        let aurora_ink = mix(
            AURORA_CYAN,
            AURORA_MAGENTA,
            sin(point.x * AURORA_COLOR_FREQ + uniforms.time_seconds * AURORA_COLOR_DRIFT) * 0.5 + 0.5,
        );
        color = color + aurora_ink * aurora_wave * aurora_band * AURORA_ALPHA * system_factor;
    }

    // Crescent moon — disc minus offset disc leaves the lit sliver.
    let moon_outer_dist = length(point - MOON_POS);
    let moon_inner_dist = length(point - (MOON_POS + MOON_BITE));
    let moon_outer = 1.0 - smoothstep(MOON_OUTER_LO, MOON_OUTER_HI, moon_outer_dist);
    let moon_inner = smoothstep(MOON_INNER_LO, MOON_INNER_HI, moon_inner_dist);
    let crescent = moon_outer * moon_inner;
    let crescent_alpha = 1.0 - smoothstep(MOON_FADE_LO, MOON_FADE_HI, uniforms.theme_progress);
    color = mix(color, MOON_INK, clamp(crescent, 0.0, 1.0) * crescent_alpha * MOON_ALPHA);

    // Canyon mesas — the back layer parallaxes with progress; pixels
    // below each height function fill with that mesa's color.
    let parallax = (uniforms.theme_progress - 1.0) * PARALLAX_RANGE;
    let back_x = point.x + parallax;
    let day_alpha = smoothstep(DAY_FADE_LO, DAY_FADE_HI, uniforms.theme_progress);
    let heat_shimmer = sin(point.x * SHIMMER_FREQ + uniforms.time_seconds * SHIMMER_SPEED) * SHIMMER_AMP * day_alpha;

    let back_swell = BACK_BASE + BACK_SWELL_AMP * sin(back_x * BACK_SWELL_FREQ) + BACK_DETAIL_AMP * cos(back_x * BACK_DETAIL_FREQ);
    let plateau1 = (1.0 - smoothstep(0.0, PLATEAU1_WIDTH, abs(back_x - PLATEAU1_X))) * PLATEAU1_AMP;
    let plateau2 = (1.0 - smoothstep(0.0, PLATEAU2_WIDTH, abs(back_x - PLATEAU2_X))) * PLATEAU2_AMP;
    let back_height = back_swell + plateau1 + plateau2 + heat_shimmer * BACK_SHIMMER_SHARE;

    var back_ink: vec3<f32>;
    if (uniforms.theme_progress <= 1.0) {
        back_ink = mix(MESA_BACK_NIGHT, MESA_BACK_SYSTEM, uniforms.theme_progress);
    } else {
        back_ink = mix(MESA_BACK_SYSTEM, MESA_BACK_DAY, uniforms.theme_progress - 1.0);
    }

    let back_mask = 1.0 - smoothstep(back_height - MESA_AA_LO, back_height + MESA_AA_HI, point.y);
    color = mix(color, back_ink, back_mask);

    let front_height = FRONT_BASE
        + FRONT_SWELL_AMP * sin((point.x - parallax * FRONT_PARALLAX) * FRONT_SWELL_FREQ + FRONT_SWELL_PHASE)
        + FRONT_DETAIL_AMP * cos(point.x * FRONT_DETAIL_FREQ)
        + heat_shimmer;
    var front_ink: vec3<f32>;
    if (uniforms.theme_progress <= 1.0) {
        front_ink = mix(MESA_FRONT_NIGHT, MESA_FRONT_SYSTEM, uniforms.theme_progress);
    } else {
        front_ink = mix(MESA_FRONT_SYSTEM, MESA_FRONT_DAY, uniforms.theme_progress - 1.0);
    }

    let front_mask = 1.0 - smoothstep(front_height - MESA_AA_LO, front_height + MESA_AA_HI, point.y);
    color = mix(color, front_ink, front_mask);

    // Inset bezel.
    let bezel = smoothstep(BEZEL_LO, BEZEL_HI, track_dist);
    color = mix(color, color * BEZEL_DARKEN, bezel * BEZEL_STRENGTH);

    // Tap ripple — expanding ring from the tap X on the track midline.
    let ripple_elapsed = uniforms.time_seconds - uniforms.ripple_time;
    if (ripple_elapsed >= 0.0 && ripple_elapsed < RIPPLE_LIFE) {
        let ripple_x = (uniforms.ripple_pos / uniforms.resolution_css.x) * aspect;
        let ripple_dist = length(point - vec2<f32>(ripple_x, TRACK_Y));
        let ripple_radius = ripple_elapsed * RIPPLE_SPEED;
        let ripple_wave = (1.0 - smoothstep(0.0, RIPPLE_WIDTH, abs(ripple_dist - ripple_radius)))
            * (1.0 - ripple_elapsed / RIPPLE_LIFE);
        color = color + RIPPLE_INK * ripple_wave * RIPPLE_STRENGTH;
    }

    // 3D knob — moon/system/sun tint + lunar craters + corona halo.
    let knob_x_norm = (uniforms.knob_center_x / uniforms.resolution_css.x) * aspect;
    let knob_center = vec2<f32>(knob_x_norm, TRACK_Y);
    let knob_dist = length(point - knob_center);

    let knob_shadow = (1.0 - smoothstep(KNOB_RADIUS, KNOB_RADIUS + KNOB_SHADOW_WIDTH, knob_dist)) * KNOB_SHADOW_STRENGTH;
    color = mix(color, KNOB_SHADOW_INK, knob_shadow);

    var knob_ink: vec3<f32>;
    if (uniforms.theme_progress <= 1.0) {
        knob_ink = mix(KNOB_NIGHT, KNOB_SYSTEM, uniforms.theme_progress);
    } else {
        knob_ink = mix(KNOB_SYSTEM, KNOB_DAY, uniforms.theme_progress - 1.0);
    }

    // Lunar craters — five small circles at fixed offsets with
    // individually-tuned radii (irregular spacing sells the moon);
    // they fade out as the knob morphs moon→sun.
    let crater_alpha = 1.0 - smoothstep(CRATER_FADE_LO, CRATER_FADE_HI, uniforms.theme_progress);
    if (crater_alpha > LIVE_EPSILON) {
        let knob_local = point - knob_center;
        let crater1 = 1.0 - smoothstep(0.075, 0.095, length(knob_local - vec2<f32>(-0.13, 0.12)));
        let crater2 = 1.0 - smoothstep(0.053, 0.071, length(knob_local - vec2<f32>(-0.11, -0.11)));
        let crater3 = 1.0 - smoothstep(0.041, 0.059, length(knob_local - vec2<f32>(0.12, -0.07)));
        let crater4 = 1.0 - smoothstep(0.059, 0.077, length(knob_local - vec2<f32>(0.09, 0.11)));
        let crater5 = 1.0 - smoothstep(0.024, 0.042, length(knob_local - vec2<f32>(-0.01, 0.02)));
        let any_crater = clamp(crater1 + crater2 + crater3 + crater4 + crater5, 0.0, 1.0);
        knob_ink = mix(knob_ink, CRATER_INK, any_crater * crater_alpha * CRATER_STRENGTH);
    }

    let sphere_shade = 1.0 - (knob_dist / KNOB_RADIUS) * KNOB_RIM_DARKEN;
    let lit_knob = knob_ink * sphere_shade;

    let knob_alpha = 1.0 - smoothstep(KNOB_RADIUS - KNOB_AA_LO, KNOB_RADIUS + KNOB_AA_HI, knob_dist);
    color = mix(color, lit_knob, knob_alpha);

    if (day_alpha > LIVE_EPSILON) {
        let sun_halo = (1.0 - smoothstep(
            KNOB_RADIUS,
            KNOB_RADIUS + SUN_HALO_WIDTH + SUN_HALO_PULSE_AMP * sin(uniforms.time_seconds * SUN_HALO_PULSE_SPEED),
            knob_dist,
        )) * SUN_HALO_ALPHA * day_alpha;
        color = color + SUN_HALO_INK * sun_halo;
    }
    if (system_factor > LIVE_EPSILON) {
        let system_halo = (1.0 - smoothstep(
            KNOB_RADIUS,
            KNOB_RADIUS + SYSTEM_HALO_WIDTH + SYSTEM_HALO_PULSE_AMP * sin(uniforms.time_seconds * SYSTEM_HALO_PULSE_SPEED),
            knob_dist,
        )) * SYSTEM_HALO_ALPHA * system_factor;
        color = color + SYSTEM_HALO_INK * system_halo;
    }

    return vec4<f32>(color * border_alpha, border_alpha);
}
