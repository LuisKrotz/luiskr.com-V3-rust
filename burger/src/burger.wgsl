// burger.wgsl — the nav hamburger's three hairline bars.
//
// WGSL port of burger-button's fragment shader
// (core/utils/canvas/widgets/burger-button/shaders.ts) — the port is
// faithful: every constant below matches the GLSL original so the
// rendered pixels are identical when the same uniforms arrive.
//
// ════════════════════════════════════════════════════════════════════
// HOW THIS DIFFERS FROM A TRADITIONAL UI ELEMENT
// ════════════════════════════════════════════════════════════════════
//
// A normal page draws the burger as three <div> bars (or an SVG). The
// browser rasterizes them once; animating means moving DOM nodes —
// layout → paint → composite every frame (~ms each at worst).
//
// Here there is NO geometry at all: a single fullscreen triangle covers
// the widget rect, and this fragment shader answers "is this PIXEL part
// of a bar?" per pixel per frame using signed-distance-field math.
// Cost: one draw call + ~30 ALU ops per pixel — for a 36×14 widget that
// is ~15k ops total, nanoseconds on GPU. No layout pass exists; the
// shader IS the layout.
//
// This file is the widget's FRAGMENT-SIDE chunk only — the shared
// sd_capsule primitive, the ink/glow palette, and the vs_main triangle
// come from gpu_mini's wgsl library (injected by init_widget).
//
// Uniform layout (f32s, must match burger.rs::UNIFORM_F32S = 8):
//   [0]   time_seconds   seconds since widget mount — drives every sine
//   [1,2] resolution_css widget pixel size (CSS px design units)
//   [3]   theme_dark     0.0 = light theme, 1.0 = dark theme (lerp)
//   [4..7] clip_rect     widget clip rect in canvas DEVICE px (x,y,w,h)

struct Uniforms {
    time_seconds: f32,
    resolution_css: vec2<f32>,
    theme_dark: f32,
    clip_rect: vec4<f32>, // x, y, w, h — canvas DEVICE px (widget's clip)
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// ── Tunables ─────────────────────────────────────────────────────────
// Bar geometry in UV units. SAFE RANGES:
//   BAR_WIDTH_UV 0.54  — bar half-extent 0.27 each side of center; >0.9
//                        and bars reach the widget's edge, look cropped.
//   BAR_HEIGHT_UV 0.045 — hairline ~2px on a 44px-tall widget; raise to
//                        0.07 for a chunkier bar, keep < BAR_GAP_UV or
//                        the bars merge.
//   BAR_GAP_UV 0.16    — center spacing ±0.16 around y=0.5.
const BAR_WIDTH_UV: f32 = 0.54;
const BAR_HEIGHT_UV: f32 = 0.045;
const BAR_GAP_UV: f32 = 0.16;

// Wave ripple: sin(widget_uv.x * FREQ − t * SPEED + phase) * AMPLITUDE.
//   FREQ 5.0      → ~0.8 wavelengths across the bar (spatial frequency)
//   SPEED 0.7     → wave travels right at 0.7/5 ≈ 0.14 uv/s ≈ 9s period
//   AMPLITUDE 0.009 → ±0.9% vertical shimmer. Raise toward 0.02 for a
//     livelier wave; >0.03 makes hairlines visibly crawl at 120/240Hz.
//   Phases are thirds of ~2π — bars never crest together, so the group
//   reads as one continuous surface.
const WAVE_SPATIAL_FREQ: f32 = 5.0;
const WAVE_SPEED: f32 = 0.7;
const WAVE_AMPLITUDE_UV: f32 = 0.009;
const WAVE_PHASE_MIDDLE: f32 = 1.3;
const WAVE_PHASE_TOP: f32 = 2.6;

// Per-bar vertical bob at 0.55 rad/s (~11s period), amplitude 0.4% of
// widget height. Detuned phases keep it uncorrelated with the ripple —
// two detuned frequencies never visibly "beat".
const BOB_SPEED: f32 = 0.55;
const BOB_AMPLITUDE_UV: f32 = 0.004;
const BOB_PHASE_MIDDLE: f32 = 1.2;
const BOB_PHASE_TOP: f32 = 2.4;

// Breathing: bar widths grow/shrink ±4% at 0.45 rad/s (~14s period).
const BREATHE_SPEED: f32 = 0.45;
const BREATHE_DEPTH: f32 = 0.04;
// ±0.03 width offsets keep the middle bar visually equal as it
// counter-shrinks against the outer bars' grow (constant ink mass).
const MIDDLE_BAR_EXTRA_UV: f32 = 0.03;
const TOP_BAR_LESS_UV: f32 = 0.03;

// Anti-aliasing + halo. The AA band is sized in FIELD units via
// fwidth — a ~2.2px soft edge at ANY zoom/DPR; MIN_FIELD_GRADIENT
// guards the degenerate flat-field case (fwidth=0 → hard alias).
const MIN_FIELD_GRADIENT: f32 = 1e-4;
const AA_BAND_SCALE: f32 = 1.1;
// Halo: a 0..5.5px-wide field bloom at 20% luminance; keep <0.3 or the
// hairlines bloom fat. HALO_ALPHA_FLOOR lets pure-halo pixels show.
const HALO_RADIUS_SCALE: f32 = 5.5;
const HALO_STRENGTH: f32 = 0.2;
const HALO_ALPHA_FLOOR: f32 = 0.25;

// Luminance breathing ±2% at 0.5 rad/s (~12.6s) — slow enough to be
// sub-perceptual as motion but keeps the icon alive on idle.
const PULSE_BASE: f32 = 0.98;
const PULSE_DEPTH: f32 = 0.02;
const PULSE_SPEED: f32 = 0.5;

// A horizontal bar as a capsule: height h → radius h/2, width w → the
// spine runs from 0.5−halfLen to 0.5+halfLen (UV space, centered at
// x=0.5). `max(width*0.5 − radius, 0)` guards the degenerate w<2r case
// (a capsule whose "spine" is negative-length — the clamp keeps it a
// pure circle).
fn sd_horizontal_bar(point: vec2<f32>, center_y: f32, width: f32, height: f32) -> f32 {
    let radius = height * 0.5;
    let half_len = max(width * 0.5 - radius, 0.0);

    return sd_capsule(
        point,
        vec2<f32>(0.5 - half_len, center_y),
        vec2<f32>(0.5 + half_len, center_y),
        radius,
    );
}

@fragment
fn fs_main(@builtin(position) pixel_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // `pixel_coord` arrives in SURFACE pixels (device px, whole canvas).
    // The original shader ran on its own canvas where frag==widget px.
    // (pixel_coord − clip_rect.xy)/clip_rect.zw converts to widget-UV
    // 0..1 so the ported math is byte-identical to the canvas version.
    // NOTE: clip_rect is DEVICE px (written by `resize`) — same unit as
    // `pixel_coord`, so the division is exact at any DPR.
    // widget_uv.y is flipped: WGSL frag coords are framebuffer-origin
    // (y=0 TOP) but the ported GLSL measured gl_FragCoord (y=0 BOTTOM)
    // — the bars' wave phase and asymmetry constants assume bottom-up.
    let widget_uv = vec2<f32>(
        (pixel_coord.x - uniforms.clip_rect.x) / uniforms.clip_rect.z,
        1.0 - (pixel_coord.y - uniforms.clip_rect.y) / uniforms.clip_rect.w,
    );
    let time_seconds = uniforms.time_seconds;

    // ── Per-bar offsets ─────────────────────────────────────────────
    // In bottom-origin uv, bar at 0.5−GAP is the BOTTOM bar, 0.5 the
    // middle, 0.5+GAP the top.
    let wave_bottom = sin(widget_uv.x * WAVE_SPATIAL_FREQ - time_seconds * WAVE_SPEED) * WAVE_AMPLITUDE_UV;
    let wave_middle = sin(widget_uv.x * WAVE_SPATIAL_FREQ - time_seconds * WAVE_SPEED + WAVE_PHASE_MIDDLE) * WAVE_AMPLITUDE_UV;
    let wave_top = sin(widget_uv.x * WAVE_SPATIAL_FREQ - time_seconds * WAVE_SPEED + WAVE_PHASE_TOP) * WAVE_AMPLITUDE_UV;

    let bob_bottom = sin(time_seconds * BOB_SPEED) * BOB_AMPLITUDE_UV;
    let bob_middle = sin(time_seconds * BOB_SPEED + BOB_PHASE_MIDDLE) * BOB_AMPLITUDE_UV;
    let bob_top = sin(time_seconds * BOB_SPEED + BOB_PHASE_TOP) * BOB_AMPLITUDE_UV;

    // grow oscillates 0.96..1.04; outer bars scale by `grow`, the middle
    // by `2.0 − grow` — it counter-shrinks so total ink mass is constant
    // (eye reads it as soft pressure, not flicker).
    let grow = 1.0 + BREATHE_DEPTH * sin(time_seconds * BREATHE_SPEED);

    // Each bar's SDF evaluated at the wave-shifted sample point. Moving
    // the SAMPLE by −wave is mathematically identical to moving the BAR
    // by +wave (SDF convention: transform the point, not the shape).
    let dist_bottom_bar = sd_horizontal_bar(
        vec2<f32>(widget_uv.x, widget_uv.y + wave_bottom),
        0.5 - BAR_GAP_UV + bob_bottom,
        BAR_WIDTH_UV * grow,
        BAR_HEIGHT_UV,
    );
    let dist_middle_bar = sd_horizontal_bar(
        vec2<f32>(widget_uv.x, widget_uv.y + wave_middle),
        0.5 + bob_middle,
        (BAR_WIDTH_UV + MIDDLE_BAR_EXTRA_UV) * (2.0 - grow),
        BAR_HEIGHT_UV,
    );
    let dist_top_bar = sd_horizontal_bar(
        vec2<f32>(widget_uv.x, widget_uv.y + wave_top),
        0.5 + BAR_GAP_UV + bob_top,
        (BAR_WIDTH_UV - TOP_BAR_LESS_UV) * grow,
        BAR_HEIGHT_UV,
    );

    // Union of the three bars = minimum distance (standard SDF union).
    let bars_field = min(min(dist_bottom_bar, dist_middle_bar), dist_top_bar);

    // fwidth(bars_field) = |∂field/∂x| + |∂field/∂y| — how fast the field
    // changes per pixel; smoothstep over ±1.1×fwidth produces the ~2.2px
    // soft edge that is the whole reason SDFs beat alpha sprites.
    let edge_width = max(fwidth(bars_field), MIN_FIELD_GRADIENT);
    let bar_alpha = 1.0 - smoothstep(-edge_width * AA_BAND_SCALE, edge_width * AA_BAND_SCALE, bars_field);
    let halo = (1.0 - smoothstep(0.0, edge_width * HALO_RADIUS_SCALE, bars_field)) * HALO_STRENGTH;

    let luminance_pulse = PULSE_BASE + PULSE_DEPTH * sin(time_seconds * PULSE_SPEED);

    // Theme inks come from the shared palette so every widget — and the
    // WebGL fallback — paints the identical ink. `mix` lerps on
    // theme_dark: 1.0 → INK_ON_DARK.
    let ink = mix(INK_ON_LIGHT * luminance_pulse, INK_ON_DARK * luminance_pulse, uniforms.theme_dark);
    let halo_color = mix(GLOW_ON_LIGHT, GLOW_ON_DARK, uniforms.theme_dark);

    // Final composite: bar ink + separate halo, alpha dominated by the
    // bar with a floor from the halo so pure-halo pixels still show.
    let final_rgb = ink * bar_alpha + halo_color * halo;
    let final_alpha = max(bar_alpha, halo * HALO_ALPHA_FLOOR);

    return vec4<f32>(final_rgb, final_alpha);
}
