// switch_slider.wgsl — the preference toggle (pill + knob).
//
// WGSL port of core/utils/canvas/widgets/switch-slider/shaders.ts.
// Unlike the other widgets this shader works in RAW PIXELS, not UV —
// the original canvas was a fixed 28px-tall strip and every number is
// pixel-perfect for it. `resolution_css` carries that design size; at
// other sizes the pixel values do NOT scale (a deliberate choice — the
// toggle is a fixed-size control, not a responsive surface).
//
// Fragment-side chunk only — dist_to_segment + vs_main come from
// gpu-mini's wgsl library.
//
// Uniform layout (f32s, switch_slider.rs::UNIFORM_F32S = 12):
//   [0,1] resolution_css design size, px (track is 14px tall, radius 14)
//   [2]   time_seconds   seconds — drives wave/streak animation
//   [3]   switch_progress 0=off..1=on (eased by JS — shader just lerps)
//   [4]   knob_center_x  knob center X in px (JS owns the spring)
//   [5]   context_id     0=stats 1=grid 2=motion 3=contrast — picks
//                        track color+pattern
//   [6,7] reserved       16B-alignment pad (vec4 boundary)
//   [8..11] clip_rect    widget clip rect, canvas DEVICE px
//
// Render order inside this shader (each layer darkens/brightens the
// previous — order matters, don't reorder):
//   track capsule → context pattern → inset bezel → knob shadow →
//   knob sphere → accent dot → outer AA mask.

struct Uniforms {
    resolution_css: vec2<f32>,
    time_seconds: f32,
    switch_progress: f32,
    knob_center_x: f32,
    context_id: f32, // 0=stats, 1=grid, 2=motion, 3=contrast
    reserved0: f32,
    reserved1: f32,
    clip_rect: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// ── Tunables ─────────────────────────────────────────────────────────

// Track geometry — the pill is a 14px-radius capsule spanning
// (14,14)→(res.x−14,14); the toggle was authored at that exact size.
const TRACK_Y_PX: f32 = 14.0;
const TRACK_RADIUS: f32 = 14.0;
const TRACK_END_INSET: f32 = 14.0;
const BORDER_AA_LO: f32 = 13.2;
const BORDER_AA_HI: f32 = 14.0;
// Inset bezel — a darkening ramp in the outer 2.5px of the track that
// makes the pill read as inset glass.
const BEZEL_LO: f32 = 11.5;
const BEZEL_HI: f32 = 14.0;
const BEZEL_STRENGTH: f32 = 0.35;
const BEZEL_INK = vec3<f32>(0.0);

// Track inks — the off state plus one "on" accent per context. These
// are the same semantic accents the DOM toggles use; each context's
// pattern below tints in the same family.
const TRACK_OFF = vec3<f32>(0.18, 0.19, 0.22);
const TRACK_STATS = vec3<f32>(0.08, 0.58, 0.36);   // emerald
const TRACK_GRID = vec3<f32>(0.06, 0.48, 0.72);    // cyan blueprint
const TRACK_MOTION = vec3<f32>(0.48, 0.24, 0.68);  // violet
const TRACK_CONTRAST = vec3<f32>(0.78, 0.50, 0.08);// amber

// Context pattern inks.
const WAVE_DIM = vec3<f32>(0.35, 0.40, 0.45);   // ECG trace when off
const WAVE_LIT = vec3<f32>(0.45, 1.0, 0.65);    // ECG trace when on
const GRID_DIM = vec3<f32>(0.30, 0.35, 0.42);   // blueprint when off
const GRID_LIT = vec3<f32>(0.50, 0.85, 1.0);    // blueprint when on
const STREAK_INK = vec3<f32>(0.85, 0.70, 1.0);  // comet streaks (off)
const DATUM_INK = vec3<f32>(0.95, 0.85, 1.0);   // calm datum (on)
const WIPE_DARK = vec3<f32>(0.05, 0.05, 0.08);  // comparator dark side
const WIPE_LIGHT = vec3<f32>(0.98, 0.98, 1.0);  // comparator light side
const SEAM_INK = vec3<f32>(1.0, 0.78, 0.30);    // comparator cursor

// Stats pattern — an ECG/heartbeat trace scrolling 20px/s; base sine
// ±2.5px plus a 36px-period QRS spike window.
const ECG_SCROLL_PX_S: f32 = 20.0;
const ECG_FREQ: f32 = 0.35;
const ECG_SWELL: f32 = 2.5;
const ECG_PERIOD: f32 = 36.0;
const ECG_SPIKE_LO: f32 = 12.0;
const ECG_SPIKE_HI: f32 = 18.0;
const ECG_SPIKE_CENTER: f32 = 15.0;
const ECG_SPIKE_SLOPE: f32 = -2.8;
const ECG_OFF_DAMP: f32 = 0.3;
const ECG_ON_GAIN: f32 = 0.7;
const ECG_GLOW_WIDTH: f32 = 1.6;
const ECG_STRENGTH: f32 = 0.85;

// Grid pattern — blueprint hatch: verticals every 8px drifting 3px/s
// + fixed horizontals every 7px; fades in as the switch turns on.
const GRID_X_PERIOD: f32 = 8.0;
const GRID_X_CENTER: f32 = 4.0;
const GRID_X_SCROLL: f32 = 3.0;
const GRID_Y_PERIOD: f32 = 7.0;
const GRID_Y_CENTER: f32 = 3.5;
const GRID_LINE_SOFT: f32 = 1.0;
const GRID_OFF_ALPHA: f32 = 0.25;
const GRID_ON_GAIN: f32 = 0.55;

// Motion pattern — off: diagonal comet streaks at 6px/s fading over
// 12px of their 20px cycle; on: a single calm datum line at center.
const STREAK_PERIOD: f32 = 20.0;
const STREAK_Y_PERIOD: f32 = 7.0;
const STREAK_Y_CENTER: f32 = 3.5;
const STREAK_SCROLL: f32 = 6.0;
const STREAK_FADE: f32 = 12.0;
const STREAK_SOFT: f32 = 1.2;
const STREAK_ALPHA: f32 = 0.45;
const DATUM_SOFT: f32 = 1.2;
const DATUM_ALPHA: f32 = 0.60;

// Contrast pattern — a sweeping luminance split: a vertical seam
// travels ±17px around the 27px mark on a slow sine.
const SEAM_CENTER: f32 = 27.0;
const SEAM_TRAVEL: f32 = 17.0;
const SEAM_SPEED: f32 = 0.9;
const SEAM_SOFT: f32 = 0.9;
const SEAM_CURSOR_SOFT: f32 = 1.0;
const WIPE_OFF_ALPHA: f32 = 0.30;
const WIPE_ON_GAIN: f32 = 0.60;
const SEAM_OFF_ALPHA: f32 = 0.30;
const SEAM_ON_GAIN: f32 = 0.50;

// The knob — a fake-3D sphere from one distance field: a soft shadow
// ring past the edge, ~15% rim darkening (cheap Lambert-ish shading —
// no normals needed at this size), and a 2.7px accent dot at center.
const KNOB_RADIUS: f32 = 11.0;
const KNOB_SHADOW_WIDTH: f32 = 2.5;
const KNOB_SHADOW_STRENGTH: f32 = 0.35;
const KNOB_SHADOW_INK = vec3<f32>(0.0);
const KNOB_RIM_DARKEN: f32 = 0.15;
const KNOB_SKIN = vec3<f32>(0.98, 0.98, 1.0);
const KNOB_AA_LO: f32 = -0.6;
const KNOB_AA_HI: f32 = 0.4;
const DOT_RADIUS_LO: f32 = 2.0;
const DOT_RADIUS_HI: f32 = 2.7;
const DOT_OFF = vec3<f32>(0.6, 0.65, 0.7);
const DOT_STATS = vec3<f32>(0.1, 0.85, 0.45);
const DOT_GRID = vec3<f32>(0.1, 0.65, 0.95);
const DOT_MOTION = vec3<f32>(0.65, 0.4, 0.95);
const DOT_CONTRAST = vec3<f32>(1.0, 0.75, 0.25);

// Context thresholds — context_id is an f32 (uniform blocks carry no
// ints); the midpoint steps bucket it back to its enum.
const CTX_STATS_MAX: f32 = 0.5;
const CTX_GRID_MAX: f32 = 1.5;
const CTX_MOTION_MAX: f32 = 2.5;

@fragment
fn fs_main(@builtin(position) pixel_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // The original ran in a dedicated 28px-tall canvas; normalize into
    // the widget rect and continue in its own px space. WGSL
    // `pixel_coord` is framebuffer-origin (y=0 at TOP) while the ported
    // GLSL math assumes gl_FragCoord/v_uv semantics (y=0 at BOTTOM) —
    // widget_uv.y is flipped here so every constant below measures
    // from the bottom edge exactly like the original shader did.
    let widget_uv = vec2<f32>(
        (pixel_coord.x - uniforms.clip_rect.x) / uniforms.clip_rect.z,
        1.0 - (pixel_coord.y - uniforms.clip_rect.y) / uniforms.clip_rect.w,
    );
    let pixel = widget_uv * uniforms.resolution_css;

    let track_start = vec2<f32>(TRACK_END_INSET, TRACK_Y_PX);
    let track_end = vec2<f32>(uniforms.resolution_css.x - TRACK_END_INSET, TRACK_Y_PX);
    let track_dist = dist_to_segment(pixel, track_start, track_end);

    if (track_dist > TRACK_RADIUS) {
        discard;
    }

    let border_alpha = 1.0 - smoothstep(BORDER_AA_LO, BORDER_AA_HI, track_dist);

    var track_ink = TRACK_OFF;

    if (uniforms.context_id < CTX_STATS_MAX) {
        track_ink = TRACK_STATS;
    } else if (uniforms.context_id < CTX_GRID_MAX) {
        track_ink = TRACK_GRID;
    } else if (uniforms.context_id < CTX_MOTION_MAX) {
        track_ink = TRACK_MOTION;
    } else {
        track_ink = TRACK_CONTRAST;
    }

    var color = mix(TRACK_OFF, track_ink, uniforms.switch_progress);

    if (uniforms.context_id < CTX_STATS_MAX) {
        // STATS — an ECG/heartbeat trace. wave_x scrolls the trace
        // right-to-left; the base sine gives the swell; the ECG_PERIOD
        // window adds the sharp spike (a triangular slope centered on
        // ECG_SPIKE_CENTER — the classic QRS complex shape). The
        // progress term damps the trace to 30% when off — heart
        // barely beating.
        let wave_x = pixel.x + uniforms.time_seconds * ECG_SCROLL_PX_S;
        var pulse = sin(wave_x * ECG_FREQ) * ECG_SWELL;

        let spike_phase = wave_x % ECG_PERIOD;
        if (spike_phase > ECG_SPIKE_LO && spike_phase < ECG_SPIKE_HI) {
            pulse = pulse + (spike_phase - ECG_SPIKE_CENTER) * ECG_SPIKE_SLOPE;
        }

        let wave_y = TRACK_Y_PX + pulse * (ECG_OFF_DAMP + ECG_ON_GAIN * uniforms.switch_progress);
        let wave_dist = abs(pixel.y - wave_y);
        let wave_glow = 1.0 - smoothstep(0.0, ECG_GLOW_WIDTH, wave_dist);
        let wave_ink = mix(WAVE_DIM, WAVE_LIT, uniforms.switch_progress);
        color = color + wave_ink * wave_glow * ECG_STRENGTH;
    } else if (uniforms.context_id < CTX_GRID_MAX) {
        // GRID — blueprint hatch: vertical lines drifting + fixed
        // horizontals. Each `abs(mod − center)` gives distance-to-
        // nearest-line; the smoothstep turns it into a soft line mask;
        // adding X+Y makes a crosshatch. The blueprint fades IN as the
        // toggle turns on — barely visible when off.
        let grid_x = (pixel.x + uniforms.time_seconds * GRID_X_SCROLL) % GRID_X_PERIOD;
        let grid_y = pixel.y % GRID_Y_PERIOD;
        let is_line_x = 1.0 - smoothstep(0.0, GRID_LINE_SOFT, abs(grid_x - GRID_X_CENTER));
        let is_line_y = 1.0 - smoothstep(0.0, GRID_LINE_SOFT, abs(grid_y - GRID_Y_CENTER));
        let grid_pattern = clamp(is_line_x + is_line_y, 0.0, 1.0);
        let grid_ink = mix(GRID_DIM, GRID_LIT, uniforms.switch_progress);
        color = color + grid_ink * grid_pattern * (GRID_OFF_ALPHA + GRID_ON_GAIN * uniforms.switch_progress);
    } else if (uniforms.context_id < CTX_MOTION_MAX) {
        // MOTION — when OFF (reduced-motion): diagonal comet streaks
        // fading over their cycle. When ON: a single calm datum line —
        // the reduced-motion setting literally reduces the motion
        // rendered.
        if (uniforms.switch_progress < 0.5) {
            let streak_y = pixel.y % STREAK_Y_PERIOD;
            let streak_move = (pixel.x - uniforms.time_seconds * STREAK_SCROLL) % STREAK_PERIOD;
            let streak_line = 1.0 - smoothstep(0.0, STREAK_SOFT, abs(streak_y - STREAK_Y_CENTER));
            let streak_fade = 1.0 - smoothstep(0.0, STREAK_FADE, streak_move);
            color = color + STREAK_INK * streak_line * streak_fade * STREAK_ALPHA;
        } else {
            let datum_line = 1.0 - smoothstep(0.0, DATUM_SOFT, abs(pixel.y - TRACK_Y_PX));
            color = color + DATUM_INK * datum_line * DATUM_ALPHA;
        }
    } else {
        // CONTRAST — sweeping luminance split ("comparison wipe"): a
        // vertical seam travels the track on a slow sine; left of it
        // near-black, right near-white — the before/after split of a
        // contrast comparator. The split fades IN as the toggle turns
        // on — a soft grey hint when off. The seam is the comparator's
        // cursor riding the boundary.
        let seam_x = SEAM_CENTER + sin(uniforms.time_seconds * SEAM_SPEED) * SEAM_TRAVEL;
        let wipe_side = smoothstep(seam_x - SEAM_SOFT, seam_x + SEAM_SOFT, pixel.x);
        let wipe_ink = mix(WIPE_DARK, WIPE_LIGHT, wipe_side);
        color = mix(color, wipe_ink, WIPE_OFF_ALPHA + WIPE_ON_GAIN * uniforms.switch_progress);

        let seam_cursor = 1.0 - smoothstep(0.0, SEAM_CURSOR_SOFT, abs(pixel.x - seam_x));
        color = color + SEAM_INK * seam_cursor * (SEAM_OFF_ALPHA + SEAM_ON_GAIN * uniforms.switch_progress);
    }

    // Inset bezel — the darkening ramp in the outer band of the track
    // that makes the pill read as inset glass.
    let bezel = smoothstep(BEZEL_LO, BEZEL_HI, track_dist);
    color = mix(color, BEZEL_INK, bezel * BEZEL_STRENGTH);

    // ── The knob ─────────────────────────────────────────────────────
    let knob_center = vec2<f32>(uniforms.knob_center_x, TRACK_Y_PX);
    let knob_dist = length(pixel - knob_center);

    let knob_shadow = (1.0 - smoothstep(KNOB_RADIUS, KNOB_RADIUS + KNOB_SHADOW_WIDTH, knob_dist)) * KNOB_SHADOW_STRENGTH;
    color = mix(color, KNOB_SHADOW_INK, knob_shadow);

    let sphere_shade = 1.0 - clamp(knob_dist / KNOB_RADIUS, 0.0, 1.0) * KNOB_RIM_DARKEN;
    var knob_color = KNOB_SKIN * sphere_shade;

    let dot_dist = length(pixel - knob_center);
    let dot_light = 1.0 - smoothstep(DOT_RADIUS_LO, DOT_RADIUS_HI, dot_dist);

    var dot_ink: vec3<f32>;
    if (uniforms.context_id < CTX_STATS_MAX) {
        dot_ink = mix(DOT_OFF, DOT_STATS, uniforms.switch_progress);
    } else if (uniforms.context_id < CTX_GRID_MAX) {
        dot_ink = mix(DOT_OFF, DOT_GRID, uniforms.switch_progress);
    } else if (uniforms.context_id < CTX_MOTION_MAX) {
        dot_ink = mix(DOT_OFF, DOT_MOTION, uniforms.switch_progress);
    } else {
        dot_ink = mix(DOT_OFF, DOT_CONTRAST, uniforms.switch_progress);
    }

    knob_color = mix(knob_color, dot_ink, dot_light);

    let knob_alpha = 1.0 - smoothstep(KNOB_RADIUS + KNOB_AA_LO, KNOB_RADIUS + KNOB_AA_HI, knob_dist);
    color = mix(color, knob_color, knob_alpha);

    return vec4<f32>(color * border_alpha, border_alpha);
}
