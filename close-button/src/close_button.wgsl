// close_button.wgsl — the modal close button's liquid-fill hover.
//
// WGSL port of core/utils/canvas/widgets/close-button/shaders.ts.
// On hover a turquoise liquid rises inside the circular button with a
// wobbling foam crest and rising bubbles; the X strokes self-draw and
// slowly rotate; a tap emits a shockwave ring. All math is in PIXEL
// space relative to the button center (point_rel) — the button is
// circular, so pixel space is the natural domain (radius clipping).
//
// Fragment-side chunk only — dist_to_segment + the disc/palette tokens
// + the vs_main triangle come from gpu-mini's wgsl library.
//
// Uniform layout (f32s, close_button.rs::UNIFORM_F32S = 12):
//   [0,1] resolution_css design size px (button diameter)
//   [2]   time_seconds   seconds — wave phase, bubbles, shockwave clock
//   [3]   liquid_fill    0..1 fill height (JS eases on hover)
//   [4]   x_draw         0..1 stroke progress for the self-drawing X
//   [5]   x_rotation     X rotation angle, rad (JS advances it slowly)
//   [6]   click_seconds  seconds-stamp of last click (−1 = none)
//   [7]   reserved       16B alignment pad
//   [8..11] clip_rect    widget clip rect, canvas DEVICE px

struct Uniforms {
    resolution_css: vec2<f32>,
    time_seconds: f32,
    liquid_fill: f32,
    x_draw: f32,
    x_rotation: f32,
    click_seconds: f32,
    reserved: f32,
    clip_rect: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// ── Tunables ─────────────────────────────────────────────────────────

// Widget colors — the liquid palette is unique to this widget (the
// turquoise pour reads "fill progress" anywhere it appears).
const LIQUID_SURFACE = vec3<f32>(0.40, 0.988, 0.945); // bright surface teal
const LIQUID_DEEP = vec3<f32>(0.18, 0.65, 0.62);      // absorbed-with-depth teal
const FOAM_TINT = vec3<f32>(0.75, 1.0, 0.98);         // crest glow
const BUBBLE_TINT = vec3<f32>(0.85, 1.0, 0.98);       // ring + specular
const STROKE_INK = vec3<f32>(1.0);                    // the X strokes

// Fill sweep: liquid_fill∈[0,1] maps to [−1.1,+1.1]×radius — the ±10%
// overshoot lets the pour fully cover (1.1) and fully clear (−1.1) the
// circle instead of stopping exactly at the rim.
const FILL_OVERSHOOT: f32 = 1.1;

// Crest wave — two detuned sines: detuned spatial freqs (0.28 & 0.45
// rad/px) and temporal freqs (4.0 & 2.8 rad/s) prevent any visible
// periodicity. WAVE_RAMP keeps the surface flat until the fill is >40%
// risen — a wavy crest on an empty button reads as broken.
const CREST_FREQ_A: f32 = 0.28;
const CREST_SPEED_A: f32 = 4.0;
const CREST_AMP_A: f32 = 1.6;
const CREST_FREQ_B: f32 = 0.45;
const CREST_SPEED_B: f32 = 2.8;
const CREST_AMP_B: f32 = 1.0;
const CREST_WAVE_RAMP: f32 = 2.5;
const CREST_WIDTH_PX: f32 = 2.0;   // additive glow band hugging the line
const CREST_STRENGTH: f32 = 0.75;

// Bubbles — four procedural rings rising at staggered speeds, x-jitter
// by a slow sine, y wrapping on a 2.2×radius recycle cycle. Each is a
// RING (inner × outer smoothstep shell) plus an offset specular spot.
const BUBBLE_COUNT: u32 = 4u;
const BUBBLE_SPEED_BASE: f32 = 12.0; // px/s for bubble 0
const BUBBLE_SPEED_STEP: f32 = 4.0;  // +px/s per bubble index
const BUBBLE_X_BASE: f32 = -8.0;
const BUBBLE_X_STEP: f32 = 5.5;
const BUBBLE_JITTER_SPEED: f32 = 2.5;
const BUBBLE_JITTER_PHASE: f32 = 1.7;
const BUBBLE_JITTER_AMP: f32 = 2.5;
const BUBBLE_PHASE_STEP: f32 = 9.0;
const BUBBLE_CYCLE_RADII: f32 = 2.2;
const BUBBLE_RADIUS_BASE: f32 = 1.2;
const BUBBLE_RADIUS_ALT: f32 = 0.7;  // even/odd alternate size bump
const BUBBLE_RING_SOFT: f32 = 0.4;
const BUBBLE_SHELL_INNER: f32 = 1.2;
const BUBBLE_SHELL_OUTER: f32 = 0.3;
const BUBBLE_SPOT_OFFSET: f32 = 0.4;
const BUBBLE_SPOT_RADIUS: f32 = 0.6;
const BUBBLE_RING_STRENGTH: f32 = 0.55;
const BUBBLE_SPOT_STRENGTH: f32 = 0.75;
const LIQUID_EPSILON: f32 = 0.01;    // below this the pour is invisible

// Chrome — the disc's own shading (DISC_SLATE comes from the palette).
const BORDER_AA_PX: f32 = 1.0;   // edge fade band
const RIM_WIDTH_PX: f32 = 3.0;   // inset rim shadow band
const RIM_DARKEN: f32 = 0.35;    // shadow strength at the rim
const RIM_FACTOR: f32 = 0.7;     // color multiplier at full shadow

// The X — reach 0.38·radius keeps it inside the fill circle with a
// comfortable margin; ~0.8px AA via the smoothstep band.
const X_REACH_FRAC: f32 = 0.38;
const STROKE_AA_LO: f32 = 0.9;
const STROKE_AA_HI: f32 = 1.7;
const DRAW_EPSILON: f32 = 0.01; // strokes stay ink-free until draw starts

// Click shockwave — a 2.5px ring expanding at 45px/s, alpha decaying
// over 0.4s (≈18px travel — most of the button's radius).
const SHOCK_LIFETIME_S: f32 = 0.4;
const SHOCK_SPEED_PX_S: f32 = 45.0;
const SHOCK_WIDTH_PX: f32 = 2.5;
const SHOCK_STRENGTH: f32 = 0.8;

@fragment
fn fs_main(@builtin(position) pixel_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // widget_uv.y flipped: WGSL `pixel_coord` is framebuffer-origin
    // (y=0 TOP) while the ported GLSL used v_uv/gl_FragCoord (y=0
    // BOTTOM) — the fill sweep direction and lighting terms assume
    // bottom-origin.
    let widget_uv = vec2<f32>(
        (pixel_coord.x - uniforms.clip_rect.x) / uniforms.clip_rect.z,
        1.0 - (pixel_coord.y - uniforms.clip_rect.y) / uniforms.clip_rect.w,
    );
    let pixel = widget_uv * uniforms.resolution_css;
    let center = uniforms.resolution_css * 0.5;
    let point_rel = pixel - center;
    let radius = min(center.x, center.y) - BORDER_AA_PX;
    let center_dist = length(point_rel);

    if (center_dist > radius) {
        discard;
    }

    let border_alpha = 1.0 - smoothstep(radius - BORDER_AA_PX, radius, center_dist);

    var color = DISC_SLATE;

    // ── Liquid fill ──────────────────────────────────────────────────
    let fill_level = (uniforms.liquid_fill * 2.0 * FILL_OVERSHOOT - FILL_OVERSHOOT) * radius;
    let crest_wave = sin(point_rel.x * CREST_FREQ_A + uniforms.time_seconds * CREST_SPEED_A)
            * CREST_AMP_A * min(uniforms.liquid_fill * CREST_WAVE_RAMP, 1.0)
        + cos(point_rel.x * CREST_FREQ_B - uniforms.time_seconds * CREST_SPEED_B)
            * CREST_AMP_B * min(uniforms.liquid_fill * CREST_WAVE_RAMP, 1.0);
    let surface_y = fill_level + crest_wave;

    if (point_rel.y <= surface_y && uniforms.liquid_fill > LIQUID_EPSILON) {
        // Depth tint: normalized depth below the crest lerps bright
        // surface-teal → deep teal (light absorbed with depth).
        let depth = clamp((surface_y - point_rel.y) / (radius * 2.0), 0.0, 1.0);
        var liquid_color = mix(LIQUID_SURFACE, LIQUID_DEEP, depth);

        // Foam crest — a 2px additive glow hugging the surface line.
        let crest_dist = abs(point_rel.y - surface_y);
        let crest = 1.0 - smoothstep(0.0, CREST_WIDTH_PX, crest_dist);
        liquid_color = liquid_color + FOAM_TINT * crest * CREST_STRENGTH;

        // Bubbles — rings rising at staggered speeds, x-jittered by a
        // slow sine; y wraps on the recycle cycle so bubbles recycle
        // continuously. Only bubbles below the crest and inside the
        // button paint (`bubble_y <= surface_y`).
        for (var index = 0u; index < BUBBLE_COUNT; index = index + 1u) {
            let bubble_i = f32(index);
            let bubble_speed = BUBBLE_SPEED_BASE + bubble_i * BUBBLE_SPEED_STEP;
            let bubble_x = BUBBLE_X_BASE + bubble_i * BUBBLE_X_STEP
                + sin(uniforms.time_seconds * BUBBLE_JITTER_SPEED + bubble_i * BUBBLE_JITTER_PHASE) * BUBBLE_JITTER_AMP;
            let bubble_y = (uniforms.time_seconds * bubble_speed + bubble_i * BUBBLE_PHASE_STEP)
                % (radius * BUBBLE_CYCLE_RADII) - radius;
            let bubble_pos = vec2<f32>(bubble_x, bubble_y);
            let bubble_radius = BUBBLE_RADIUS_BASE + (bubble_i % 2.0) * BUBBLE_RADIUS_ALT;

            if (bubble_y <= surface_y && bubble_y > -radius) {
                let bubble_dist = length(point_rel - bubble_pos);
                let bubble_ring = (1.0 - smoothstep(bubble_radius - BUBBLE_RING_SOFT, bubble_radius + BUBBLE_RING_SOFT, bubble_dist))
                    * smoothstep(bubble_radius - BUBBLE_SHELL_INNER, bubble_radius - BUBBLE_SHELL_OUTER, bubble_dist);
                let bubble_spot = 1.0 - smoothstep(
                    0.0,
                    BUBBLE_SPOT_RADIUS,
                    length(point_rel - (bubble_pos + vec2<f32>(BUBBLE_SPOT_OFFSET))),
                );
                liquid_color = liquid_color
                    + BUBBLE_TINT * (bubble_ring * BUBBLE_RING_STRENGTH + bubble_spot * BUBBLE_SPOT_STRENGTH);
            }
        }

        color = mix(color, liquid_color, border_alpha);
    }

    // Inset rim shadow.
    let rim_shadow = smoothstep(radius - RIM_WIDTH_PX, radius, center_dist);
    color = mix(color, color * RIM_FACTOR, rim_shadow * RIM_DARKEN);

    // Rotating X strokes — the sample point is rotated by −x_rotation
    // (rotate the POINT, not the strokes — same SDF convention as the
    // burger). Each stroke is a segment whose B endpoint lerp-grows
    // with `x_draw` — that's the self-drawing effect: distance to a
    // partial-length segment only inks the drawn part.
    let point_rot = vec2<f32>(
        point_rel.x * cos(uniforms.x_rotation) - point_rel.y * sin(uniforms.x_rotation),
        point_rel.x * sin(uniforms.x_rotation) + point_rel.y * cos(uniforms.x_rotation),
    );

    let stroke_half_len = radius * X_REACH_FRAC;
    let draw_progress = clamp(uniforms.x_draw, 0.0, 1.0);

    let stroke_a_start = vec2<f32>(-stroke_half_len, -stroke_half_len);
    let stroke_a_end = stroke_a_start + vec2<f32>(stroke_half_len * 2.0) * draw_progress;
    let dist_stroke_a = dist_to_segment(point_rot, stroke_a_start, stroke_a_end);

    let stroke_b_start = vec2<f32>(-stroke_half_len, stroke_half_len);
    let stroke_b_end = stroke_b_start + vec2<f32>(stroke_half_len * 2.0, -stroke_half_len * 2.0) * draw_progress;
    let dist_stroke_b = dist_to_segment(point_rot, stroke_b_start, stroke_b_end);

    let stroke_a = (1.0 - smoothstep(STROKE_AA_LO, STROKE_AA_HI, dist_stroke_a)) * step(DRAW_EPSILON, draw_progress);
    let stroke_b = (1.0 - smoothstep(STROKE_AA_LO, STROKE_AA_HI, dist_stroke_b)) * step(DRAW_EPSILON, draw_progress);
    let x_ink = clamp(stroke_a + stroke_b, 0.0, 1.0);

    color = mix(color, STROKE_INK, x_ink * border_alpha);

    // Click shockwave — a ring expanding at SHOCK_SPEED_PX_S, alpha
    // decaying to zero over SHOCK_LIFETIME_S. click_seconds = −1 keeps
    // it permanently inert.
    let click_elapsed = uniforms.time_seconds - uniforms.click_seconds;
    if (click_elapsed >= 0.0 && click_elapsed < SHOCK_LIFETIME_S) {
        let shock_radius = click_elapsed * SHOCK_SPEED_PX_S;
        let shock_ring = (1.0 - smoothstep(0.0, SHOCK_WIDTH_PX, abs(center_dist - shock_radius)))
            * (1.0 - click_elapsed / SHOCK_LIFETIME_S);
        color = color + LIQUID_SURFACE * shock_ring * SHOCK_STRENGTH;
    }

    return vec4<f32>(color, border_alpha);
}
