// menu_background.wgsl — the fullscreen menu's flowing contour field.
//
// WGSL port of core/utils/canvas/loaders/menu-background/shaders.ts.
// The visual is topographic isolines over a DOUBLE-DOMAIN-WARPED fbm
// field — the "Iris van Herpen fabric" look: contour lines that bend
// organically instead of drawing geometric shapes.
//
// Fragment-side chunk only — hash/value_noise + the vs_main triangle
// come from gpu-mini's wgsl library.
//
// Uniform layout (f32s, menu_background.rs::UNIFORM_F32S = 16):
//   [0]   time_seconds  seconds (internally scaled ×DRIFT_SPEED — ~55s)
//   [1,2] resolution_css widget px size
//   [3]   menu_reveal   0..1 menu open amount (JS eases)
//   [4..6] ink_primary  primary ink rgb (theme token)
//   [7]   global_alpha  global opacity multiplier
//   [8..10] ink_secondary secondary ink rgb — field value picks inks
//   [11]  reserved      16B alignment pad
//   [12..15] clip_rect  widget clip rect, canvas DEVICE px
//
// COST NOTE: fbm is 4 octaves of value noise → each octave = 4 hash
// evaluations; we call fbm 5× per pixel (warp_first×2, warp_second×2,
// field) = 80 hashes/px. On a 4K screen that's ~660M ops/frame —
// expensive, but the widget only exists while the menu is open, and
// fullscreen shaders are the GPU's home turf. Halving octaves (4→3)
// cuts ~25% cost at slight quality loss — the safe lever if a low-end
// GPU ever struggles.

struct Uniforms {
    time_seconds: f32,
    resolution_css: vec2<f32>,
    menu_reveal: f32,
    ink_primary: vec3<f32>,
    global_alpha: f32,
    ink_secondary: vec3<f32>,
    reserved: f32,
    clip_rect: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// ── Tunables ─────────────────────────────────────────────────────────

// Drift — the warp vectors drift at 0.018× real time (~55s cycle):
// glacial on purpose, the menu backdrop should feel alive, never
// animated. The per-axis multipliers keep the drift vectors from
// synchronizing into visible loops.
const DRIFT_SPEED: f32 = 0.018;
const DRIFT_FIRST_X: f32 = 1.0;
const DRIFT_FIRST_Y: f32 = -0.7;
const DRIFT_SECOND_X: f32 = 0.6;
const DRIFT_SECOND_Y: f32 = -0.4;
// Non-aligned input offsets — they decorrelate the fbm taps (any
// distinct values work; do NOT "round" them — alignment creates
// visible periodicity).
const DECORR_FIRST: f32 = 3.1;
const DECORR_SECOND_A = vec2<f32>(1.7, 9.2);
const DECORR_SECOND_B = vec2<f32>(8.3, 2.8);

// The fbm construction: 4 octaves, each at ~2× frequency and ½
// amplitude (lacunarity 2.03, gain 0.5 — classic musgrave params;
// 2.03 is deliberately off-2 to avoid octave alignment artifacts).
// OCTAVE_ROTATE (0.8,0.6,-0.6,0.8 — a rotation+scale hybrid) de-axes
// each octave so contours don't visibly grid-align. Output range ≈
// [0, ~0.94) — sums the geometric series 0.5+0.25+0.125+0.0625.
const FBM_OCTAVES: u32 = 4u;
const FBM_GAIN: f32 = 0.5;
const FBM_LACUNARITY: f32 = 2.03;
const FBM_FREQ: f32 = 1.6;
const OCTAVE_ROTATE = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
const WARP_FIRST_SCALE: f32 = 2.2;  // how strongly q displaces r's input
const WARP_SECOND_SCALE: f32 = 2.0; // how strongly r displaces f's input

// Isoline renderer — the field value scaled into LEVELS bands; a
// contour is a ~1px hairline at every band boundary. Major contours
// (every MAJOR_EVERY-th band) get the wider/dimmer stroke.
const LEVELS: f32 = 14.0;
const LINE_INNER_PX: f32 = 0.45;
const LINE_OUTER_PX: f32 = 1.05;
const MAJOR_INNER_PX: f32 = 0.7;
const MAJOR_OUTER_PX: f32 = 1.5;
const MAJOR_EVERY: f32 = 6.0;
const MAJOR_STRENGTH: f32 = 0.9;
const MIN_BAND_GRADIENT: f32 = 1e-4;

// Vignette — concentrates the field near the link column.
const VIGNETTE_INNER: f32 = 0.15;
const VIGNETTE_OUTER: f32 = 0.7;

// Ink drift — where in the field range the two inks crossfade.
const INK_MIX_LO: f32 = 0.2;
const INK_MIX_HI: f32 = 0.8;

// fbm_field — fractal Brownian motion over the shared value_noise:
// FBM_OCTAVES octaves of noise, each rotated+scaled by OCTAVE_ROTATE·
// FBM_LACUNARITY and halved in amplitude. The rotation de-axes the
// octaves; the geometric amplitude sum bounds output to ≈0.94.
fn fbm_field(point: vec2<f32>) -> f32 {
    var sample_point = point;
    var sum = 0.0;
    var amplitude = FBM_GAIN;

    for (var octave = 0u; octave < FBM_OCTAVES; octave = octave + 1u) {
        sum = sum + amplitude * value_noise(sample_point);
        sample_point = OCTAVE_ROTATE * sample_point * FBM_LACUNARITY;
        amplitude = amplitude * FBM_GAIN;
    }

    return sum;
}

@fragment
fn fs_main(@builtin(position) pixel_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // Centered, aspect-corrected coords — same normalization as the
    // original (u_res there was the canvas px size; here the rect).
    // min(res) normalization keeps the field isotropic (circles stay
    // circles) on any viewport aspect.
    let resolution = vec2<f32>(uniforms.clip_rect.z, uniforms.clip_rect.w);
    // y is flipped: WGSL `pixel_coord` is framebuffer-origin (y=0 TOP)
    // but the ported GLSL used gl_FragCoord (y=0 BOTTOM) — without the
    // flip the warp drift and reveal wipe directions run mirrored.
    let norm_coord = (vec2<f32>(
        pixel_coord.x - uniforms.clip_rect.x,
        uniforms.clip_rect.w - (pixel_coord.y - uniforms.clip_rect.y),
    ) - 0.5 * resolution) / min(resolution.x, resolution.y);
    let drift_time = uniforms.time_seconds * DRIFT_SPEED;

    // ── Double domain-warp (the "why it looks like fabric") ──────────
    // warp_first = a 2D fbm displacement of norm_coord. warp_second = a
    // SECOND fbm displacement whose input is already warped by
    // warp_first×WARP_FIRST_SCALE — warping a warp is what turns
    // grid-ish noise into flowing cloth folds (IQ's "domain warping"
    // technique). field_value = the final field, warped by
    // warp_second×WARP_SECOND_SCALE.
    let warp_first = vec2<f32>(
        fbm_field(norm_coord * FBM_FREQ + vec2<f32>(drift_time * DRIFT_FIRST_X)),
        fbm_field(norm_coord * FBM_FREQ + vec2<f32>(drift_time * DRIFT_FIRST_Y + DECORR_FIRST)),
    );
    let warp_second = vec2<f32>(
        fbm_field(norm_coord * FBM_FREQ + WARP_FIRST_SCALE * warp_first + DECORR_SECOND_A + vec2<f32>(drift_time * DRIFT_SECOND_X)),
        fbm_field(norm_coord * FBM_FREQ + WARP_FIRST_SCALE * warp_first + DECORR_SECOND_B + vec2<f32>(drift_time * DRIFT_SECOND_Y)),
    );
    let field_value = fbm_field(norm_coord * FBM_FREQ + WARP_SECOND_SCALE * warp_second);

    // ── Isolines — the contour-line renderer ─────────────────────────
    // band = field_value × LEVELS: the field scaled into level bands.
    // band_fract is distance INSIDE a band; line_dist is distance to
    // the nearest contour line. The critical trick: line_dist_px =
    // line_dist / fwidth(band) — dividing by the screen-space gradient
    // converts "distance in band units" to "distance in PIXELS", so
    // every contour is a uniform ~1px hairline whether the field is
    // steep or flat (a fixed band-width threshold would produce lines
    // of varying thickness + moiré).
    let band = field_value * LEVELS;
    let band_fract = fract(band);
    let line_dist = min(band_fract, 1.0 - band_fract);
    let band_gradient = max(fwidth(band), MIN_BAND_GRADIENT);
    let line_dist_px = line_dist / band_gradient;
    var line_mask = 1.0 - smoothstep(LINE_INNER_PX, LINE_OUTER_PX, line_dist_px);
    // Every MAJOR_EVERY-th band is a "major" contour (thicker, dimmer
    // than full line) — the modulo selects them; mix picks the
    // stronger mask.
    let major_line = 1.0 - smoothstep(MAJOR_INNER_PX, MAJOR_OUTER_PX, line_dist_px);
    let is_major = step((floor(band) % MAJOR_EVERY), 0.5);
    line_mask = mix(line_mask, max(line_mask, major_line * MAJOR_STRENGTH), is_major);

    // Vignette concentrates the field near the link column + the
    // reveal gate folds the whole effect behind the menu's open anim.
    let center_dist = length(norm_coord);
    let vignette = 1.0 - smoothstep(VIGNETTE_INNER, VIGNETTE_OUTER, center_dist);
    let reveal_eased = smoothstep(0.0, 1.0, uniforms.menu_reveal);
    let alpha = line_mask * vignette * reveal_eased * uniforms.global_alpha;

    // Ink drifts between the two theme inks across the field.
    let ink_mix_t = smoothstep(INK_MIX_LO, INK_MIX_HI, field_value);
    let ink = mix(uniforms.ink_primary, uniforms.ink_secondary, ink_mix_t);

    return vec4<f32>(ink, alpha);
}
