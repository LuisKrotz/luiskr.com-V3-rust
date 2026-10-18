// menu_background.wgsl — the fullscreen menu's flowing contour field.
//
// WGSL port of core/utils/canvas/loaders/menu-background/shaders.ts.
// The visual is topographic isolines over a DOUBLE-DOMAIN-WARPED fbm
// field — the "Iris van Herpen fabric" look: contour lines that bend
// organically instead of drawing geometric shapes.
//
// Uniform layout (f32s, menu_background.rs::UNIFORM_F32S = 16):
//   [0]   time      seconds (internally scaled ×0.018 — ~55s drift)
//   [1,2] res       widget px size
//   [3]   reveal    0..1 menu open amount (JS eases)
//   [4..6] color    primary ink rgb (theme token)
//   [7]   alpha     global opacity multiplier
//   [8..10] color2  secondary ink rgb — field value picks between inks
//   [11]  pad       16B alignment pad
//   [12..15] rect   widget clip rect, canvas CSS px
//
// COST NOTE: fbm is 4 octaves of value noise → each octave = 4 hash
// evaluations; we call fbm 5× per pixel (q×2, r×2, f) = 80 hashes/px.
// On a 4K screen that's ~660M ops/frame — expensive, but the widget
// only exists while the menu is open, and fullscreen shaders are the
// GPU's home turf. Halving octaves (4→3) cuts ~25% cost at slight
// quality loss — the safe lever if a low-end GPU ever struggles.

struct Uniforms {
    time: f32,
    res: vec2<f32>,
    reveal: f32,
    color: vec3<f32>,
    alpha: f32,
    color2: vec3<f32>,
    pad: f32,
    rect: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

// hash(p): deterministic cell random — the sin·fract hash with a
// different constant pair (127.1, 311.7) than theme_slider's, so the
// two fields don't correlate if they ever coexist.
fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

// noise(p): 2D VALUE NOISE — hash the 4 cell corners, bilinear mix.
// The smoothstep-polynomial `f*f*(3-2f)` (Perlin's "fade" curve) makes
// the interpolation C¹-continuous — derivatives match at cell borders,
// which is what keeps contours smooth instead of faceted. This is THE
// textbook construction; the numbers are Perlin's originals.
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i), hash(i + vec2<f32>(1.0, 0.0)), f.x),
        mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), f.x),
        f.y,
    );
}

// fbm(p): fractal Brownian motion — 4 octaves of noise, each at ~2×
// frequency and ½ amplitude (lacunarity 2.03, gain 0.5 — classic
// "musgrave" params; 2.03 is deliberately off-2 to avoid octave
// alignment artifacts). The rotation matrix m (0.8,0.6,-0.6,0.8 —
// a rotation+scale hybrid) de-axes each octave so contours don't
// visibly grid-align. Output range ≈ [0, ~0.94) — sums a geometric
// series 0.5+0.25+0.125+0.0625 ≈ 0.94 max.
fn fbm(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var v = 0.0;
    var a = 0.5;
    let m = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    for (var i = 0; i < 4; i = i + 1) {
        v = v + a * noise(p);
        p = m * p * 2.03;
        a = a * 0.5;
    }
    return v;
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // Centered, aspect-corrected coords — same normalization as the
    // original (u_res there was the canvas px size; here the rect).
    // min(res) normalization keeps the field isotropic (circles stay
    // circles) on any viewport aspect.
    let res = vec2<f32>(u.rect.z, u.rect.w);
    let p = (vec2<f32>(frag.x - u.rect.x, frag.y - u.rect.y) - 0.5 * res) / min(res.x, res.y);
    let t = u.time * 0.018;

    // ── Double domain-warp (the "why it looks like fabric") ──────────
    // q = a 2D fbm displacement of p. r = a SECOND fbm displacement
    // whose input is already warped by q×2.2 — warping a warp is what
    // turns grid-ish noise into flowing cloth folds (IQ's "domain
    // warping" technique). f = the final field, warped by r×2.0.
    // The ±t terms drift the warp vectors at 0.018× time (~55s cycle)
    // — glacial on purpose: the menu backdrop should feel alive, never
    // animated. Constants 1.7/9.2/8.3/2.8 are just non-aligned offsets
    // that decorrelate the fbm inputs (any distinct values work — don't
    // "round" them, alignment creates visible periodicity).
    let q = vec2<f32>(fbm(p * 1.6 + vec2<f32>(t)), fbm(p * 1.6 + vec2<f32>(-t * 0.7 + 3.1)));
    let r = vec2<f32>(
        fbm(p * 1.6 + 2.2 * q + vec2<f32>(1.7, 9.2) + vec2<f32>(t * 0.6)),
        fbm(p * 1.6 + 2.2 * q + vec2<f32>(8.3, 2.8) + vec2<f32>(-t * 0.4)),
    );
    let f = fbm(p * 1.6 + 2.0 * r);

    // ── Isolines — the contour-line renderer ─────────────────────────
    // `band = f * 14`: the field value scaled into 14 level bands.
    // `v = fract(band)` is distance INSIDE a band; `d = min(v, 1-v)` is
    // distance to the nearest contour line. The critical trick:
    // `dpx = d / fwidth(band)` — dividing by the screen-space gradient
    // converts "distance in band units" to "distance in PIXELS", so
    // every contour is a uniform ~1px hairline whether the field is
    // steep or flat (a fixed band-width threshold would produce
    // lines of varying thickness + moiré).
    let levels = 14.0;
    let band = f * levels;
    let v = fract(band);
    let d = min(v, 1.0 - v);
    let grad = max(fwidth(band), 1e-4);
    let dpx = d / grad;
    var line = 1.0 - smoothstep(0.45, 1.05, dpx);
    // Every 6th band is a "major" contour (thicker, dimmer than full
    // line) — step(mod,0.5) selects them; mix picks the stronger mask.
    let majorLine = 1.0 - smoothstep(0.7, 1.5, dpx);
    let major = step((floor(band) % 6.0), 0.5);
    line = mix(line, max(line, majorLine * 0.9), major);

    // Vignette concentrates the field near the link column + reveal
    // gates the whole effect behind the menu's open animation.
    let len = length(p);
    let vign = 1.0 - smoothstep(0.15, 0.7, len);
    let rev = smoothstep(0.0, 1.0, u.reveal);
    let a = line * vign * rev * u.alpha;

    // Ink drifts between the two theme inks across the field.
    let wm = smoothstep(0.2, 0.8, f);
    let ink = mix(u.color, u.color2, wm);

    return vec4<f32>(ink, a);
}
