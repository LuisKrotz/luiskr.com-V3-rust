// glitch.wgsl — the about-portrait VHS corruption pass.
//
// One fullscreen textured draw replaces the whole 2D pipeline:
// `fillRect` grain ×460/frame, `drawImage` channel ghosts, band
// re-blits, scanline pattern fills — all become per-fragment texture
// lookups hashed from a per-frame seed. Tunables mirror
// `website/components/home/about/glitch/types.ts` CONSTANT-for-
// constant — bump both together.
//
// Fragment-side chunk only — the fullscreen-triangle vs_main arrives
// via gpu-mini's wgsl library.
//
// # Pipeline order (mirrors `fx.ts frame()`)
//
//   1. circle clip        — portrait is a circle; outside → α=0
//   2. hit gate           — hash(seed) < HIT (0.05 idle / 0.65 hover)
//   3. rgbSplit (hit)     — red at −dx, cyan at +dx (replaces base)
//   4. sliceShift (hit)   — ≤6/≤2 band slots, uv.x += dx per band
//   5. blockTear (hit∩hov∩50%) — ≤3 rects duplicated at ±dx
//   6. rollBar (hover)    — brightened band sweeping at ROLL_SPEED
//   7. trackBar (hov∩50%) — thin white strip at hashed y
//   8. invertFlash (hov∩2%) — 1−rgb for the frame
//   9. scanlines          — 3px-period dark lines + frame jitter
//   10. grain             — per-pixel hash speckle (denser on hover)
//
// # Hash scheme
//
// `seed` arrives bitcast in the uniform — Rust advances xorshift32
// once per frame. `hash_slot` below is a Wang-ish mix: constants are
// primes, the avalanche is two shifts + a multiply — enough
// decorrelation for visual noise at ~1 ALU op's cost. Every "random
// per frame" decision is hash_slot(seed, SLOT_CONSTANT) —
// deterministic, frame-coherent. (Widget-specific by design — the
// shared library's sin-hash can't take a u32 seed.)

struct Uniforms {
    resolution_dev: vec2<f32>,  // canvas device px
    time_seconds: f32,          // seconds — roll sweep clock
    hover_amount: f32,          // 0 idle / 1 hover
    seed: u32,                  // xorshift32 state (bitcast f32 → u32)
    cover_rect: vec4<f32>,      // dx,dy,dw,dh device px (cover-fit rect)
    reserved: vec3<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var portrait: texture_2d<f32>;
@group(0) @binding(2) var portrait_sampler: sampler;

// ── Tunables (types.ts parity — see file doc) ────────────────────────
const IDLE_HIT      : f32 = 0.05;   // _IDLE_HIT
const HOVER_HIT     : f32 = 0.65;   // _HOVER_HIT
const IDLE_GRAIN    : f32 = 0.0003; // _IDLE_GRAIN 16 px on ~250² ≈ 0.03% coverage
const HOVER_GRAIN   : f32 = 0.007;  // _HOVER_GRAIN 460 px ≈ 0.7%
const GRAIN_ALPHA   : f32 = 0.22;   // _GRAIN_ALPHA
const IDLE_GRAIN_A  : f32 = 0.096;  // _SCAN_ALPHA·1.6 (idle grain alpha)
const IDLE_RGB      : f32 = 1.4;    // _IDLE_RGB (device px)
const HOVER_RGB_MIN : f32 = 1.0;    // _HOVER_RGB_MIN
const HOVER_RGB_MAX : f32 = 5.0;    // _HOVER_RGB_MAX
const IDLE_TEAR     : f32 = 0.02;   // _IDLE_TEAR (width fraction)
const HOVER_TEAR    : f32 = 0.035;  // _HOVER_TEAR
const IDLE_BANDS    : i32 = 2;      // _IDLE_BANDS
const HOVER_BANDS   : i32 = 6;      // _HOVER_BANDS
const IDLE_BAND_MIN : f32 = 0.006;  // _IDLE_BAND_MIN (height fraction)
const IDLE_BAND_MAX : f32 = 0.03;   // _IDLE_BAND_MAX
const HOV_BAND_MIN  : f32 = 0.01;   // _HOVER_BAND_MIN
const HOV_BAND_MAX  : f32 = 0.06;   // _HOVER_BAND_MAX
const HOVER_BLOCKS  : i32 = 3;      // _HOVER_BLOCKS
const BLOCK_W_MAX   : f32 = 0.18;   // _BLOCK_W_MAX
const BLOCK_H_MAX   : f32 = 0.05;   // _BLOCK_H_MAX
const BLOCK_W_MIN   : f32 = 0.05;   // 0.05 + rand·MAX (fx.ts)
const BLOCK_H_MIN   : f32 = 0.015;  // 0.015 + rand·MAX
const BLOCK_THROW   : f32 = 0.05;   // _BLOCK_THROW
const TRACK_H       : f32 = 0.02;   // _TRACK_H
const TRACK_ALPHA   : f32 = 0.3;    // trackBar globalAlpha
const INVERT_CHANCE : f32 = 0.02;   // _INVERT_CHANCE
const ROLL_H        : f32 = 0.12;   // _ROLL_H
const ROLL_SPEED    : f32 = 0.4;    // _ROLL_SPEED
const SCAN_PERIOD   : f32 = 3.0;    // _SCAN_PERIOD (device px)
const SCAN_ALPHA    : f32 = 0.06;   // _SCAN_ALPHA
const CHROMA_CHANCE : f32 = 0.3;    // _CHROMA_CHANCE

// Hash slots — every "random per frame" decision is
// hash_slot(SLOT_×) so slots decorrelate by prime-spaced offsets.
const SLOT_HIT_GATE: u32 = 1u;
const SLOT_BAND_BASE: u32 = 2u;     // band i → base + i×BAND_STRIDE
const SLOT_BAND_STRIDE: u32 = 13u;
const SLOT_BLOCK_GATE: u32 = 60u;
const SLOT_BLOCK_BASE: u32 = 70u;   // block i → base + i×BLOCK_STRIDE
const SLOT_BLOCK_STRIDE: u32 = 17u;
const SLOT_RGB_SPLIT: u32 = 90u;
const SLOT_ROLL_TEAR: u32 = 100u;
const SLOT_TRACK_GATE: u32 = 110u;
const SLOT_TRACK_Y: u32 = 111u;
const SLOT_INVERT: u32 = 120u;
const SLOT_SCAN_JITTER: u32 = 130u;
const SLOT_GRAIN_PICK: u32 = 200u;
const SLOT_GRAIN_INK: u32 = 201u;
const SLOT_GRAIN_CHROMA: u32 = 202u;
const SLOT_GRAIN_CHANNEL: u32 = 203u;
// The roll sweep wraps over 1.3 (not 1.0) so the band spends 30% of
// its cycle fully off-screen — a pause between passes.
const ROLL_WRAP: f32 = 1.3;
// Wang-scramble output mask — top 24 bits → 0..1.
const HASH_MASK_MAX: f32 = 16777216.0;
// Block-tear chance — the JS Math.random()<0.5 parity.
const BLOCK_CHANCE: f32 = 0.5;
// Grain ink coin-flip + hover gate threshold.
const HALF_CHANCE: f32 = 0.5;

/// `hash_slot` — per-frame hash: mix (seed, slot) into a 0..1 f32.
/// Two multiplies + xor — the cheap Wang scramble; band slots differ
/// by prime offsets so slots don't correlate.
fn hash_slot(slot: u32) -> f32 {
    var state = uniforms.seed ^ (slot * 0x9E3779B9u);

    state ^= state >> 16u;
    state *= 0x7FEB352Du;
    state ^= state >> 15u;
    state *= 0x846CA68Bu;
    state ^= state >> 16u;

    return f32(state & 0xFFFFFFu) / HASH_MASK_MAX;
}

/// `hash_pixel` — same hash salted by the pixel coord — grain noise
/// varies per-pixel per-frame.
fn hash_pixel(pixel: vec2<f32>, salt: u32) -> f32 {
    var state = uniforms.seed
        ^ (u32(pixel.x) * 0x85EBCA6Bu)
        ^ (u32(pixel.y) * 0xC2B2AE35u)
        ^ (salt * 0x9E3779B9u);

    state ^= state >> 16u;
    state *= 0x7FEB352Du;
    state ^= state >> 15u;
    state *= 0x846CA68Bu;
    state ^= state >> 16u;

    return f32(state & 0xFFFFFFu) / HASH_MASK_MAX;
}

/// `fs_main` — one fragment of the portrait quad, all passes fused.
@fragment
fn fs_main(@builtin(position) pixel_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let hovered = uniforms.hover_amount > 0.5;

    // 1 ── Circle clip: the portrait is a circle inscribed in the
    // square viewport centered on the canvas — outside → transparent
    // (the canvas composites over the DOM <img> unchanged there).
    // Mask, NOT an early return: a fragment-dependent `return` makes
    // every later control-flow node non-uniform, and textureSample's
    // implicit derivatives are only legal in uniform flow (Dawn:
    // "'textureSample' must only be called from uniform control flow").
    let viewport_size = min(uniforms.resolution_dev.x, uniforms.resolution_dev.y);
    let center_px = uniforms.resolution_dev * 0.5;
    let clip_radius = viewport_size * 0.5;
    let inside = select(0.0, 1.0, distance(pixel_coord.xy, center_px) <= clip_radius);

    // frag→uv through the cover rect — identical to the DOM's
    // object-fit:cover + object-position mapping (coverRect()).
    var uv = (pixel_coord.xy - uniforms.cover_rect.xy) / uniforms.cover_rect.zw;

    // 2 ── Hit gate: hash decides the frame's corruption class —
    // same seed → same decision for every pixel (frame-coherent).
    let hit = hash_slot(SLOT_HIT_GATE) < select(IDLE_HIT, HOVER_HIT, hovered);

    var color = vec4<f32>(0.0);

    // 3+4 ── Split base + slice tears: on hit frames the base is the
    // channel-split composite (red at −dx, cyan at +dx) — the JS
    // 'lighter' composite of complementary channels, which sums back
    // to the source except at the fringe — then bands shift uv.x.
    if (hit) {
        // ── sliceShift — band slots hashed (top, height, dx) ──────
        let band_count = select(IDLE_BANDS, HOVER_BANDS, hovered);
        let band_lo = select(IDLE_BAND_MIN, HOV_BAND_MIN, hovered);
        let band_hi = select(IDLE_BAND_MAX, HOV_BAND_MAX, hovered);
        let tear_span = select(IDLE_TEAR, HOVER_TEAR, hovered);

        for (var band_i = 0; band_i < HOVER_BANDS; band_i = band_i + 1) {
            if (band_i >= band_count) { break; }

            let slot = u32(band_i) * SLOT_BAND_STRIDE + SLOT_BAND_BASE;
            let band_top = hash_slot(slot);
            let band_height = band_lo + hash_slot(slot + 1u) * (band_hi - band_lo);
            let band_shift = (hash_slot(slot + 2u) * 2.0 - 1.0) * tear_span;

            if (uv.y >= band_top && uv.y <= band_top + band_height) {
                uv.x += band_shift;
            }
        }

        // ── blockTear — rect slots duplicated at ±dx (hover) ──────
        // Gated by the same 50% the JS Math.random()<0.5 uses.
        if (hovered && hash_slot(SLOT_BLOCK_GATE) < BLOCK_CHANCE) {
            for (var block_i = 0; block_i < HOVER_BLOCKS; block_i = block_i + 1) {
                let slot = u32(block_i) * SLOT_BLOCK_STRIDE + SLOT_BLOCK_BASE;
                let block_w = BLOCK_W_MIN + hash_slot(slot) * BLOCK_W_MAX;
                let block_h = BLOCK_H_MIN + hash_slot(slot + 1u) * BLOCK_H_MAX;
                let block_x = hash_slot(slot + 2u) * (1.0 - block_w);
                let block_y = hash_slot(slot + 3u) * (1.0 - block_h);
                let block_shift = (hash_slot(slot + 4u) * 2.0 - 1.0) * BLOCK_THROW;

                if (uv.x >= block_x + block_shift && uv.x <= block_x + block_w + block_shift &&
                    uv.y >= block_y && uv.y <= block_y + block_h) {
                    uv.x -= block_shift;   // duplicate: read the source rect
                }
            }
        }

        // ── rgbSplit — the split base (replaces base on hits) ─────
        let split_dx = select(
            hash_slot(SLOT_RGB_SPLIT) * IDLE_RGB,
            HOVER_RGB_MIN + hash_slot(SLOT_RGB_SPLIT) * (HOVER_RGB_MAX - HOVER_RGB_MIN),
            hovered,
        ) / uniforms.cover_rect.z;

        let channel_r = textureSample(portrait, portrait_sampler, uv - vec2<f32>(split_dx, 0.0)).r;
        let channel_gb = textureSample(portrait, portrait_sampler, uv + vec2<f32>(split_dx, 0.0)).gb;

        color = vec4<f32>(channel_r, channel_gb, 1.0);
    } else {
        color = textureSample(portrait, portrait_sampler, uv);
    }

    // 5 ── rollBar (hover): brightened torn band sweeping down —
    // y = ((t·0.4)%1.3)·(1+bh)−bh in uv space, plus the ±tear jitter.
    if (hovered) {
        let roll_height = ROLL_H;
        let roll_top = fract(uniforms.time_seconds * ROLL_SPEED / ROLL_WRAP) * (ROLL_WRAP + roll_height) - roll_height;

        // The torn-band sample runs unconditionally — textureSample
        // inside the fragment-varying `if` below is non-uniform flow;
        // sample first, gate the assignment.
        let torn_sample = textureSample(
            portrait, portrait_sampler,
            vec2<f32>(uv.x + (hash_slot(SLOT_ROLL_TEAR) - 0.5) * HOVER_TEAR, uv.y)
        ) * 1.3;

        if (uv.y >= roll_top && uv.y <= roll_top + roll_height) {
            color = torn_sample;
        }

        // 6 ── trackBar (50% gate): thin bright strip at hashed y.
        if (hash_slot(SLOT_TRACK_GATE) < HALF_CHANCE) {
            let track_y = hash_slot(SLOT_TRACK_Y);
            if (abs(uv.y - track_y) < TRACK_H * 0.5) {
                color = vec4<f32>(mix(color.rgb, vec3<f32>(1.0), TRACK_ALPHA), 1.0);
            }
        }

        // 7 ── invertFlash (2% gate): near-total inversion.
        if (hash_slot(SLOT_INVERT) < INVERT_CHANCE) {
            color = vec4<f32>(1.0 - color.rgb, color.a);
        }
    }

    // 8 ── scanlines: dark line every SCAN_PERIOD device px, jittered
    // ±1 period per frame (the JS translate(0, jit) pattern offset).
    let scan_jitter = floor(hash_slot(SLOT_SCAN_JITTER) * SCAN_PERIOD);

    if (fract((pixel_coord.y + scan_jitter) / SCAN_PERIOD) < 1.0 / SCAN_PERIOD) {
        color = vec4<f32>(color.rgb * (1.0 - SCAN_ALPHA), color.a);
    }

    // 9 ── grain: per-pixel hash speckle — the GPU version of the JS
    // fillRect loop, but per-pixel instead of 460 rects: coverage is
    // the same fraction, granularity is finer. Hover adds chroma
    // speckle (red/cyan) at CHROMA_CHANCE of the speckle pixels.
    let grain_pick = hash_pixel(pixel_coord.xy, SLOT_GRAIN_PICK);
    let grain_density = select(IDLE_GRAIN, HOVER_GRAIN, hovered);

    if (grain_pick < grain_density) {
        let grain_alpha = select(IDLE_GRAIN_A, GRAIN_ALPHA, hovered);

        var grain_ink = select(vec3<f32>(0.0), vec3<f32>(1.0), hash_pixel(pixel_coord.xy, SLOT_GRAIN_INK) < HALF_CHANCE);

        if (hovered && hash_pixel(pixel_coord.xy, SLOT_GRAIN_CHROMA) < CHROMA_CHANCE) {
            grain_ink = select(
                vec3<f32>(0.0, 1.0, 1.0),
                vec3<f32>(1.0, 0.0, 0.0),
                hash_pixel(pixel_coord.xy, SLOT_GRAIN_CHANNEL) < HALF_CHANCE,
            );
        }

        color = vec4<f32>(mix(color.rgb, grain_ink, grain_alpha), color.a);
    }

    // Circle-clip mask + premultiplied out — `inside` is 0 outside the
    // inscribed circle so the canvas composites transparent there.
    let alpha = color.a * inside;

    return vec4<f32>(color.rgb * alpha, alpha);
}
