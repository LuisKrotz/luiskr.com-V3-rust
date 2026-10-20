// glitch.wgsl — the about-portrait VHS corruption pass.
//
// One fullscreen textured draw replaces the whole 2D pipeline:
// `fillRect` grain ×460/frame, `drawImage` channel ghosts, band
// re-blits, scanline pattern fills — all become per-fragment texture
// lookups hashed from a per-frame seed. Tunables mirror
// `website/components/home/about/glitch/types.ts` CONSTANT-for-
// constant — bump both together.
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
// `seed` arrives bitcast in u[4] — Rust advances xorshift32 once per
// frame. `h(a,b)` below is a Wang-ish mix: constants are primes, the
// avalanche is two shifts + a multiply — enough decorrelation for
// visual noise at ~1 ALU op's cost. Every "random per frame" decision
// is h(seed, SLOT_CONSTANT) — deterministic, frame-coherent.

struct U {
    res     : vec2<f32>,   // canvas device px
    time    : f32,         // seconds — roll sweep clock
    hover   : f32,         // 0 idle / 1 hover
    seed    : u32,         // xorshift32 state (bitcast f32 → u32)
    cover   : vec4<f32>,   // dx,dy,dw,dh device px (cover-fit rect)
    _pad    : vec3<f32>,
};

@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var tex  : texture_2d<f32>;
@group(0) @binding(2) var smp  : sampler;

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

/// `h` — per-frame hash: mix (seed, slot) into a 0..1 f32. Two
/// multiplies + xor — the cheap Wang scramble; band slots differ by
/// PRIME offsets so slots don't correlate.
fn h(slot : u32) -> f32 {
    var x = u.seed ^ (slot * 0x9E3779B9u);

    x ^= x >> 16u;
    x *= 0x7FEB352Du;
    x ^= x >> 15u;
    x *= 0x846CA68Bu;
    x ^= x >> 16u;

    return f32(x & 0xFFFFFFu) / 16777216.0;
}

/// `hp` — same hash without the uniform (for grain: mix seed with the
/// pixel coord so noise varies per-pixel per-frame).
fn hp(p : vec2<f32>, salt : u32) -> f32 {
    var x = u.seed ^ (u32(p.x) * 0x85EBCA6Bu) ^ (u32(p.y) * 0xC2B2AE35u) ^ (salt * 0x9E3779B9u);

    x ^= x >> 16u;
    x *= 0x7FEB352Du;
    x ^= x >> 15u;
    x *= 0x846CA68Bu;
    x ^= x >> 16u;

    return f32(x & 0xFFFFFFu) / 16777216.0;
}

/// `fs_main` — one fragment of the portrait quad, all passes fused.
@fragment
fn fs_main(@builtin(position) frag : vec4<f32>) -> @location(0) vec4<f32> {
    let hover = u.hover > 0.5;

    // 1 ── Circle clip: the portrait is a circle inscribed in the
    // square viewport centered on the canvas — outside → transparent
    // (the canvas composites over the DOM <img> unchanged there).
    // Mask, NOT an early return: a fragment-dependent `return` makes
    // every later control-flow node non-uniform, and textureSample's
    // implicit derivatives are only legal in uniform flow (Dawn:
    // "'textureSample' must only be called from uniform control flow").
    let size   = min(u.res.x, u.res.y);
    let center = u.res * 0.5;
    let radius = size * 0.5;
    let inside = select(0.0, 1.0, distance(frag.xy, center) <= radius);

    // frag→uv through the cover rect — identical to the DOM's
    // object-fit:cover + object-position mapping (coverRect()).
    var uv = (frag.xy - u.cover.xy) / u.cover.zw;

    // 2 ── Hit gate: hash decides the frame's corruption class —
    // same seed → same decision for every pixel (frame-coherent).
    let hit = h(1u) < select(IDLE_HIT, HOVER_HIT, hover);

    var col = vec4<f32>(0.0);

    // 3+4 ── Split base + slice tears: on hit frames the base is the
    // channel-split composite (red at −dx, cyan at +dx) — the JS
    // 'lighter' composite of complementary channels, which sums back
    // to the source except at the fringe — then bands shift uv.x.
    if (hit) {
        // ── sliceShift — band slots hashed (y, height, dx) ────────
        let bands   = select(IDLE_BANDS, HOVER_BANDS, hover);
        let band_lo = select(IDLE_BAND_MIN, HOV_BAND_MIN, hover);
        let band_hi = select(IDLE_BAND_MAX, HOV_BAND_MAX, hover);
        let tear    = select(IDLE_TEAR, HOVER_TEAR, hover);

        for (var i = 0; i < HOVER_BANDS; i++) {
            if (i >= bands) { break; }

            let fi  = u32(i) * 13u + 2u;
            let y0  = h(fi);                          // band top 0..1
            let bh  = band_lo + h(fi + 1u) * (band_hi - band_lo);
            let dx  = (h(fi + 2u) * 2.0 - 1.0) * tear;

            if (uv.y >= y0 && uv.y <= y0 + bh) {
                uv.x += dx;
            }
        }

        // ── blockTear — rect slots duplicated at ±dx (hover) ──────
        // Gated by the same 50% the JS Math.random()<0.5 uses.
        if (hover && h(60u) < 0.5) {
            for (var i = 0; i < HOVER_BLOCKS; i++) {
                let fi = u32(i) * 17u + 70u;
                let bw = BLOCK_W_MIN + h(fi) * BLOCK_W_MAX;
                let bh = BLOCK_H_MIN + h(fi + 1u) * BLOCK_H_MAX;
                let bx = h(fi + 2u) * (1.0 - bw);
                let by = h(fi + 3u) * (1.0 - bh);
                let dx = (h(fi + 4u) * 2.0 - 1.0) * BLOCK_THROW;

                if (uv.x >= bx + dx && uv.x <= bx + bw + dx &&
                    uv.y >= by && uv.y <= by + bh) {
                    uv.x -= dx;   // duplicate: read the source rect
                }
            }
        }

        // ── rgbSplit — the split base (replaces base on hits) ─────
        let dx = select(h(90u) * IDLE_RGB,
                        HOVER_RGB_MIN + h(90u) * (HOVER_RGB_MAX - HOVER_RGB_MIN),
                        hover) / u.cover.z;

        let r = textureSample(tex, smp, uv - vec2<f32>(dx, 0.0)).r;
        let gb = textureSample(tex, smp, uv + vec2<f32>(dx, 0.0)).gb;

        col = vec4<f32>(r, gb, 1.0);
    } else {
        col = textureSample(tex, smp, uv);
    }

    // 5 ── rollBar (hover): brightened torn band sweeping down —
    // y = ((t·0.4)%1.3)·(1+bh)−bh in uv space, plus the ±tear jitter.
    if (hover) {
        let bh = ROLL_H;
        let y  = fract(u.time * ROLL_SPEED / 1.3) * (1.3 + bh) - bh;

        // The torn-band sample runs unconditionally — textureSample
        // inside the fragment-varying `if` below is non-uniform flow;
        // sample first, gate the assignment.
        let torn = textureSample(
            tex, smp,
            vec2<f32>(uv.x + (h(100u) - 0.5) * HOVER_TEAR, uv.y)
        ) * 1.3;

        if (uv.y >= y && uv.y <= y + bh) {
            col = torn;
        }

        // 6 ── trackBar (50% gate): thin bright strip at hashed y.
        if (h(110u) < 0.5) {
            let ty = h(111u);
            if (abs(uv.y - ty) < TRACK_H * 0.5) {
                col = vec4<f32>(mix(col.rgb, vec3<f32>(1.0), TRACK_ALPHA), 1.0);
            }
        }

        // 7 ── invertFlash (2% gate): near-total inversion.
        if (h(120u) < INVERT_CHANCE) {
            col = vec4<f32>(1.0 - col.rgb, col.a);
        }
    }

    // 8 ── scanlines: dark line every SCAN_PERIOD device px, jittered
    // ±1 period per frame (the JS translate(0, jit) pattern offset).
    let jit = floor(h(130u) * SCAN_PERIOD);
    let row = floor((frag.y + jit) / SCAN_PERIOD);

    if (fract((frag.y + jit) / SCAN_PERIOD) < 1.0 / SCAN_PERIOD) {
        col = vec4<f32>(col.rgb * (1.0 - SCAN_ALPHA), col.a);
    }

    // 9 ── grain: per-pixel hash speckle — the GPU version of the JS
    // fillRect loop, but per-pixel instead of 460 rects: coverage is
    // the same fraction, granularity is finer. Hover adds chroma
    // speckle (red/cyan) at CHROMA_CHANCE of the speckle pixels.
    let gp = hp(frag.xy, 200u);
    let density = select(IDLE_GRAIN, HOVER_GRAIN, hover);

    if (gp < density) {
        let ga = select(IDLE_GRAIN_A, GRAIN_ALPHA, hover);

        var ink = select(vec3<f32>(0.0), vec3<f32>(1.0), hp(frag.xy, 201u) < 0.5);

        if (hover && hp(frag.xy, 202u) < CHROMA_CHANCE) {
            ink = select(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), hp(frag.xy, 203u) < 0.5);
        }

        col = vec4<f32>(mix(col.rgb, ink, ga), col.a);
    }

    // Circle-clip mask + premultiplied out — `inside` is 0 outside the
    // inscribed circle so the canvas composites transparent there.
    let a = col.a * inside;

    return vec4<f32>(col.rgb * a, a);
}

/// `vs_main` — the shared fullscreen triangle (gpu-mini contract:
/// positions from vertex_index, uv through @location(0)).
@vertex
fn vs_main(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
    // 3-vertex fullscreen triangle: (0)=(-1,-1) (1)=(3,-1) (2)=(-1,3)
    // — covers the whole clip rect with no vertex buffer (gpu-mini's
    // `fullscreen_pipeline` contract).
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );

    return vec4<f32>(pos[i], 0.0, 1.0);
}
