// theme_slider.wgsl — the day/system/night preference slider.
//
// WGSL port of core/utils/canvas/widgets/theme-slider/shaders.ts.
// This is the LARGEST widget shader — a full procedural scene
// (sky gradient, starfield, sun with corona, aurora, crescent moon,
// two parallax mesa layers, 3D knob) where `progress` 0..2 morphs
// night → system-twilight → day. All coordinates are in ASPECT-
// CORRECTED track space `p` (x ∈ 0..aspect, y ∈ 0..1, track is a
// capsule of radius 0.5 centered at y=0.5).
//
// Uniform layout (f32s, theme_slider.rs::UNIFORM_F32S = 12):
//   [0,1] res         design size px
//   [2]   time        seconds — star twinkle, aurora, corona, shimmer
//   [3]   progress    0..2 — 0..1: night→system, 1..2: system→day
//   [4]   knob_x      knob center X, px (JS spring)
//   [5]   ripple_time seconds-stamp of last tap (−1 = none)
//   [6]   ripple_pos  tap X in px → expanding ring origin
//   [7]   pad         16B alignment pad
//   [8..11] rect      widget clip rect, canvas DEVICE px
//
// Layer order (painter's, back→front — do not reorder):
//   sky → stars → sun → aurora → moon → back mesa → front mesa →
//   bezel → click ripple → knob shadow → knob → knob halos.

struct Uniforms {
    res: vec2<f32>,
    time: f32,
    progress: f32,
    knob_x: f32,
    ripple_time: f32,
    ripple_pos: f32,
    pad: f32,
    rect: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

fn distCapsule(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// hash(p) → [0,1) — the classic sin-fract hash for deterministic
// per-cell randomness. dot(p, (12.9898, 78.233)) decorrelates x/y;
// ×437.585453 spreads the fractional part. NOT cryptographic — it only
// needs cells to look random for star placement. Deterministic = same
// stars every frame (essential — a time-varying hash would boil).
fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 437.585453);
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
    let uv = vec2<f32>(
        (frag.x - u.rect.x) / u.rect.z,
        (frag.y - u.rect.y) / u.rect.w,
    );

    let aspect = u.res.x / u.res.y;
    let p = vec2<f32>(uv.x * aspect, uv.y);

    let cA = vec2<f32>(0.5, 0.5);
    let cB = vec2<f32>(aspect - 0.5, 0.5);
    let trackDist = distCapsule(p, cA, cB);

    if (trackDist > 0.5) {
        discard;
    }

    let borderAA = 1.0 - smoothstep(0.485, 0.5, trackDist);

    // Sky gradient by height.
    let h = p.y;
    let nightSky = mix(vec3<f32>(0.38, 0.28, 0.60), vec3<f32>(0.46, 0.35, 0.70), h);
    let sysSky = mix(vec3<f32>(0.60, 0.38, 0.52), vec3<f32>(0.72, 0.48, 0.62), h);
    let daySky = mix(vec3<f32>(0.92, 0.48, 0.40), vec3<f32>(0.98, 0.64, 0.50), h);

    var sky: vec3<f32>;
    if (u.progress <= 1.0) {
        sky = mix(nightSky, sysSky, u.progress);
    } else {
        sky = mix(sysSky, daySky, u.progress - 1.0);
    }

    var col = sky;

    // Twinkling stars — only for progress < ~1.1 (night + twilight).
    // Grid cells of 1/20 scene-height; ~28% of cells host a star
    // (rnd > 0.72). Each star sits at its cell CENTER (0.5 offset) and
    // twinkles at time*3.5 with a per-cell phase (rnd*6.28 ≈ rnd·2π) so
    // they pulse out of sync. Radius 0.022 in scene units.
    let starAlpha = 1.0 - smoothstep(0.2, 1.1, u.progress);
    if (starAlpha > 0.01) {
        let starGrid = floor(p * 20.0);
        let rnd = hash(starGrid);
        if (rnd > 0.72) {
            let sPos = (starGrid + vec2<f32>(0.5)) / 20.0;
            let dStar = length(p - sPos);
            let twinkle = 0.5 + 0.5 * sin(u.time * 3.5 + rnd * 6.28);
            let sGlow = (1.0 - smoothstep(0.0, 0.022, dStar)) * twinkle * starAlpha;
            col = col + vec3<f32>(1.0) * sGlow;
        }
    }

    // Rising sun — fades in across the 0.3..1.8 progress window.
    // Structure: pulsing core disc + three concentric glow rings at
    // decreasing alpha + angular rays rotating slowly (angle*10 = ten
    // rays, time*0.85 rotates them ~0.14 rad/s). `coronaPulse` breathes
    // the core radius ±1.2%. `totalSun` is clamped because ring alphas
    // overlap and would sum >1.
    let sunAlpha = smoothstep(0.3, 1.8, u.progress);
    if (sunAlpha > 0.01) {
        let sunPos = vec2<f32>(0.80, 0.62);
        let dSun = length(p - sunPos);
        let coronaPulse = 0.012 * sin(u.time * 2.8);
        let core = 1.0 - smoothstep(0.135 + coronaPulse, 0.165 + coronaPulse, dSun);
        let ring1 = (1.0 - smoothstep(0.22, 0.26, dSun)) * 0.45;
        let ring2 = (1.0 - smoothstep(0.30, 0.35, dSun)) * 0.28;
        let ring3 = (1.0 - smoothstep(0.39, 0.45, dSun)) * 0.16;

        let sunAngle = atan2(p.y - sunPos.y, p.x - sunPos.x);
        let sunRays = 0.5 + 0.5 * sin(sunAngle * 10.0 + u.time * 0.85);
        let rayGlow = (1.0 - smoothstep(0.14, 0.44, dSun)) * sunRays * 0.28;

        let sunCream = vec3<f32>(0.99, 0.90, 0.75);
        let ringCol = mix(vec3<f32>(0.98, 0.76, 0.60), vec3<f32>(0.96, 0.64, 0.52), clamp(dSun / 0.45, 0.0, 1.0));
        let totalSun = clamp(core + ring1 + ring2 + ring3 + rayGlow, 0.0, 1.0);
        col = mix(col, mix(ringCol, sunCream, core), totalSun * sunAlpha);
    }

    // Twilight aurora — `sysFactor` peaks at progress=1.0 and decays
    // over a ±0.625 window (abs(p-1)*1.6 hits 1 at |Δ|=0.625), so the
    // aurora only exists around "system" mode. `auroraWave` is a
    // phase-modulated sine (inner sin bends the wave vertically — the
    // curtain effect); `auroraBand` clips it to a y∈[0.35,0.95] window;
    // the color lerps cyan→magenta across x with time drift.
    let sysFactor = 1.0 - min(1.0, abs(u.progress - 1.0) * 1.6);
    if (sysFactor > 0.01) {
        let auroraWave = sin(p.x * 2.6 + sin(p.y * 3.8 + u.time * 1.4) + u.time * 1.6) * 0.5 + 0.5;
        let auroraBand = smoothstep(0.35, 0.75, p.y) * (1.0 - smoothstep(0.70, 0.95, p.y));
        let auroraCol = mix(vec3<f32>(0.38, 0.78, 0.92), vec3<f32>(0.85, 0.45, 0.80), sin(p.x * 1.8 + u.time * 0.9) * 0.5 + 0.5);
        col = col + auroraCol * auroraWave * auroraBand * 0.38 * sysFactor;
    }

    // Crescent moon — SDF SUBTRACTION: a full disc (moonOut) minus an
    // offset inner disc (moonIn shifted (-0.052, +0.042)) — the
    // intersection leaves only the lit crescent sliver. The offset
    // vector aims the crescent's horns down-right (waxing evening moon).
    let moonPos = vec2<f32>(2.58, 0.60);
    let dMoonOut = length(p - moonPos);
    let dMoonIn = length(p - (moonPos + vec2<f32>(-0.052, 0.042)));
    let moonOut = 1.0 - smoothstep(0.16, 0.18, dMoonOut);
    let moonIn = smoothstep(0.138, 0.174, dMoonIn);
    let crescent = moonOut * moonIn;
    let crescentAlpha = 1.0 - smoothstep(0.1, 1.2, u.progress);
    col = mix(col, vec3<f32>(1.0), clamp(crescent, 0.0, 1.0) * crescentAlpha * 0.95);

    // Canyon mesas — the terrain silhouettes are HEIGHT FUNCTIONS, not
    // shapes: `backHeight(x)` = base 0.33 + slow sin swell + faster cos
    // detail + two plateau bumps (triangular plateau1/2 — distance-to-
    // center shaped with smoothstep gives flat tops, the mesa look).
    // Pixels below backHeight fill with the back-mesa color.
    // `parallax` slides the back layer ±0.12 with progress while the
    // front layer moves half as much in the opposite direction —
    // differential motion is what sells depth. `heatShimmer` adds a
    // ±0.5% high-frequency vertical wobble only during day (dayAlpha).
    let parallax = (u.progress - 1.0) * 0.12;
    let px = p.x + parallax;
    let dayAlpha = smoothstep(1.0, 1.9, u.progress);
    let heatShimmer = sin(p.x * 14.0 + u.time * 2.8) * 0.005 * dayAlpha;

    let mesa1 = 0.33 + 0.14 * sin(px * 2.6) + 0.09 * cos(px * 5.4);
    let plateau1 = (1.0 - smoothstep(0.0, 0.3, abs(px - 1.72))) * 0.16;
    let plateau2 = (1.0 - smoothstep(0.0, 0.24, abs(px - 2.68))) * 0.18;
    let backHeight = mesa1 + plateau1 + plateau2 + heatShimmer * 0.5;

    var mBackCol: vec3<f32>;
    if (u.progress <= 1.0) {
        mBackCol = mix(vec3<f32>(0.34, 0.16, 0.49), vec3<f32>(0.52, 0.24, 0.45), u.progress);
    } else {
        mBackCol = mix(vec3<f32>(0.52, 0.24, 0.45), vec3<f32>(0.76, 0.30, 0.40), u.progress - 1.0);
    }

    let duneBackAA = 1.0 - smoothstep(backHeight - 0.015, backHeight + 0.005, p.y);
    col = mix(col, mBackCol, duneBackAA);

    let frontHeight = 0.21 + 0.09 * sin((p.x - parallax * 0.5) * 2.4 + 0.8) + 0.05 * cos(p.x * 3.8) + heatShimmer;
    var mFrontCol: vec3<f32>;
    if (u.progress <= 1.0) {
        mFrontCol = mix(vec3<f32>(0.42, 0.22, 0.58), vec3<f32>(0.62, 0.30, 0.50), u.progress);
    } else {
        mFrontCol = mix(vec3<f32>(0.62, 0.30, 0.50), vec3<f32>(0.85, 0.40, 0.48), u.progress - 1.0);
    }

    let duneFrontAA = 1.0 - smoothstep(frontHeight - 0.015, frontHeight + 0.005, p.y);
    col = mix(col, mFrontCol, duneFrontAA);

    // Inset bezel.
    let bezel = smoothstep(0.46, 0.50, trackDist);
    col = mix(col, col * 0.65, bezel * 0.55);

    // Tap ripple — an expanding 0.08-thick ring, radius = elapsed×1.0
    // (one scene-height/s), alpha decaying linearly to 0 at 0.6s. The
    // `>= 0.0` guard keeps ripple_time=-1 (never tapped) inert.
    let rippleElapsed = u.time - u.ripple_time;
    if (rippleElapsed >= 0.0 && rippleElapsed < 0.6) {
        let rX = (u.ripple_pos / u.res.x) * aspect;
        let rDist = length(p - vec2<f32>(rX, 0.5));
        let rRadius = rippleElapsed * 1.0;
        let rWave = (1.0 - smoothstep(0.0, 0.08, abs(rDist - rRadius))) * (1.0 - rippleElapsed / 0.6);
        col = col + vec3<f32>(0.35, 0.25, 0.45) * rWave * 0.55;
    }

    // 3D knob — moon/sys/sun tint + lunar craters + corona halo.
    let knobXNorm = (u.knob_x / u.res.x) * aspect;
    let knobCenter = vec2<f32>(knobXNorm, 0.5);
    let knobDist = length(p - knobCenter);
    let knobRadius = 0.38;

    let knobShadow = (1.0 - smoothstep(knobRadius, knobRadius + 0.07, knobDist)) * 0.38;
    col = mix(col, vec3<f32>(0.12, 0.08, 0.20), knobShadow);

    var baseKnob: vec3<f32>;
    if (u.progress <= 1.0) {
        baseKnob = mix(vec3<f32>(0.98, 0.98, 1.0), vec3<f32>(0.98, 0.94, 0.90), u.progress);
    } else {
        baseKnob = mix(vec3<f32>(0.98, 0.94, 0.90), vec3<f32>(0.99, 0.89, 0.73), u.progress - 1.0);
    }

    // Lunar craters on the knob — five small circles at fixed offsets
    // with individually-tuned radii (the irregular spacing is what makes
    // it read as a moon and not a polka dot). Craters fade out with
    // progress (the knob morphs moon→sun; suns don't have craters).
    let craterFactor = 1.0 - smoothstep(0.1, 0.9, u.progress);
    if (craterFactor > 0.01) {
        let kp = p - knobCenter;
        let c1 = 1.0 - smoothstep(0.075, 0.095, length(kp - vec2<f32>(-0.13, 0.12)));
        let c2 = 1.0 - smoothstep(0.053, 0.071, length(kp - vec2<f32>(-0.11, -0.11)));
        let c3 = 1.0 - smoothstep(0.041, 0.059, length(kp - vec2<f32>(0.12, -0.07)));
        let c4 = 1.0 - smoothstep(0.059, 0.077, length(kp - vec2<f32>(0.09, 0.11)));
        let c5 = 1.0 - smoothstep(0.024, 0.042, length(kp - vec2<f32>(-0.01, 0.02)));
        let anyCrater = clamp(c1 + c2 + c3 + c4 + c5, 0.0, 1.0);
        baseKnob = mix(baseKnob, vec3<f32>(0.88, 0.89, 0.92), anyCrater * craterFactor * 0.75);
    }

    let sphere3D = 1.0 - (knobDist / knobRadius) * 0.14;
    let litKnob = baseKnob * sphere3D;

    let knobAA = 1.0 - smoothstep(knobRadius - 0.02, knobRadius + 0.005, knobDist);
    col = mix(col, litKnob, knobAA);

    if (dayAlpha > 0.01) {
        let sunKnobHalo = (1.0 - smoothstep(knobRadius, knobRadius + 0.08 + 0.02 * sin(u.time * 3.0), knobDist)) * 0.42 * dayAlpha;
        col = col + vec3<f32>(1.0, 0.82, 0.50) * sunKnobHalo;
    }
    if (sysFactor > 0.01) {
        let sysKnobHalo = (1.0 - smoothstep(knobRadius, knobRadius + 0.07 + 0.015 * sin(u.time * 2.4), knobDist)) * 0.35 * sysFactor;
        col = col + vec3<f32>(0.85, 0.72, 0.98) * sysKnobHalo;
    }

    return vec4<f32>(col * borderAA, borderAA);
}
