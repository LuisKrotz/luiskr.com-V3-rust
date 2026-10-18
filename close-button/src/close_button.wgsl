// close_button.wgsl — the modal close button's liquid-fill hover.
//
// WGSL port of core/utils/canvas/widgets/close-button/shaders.ts.
// On hover a turquoise liquid rises inside the circular button with a
// wobbling foam crest and rising bubbles; the X strokes self-draw and
// slowly rotate; a tap emits a shockwave ring. All math is in PIXEL
// space relative to the button center (pRel) — the button is circular,
// so pixel space is the natural domain (radius-based clipping).
//
// Uniform layout (f32s, close_button.rs::UNIFORM_F32S = 12):
//   [0,1] res         design size px (button diameter)
//   [2]   time        seconds — wave phase, bubbles, shockwave clock
//   [3]   liquid      0..1 fill height (JS eases on hover)
//   [4]   draw        0..1 stroke progress for the self-drawing X
//   [5]   rot         X rotation angle, rad (JS advances it slowly)
//   [6]   click_time  seconds-stamp of last click (−1 = none)
//   [7]   pad         16B alignment pad
//   [8..11] rect      widget clip rect, canvas DEVICE px

struct Uniforms {
    res: vec2<f32>,
    time: f32,
    liquid: f32,
    draw: f32,
    rot: f32,
    click_time: f32,
    pad: f32,
    rect: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

fn distSegment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
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
    let pixel = uv * u.res;
    let center = u.res * 0.5;
    let pRel = pixel - center;
    let radius = min(center.x, center.y) - 1.0;
    let d = length(pRel);

    if (d > radius) {
        discard;
    }

    let borderAA = 1.0 - smoothstep(radius - 1.0, radius, d);

    var col = vec3<f32>(0.22, 0.23, 0.27);

    // ── Liquid fill ──────────────────────────────────────────────────
    // targetY maps liquid∈[0,1] → [-1.1, +1.1]×radius — the ±10%
    // overshoot lets the fill fully cover (1.1) and fully clear (−1.1)
    // the circle instead of stopping exactly at the rim.
    // The crest wave is two detuned sines (0.28 & 0.45 rad/px spatial
    // freq, 4.0 & 2.8 rad/s temporal) — detuning prevents visible
    // periodicity. `min(liquid*2.5, 1.0)` is a ramp: waves stay flat
    // until the fill is >40% risen, then reach full amplitude — a wavy
    // crest on an empty button looks broken.
    let targetY = (u.liquid * 2.2 - 1.1) * radius;
    let wave = sin(pRel.x * 0.28 + u.time * 4.0) * 1.6 * min(u.liquid * 2.5, 1.0)
        + cos(pRel.x * 0.45 - u.time * 2.8) * 1.0 * min(u.liquid * 2.5, 1.0);
    let surfaceY = targetY + wave;

    if (pRel.y <= surfaceY && u.liquid > 0.01) {
        // Depth tint: normalized depth below the crest lerps bright
        // surface-teal → deep teal (light absorbed with depth).
        let depth = clamp((surfaceY - pRel.y) / (radius * 2.0), 0.0, 1.0);
        var liquidColor = mix(
            vec3<f32>(0.40, 0.988, 0.945),
            vec3<f32>(0.18, 0.65, 0.62),
            depth,
        );

        // Foam crest — a 2px additive glow hugging the surface line.
        let crestDist = abs(pRel.y - surfaceY);
        let crest = 1.0 - smoothstep(0.0, 2.0, crestDist);
        liquidColor = liquidColor + vec3<f32>(0.75, 1.0, 0.98) * crest * 0.75;

        // Bubbles — four procedurally placed rings rising at staggered
        // speeds (12+4i px/s), x-jittered by a slow sine; y wraps on a
        // 2.2×radius cycle so bubbles recycle continuously. Each bubble
        // is a RING (inner smoothstep × outer smoothstep product — the
        // shell only) plus a 0.4-offset specular spot. Bubbles below the
        // crest and inside the button only (`bY <= surfaceY`).
        for (var i = 0; i < 4; i = i + 1) {
            let fi = f32(i);
            let bSpeed = 12.0 + fi * 4.0;
            let bX = -8.0 + fi * 5.5 + sin(u.time * 2.5 + fi * 1.7) * 2.5;
            let bY = (u.time * bSpeed + fi * 9.0) % (radius * 2.2) - radius;
            let bPos = vec2<f32>(bX, bY);
            let bRad = 1.2 + (fi % 2.0) * 0.7;

            if (bY <= surfaceY && bY > -radius) {
                let dBubble = length(pRel - bPos);
                let bRing = (1.0 - smoothstep(bRad - 0.4, bRad + 0.4, dBubble)) * smoothstep(bRad - 1.2, bRad - 0.3, dBubble);
                let bSpot = 1.0 - smoothstep(0.0, 0.6, length(pRel - (bPos + vec2<f32>(0.4, 0.4))));
                liquidColor = liquidColor + vec3<f32>(0.85, 1.0, 0.98) * (bRing * 0.55 + bSpot * 0.75);
            }
        }

        col = mix(col, liquidColor, borderAA);
    }

    // Inset rim shadow.
    let rim = smoothstep(radius - 3.0, radius, d);
    col = mix(col, col * 0.7, rim * 0.35);

    // Rotating X strokes — the sample point is rotated by -rot (rotate
    // the POINT, not the strokes — same SDF convention as the burger).
    // Each stroke is a segment whose B endpoint lerp-grows with `draw`
    // (l1B = A + fullLen·progress) — that's the self-drawing effect:
    // dist to a partial-length segment only inks the drawn part.
    // halfLen 0.38·radius keeps the X inside the fill circle with a
    // comfortable margin; strokes get ~0.8px AA via smoothstep(0.9,1.7).
    let pRot = vec2<f32>(
        pRel.x * cos(u.rot) - pRel.y * sin(u.rot),
        pRel.x * sin(u.rot) + pRel.y * cos(u.rot),
    );

    let halfLen = radius * 0.38;
    let progress = clamp(u.draw, 0.0, 1.0);

    let l1A = vec2<f32>(-halfLen, -halfLen);
    let l1B = l1A + vec2<f32>(halfLen * 2.0) * progress;
    let d1 = distSegment(pRot, l1A, l1B);

    let l2A = vec2<f32>(-halfLen, halfLen);
    let l2B = l2A + vec2<f32>(halfLen * 2.0, -halfLen * 2.0) * progress;
    let d2 = distSegment(pRot, l2A, l2B);

    let stroke1 = (1.0 - smoothstep(0.9, 1.7, d1)) * step(0.01, progress);
    let stroke2 = (1.0 - smoothstep(0.9, 1.7, d2)) * step(0.01, progress);
    let totalX = clamp(stroke1 + stroke2, 0.0, 1.0);

    col = mix(col, vec3<f32>(1.0), totalX * borderAA);

    // Click shockwave — a 2.5px-wide ring expanding at 45px/s, alpha
    // decaying to zero over 0.4s (18px total travel — most of the
    // button's radius). Same "elapsed-time gate" pattern as the theme
    // slider's ripple: click_time = -1 keeps it permanently inert.
    let clickElapsed = u.time - u.click_time;
    if (clickElapsed >= 0.0 && clickElapsed < 0.4) {
        let waveRad = clickElapsed * 45.0;
        let waveShock = (1.0 - smoothstep(0.0, 2.5, abs(d - waveRad))) * (1.0 - clickElapsed / 0.4);
        col = col + vec3<f32>(0.40, 0.988, 0.945) * waveShock * 0.8;
    }

    return vec4<f32>(col, borderAA);
}
