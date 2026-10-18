// switch_slider.wgsl — the preference toggle (pill + knob).
//
// WGSL port of core/utils/canvas/widgets/switch-slider/shaders.ts.
// Unlike the other widgets this shader works in RAW PIXELS, not UV —
// the original canvas was a fixed 28px-tall strip and every number is
// pixel-perfect for it. u.res carries that design size; at other sizes
// the pixel values do NOT scale (a deliberate choice — the toggle is a
// fixed-size control, not a responsive surface).
//
// Uniform layout (f32s, switch_slider.rs::UNIFORM_F32S = 12):
//   [0,1] res       design size, px (track is 14px tall, radius 14)
//   [2]   time      seconds — drives wave/streak animation
//   [3]   progress  0=off..1=on (eased by JS — shader just lerps)
//   [4]   knob_x    knob center X in px (JS owns the spring)
//   [5]   context   0=stats 1=grid 2=motion — picks track color+pattern
//   [6,7] pad       16B-alignment pad (vec4 boundary)
//   [8..11] rect    widget clip rect, canvas DEVICE px
//
// Render order inside this shader (each layer darkens/brightens the
// previous — order matters, don't reorder):
//   track capsule → context pattern → inset bezel → knob shadow →
//   knob sphere → accent dot → outer AA mask.

struct Uniforms {
    res: vec2<f32>,
    time: f32,
    progress: f32,
    knob_x: f32,
    context: f32, // 0=stats, 1=grid, 2=motion
    pad0: f32,
    pad1: f32,
    rect: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

fn distCapsule(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
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
    // The original ran in a dedicated 28px-tall canvas; normalize into
    // the widget rect and continue in its own px space.
    let uv = vec2<f32>(
        (frag.x - u.rect.x) / u.rect.z,
        (frag.y - u.rect.y) / u.rect.w,
    );
    let pixel = uv * u.res;

    let centerA = vec2<f32>(14.0, 14.0);
    let centerB = vec2<f32>(u.res.x - 14.0, 14.0);
    let trackDist = distCapsule(pixel, centerA, centerB);

    if (trackDist > 14.0) {
        discard;
    }

    let borderAA = 1.0 - smoothstep(13.2, 14.0, trackDist);

    let offTrack = vec3<f32>(0.18, 0.19, 0.22);
    var onTrack: vec3<f32>;

    if (u.context < 0.5) {
        onTrack = vec3<f32>(0.08, 0.58, 0.36); // stats: emerald
    } else if (u.context < 1.5) {
        onTrack = vec3<f32>(0.06, 0.48, 0.72); // grid: cyan blueprint
    } else {
        onTrack = vec3<f32>(0.48, 0.24, 0.68); // motion: violet
    }

    var col = mix(offTrack, onTrack, u.progress);

    if (u.context < 0.5) {
        // STATS — an ECG/heartbeat trace. waveX scrolls the trace right-
        // to-left at 20px/s; the base sin(0.35 waveX) gives ±2.5px swell;
        // the 36px-period window [12,18) adds the sharp spike: a
        // triangular -2.8/px slope centered on phase 15 (rises 8.4px in
        // 3px, falls 8.4px back — the classic QRS complex shape).
        // `pulse * (0.3 + 0.7*progress)` damps the whole trace to 30%
        // when the switch is off — heart barely beating.
        let waveX = pixel.x + u.time * 20.0;
        var pulse = sin(waveX * 0.35) * 2.5;

        let spikePhase = waveX % 36.0;
        if (spikePhase > 12.0 && spikePhase < 18.0) {
            pulse = pulse + (spikePhase - 15.0) * -2.8;
        }

        let waveY = 14.0 + pulse * (0.3 + 0.7 * u.progress);
        let dWave = abs(pixel.y - waveY);
        let waveGlow = 1.0 - smoothstep(0.0, 1.6, dWave);
        let waveCol = mix(vec3<f32>(0.35, 0.40, 0.45), vec3<f32>(0.45, 1.0, 0.65), u.progress);
        col = col + waveCol * waveGlow * 0.85;
    } else if (u.context < 1.5) {
        // GRID — blueprint hatch: vertical lines every 8px drifting at
        // 3px/s + fixed horizontal lines every 7px. Each `abs(mod - c)`
        // gives distance-to-nearest-line; the smoothstep turns it into a
        // 1px-soft line mask; adding X+Y makes a crosshatch.
        // Amplitude (0.25 + 0.55·progress): the blueprint fades IN as
        // the toggle turns on — barely visible when off.
        let gridX = (pixel.x + u.time * 3.0) % 8.0;
        let gridY = pixel.y % 7.0;
        let isLineX = 1.0 - smoothstep(0.0, 1.0, abs(gridX - 4.0));
        let isLineY = 1.0 - smoothstep(0.0, 1.0, abs(gridY - 3.5));
        let gridPattern = clamp(isLineX + isLineY, 0.0, 1.0);
        let gridCol = mix(vec3<f32>(0.30, 0.35, 0.42), vec3<f32>(0.50, 0.85, 1.0), u.progress);
        col = col + gridCol * gridPattern * (0.25 + 0.55 * u.progress);
    } else {
        // MOTION — when OFF (reduced-motion): diagonal comet streaks
        // moving at 6px/s, fading over 12px of their 20px cycle.
        // When ON: a single calm datum line at track center — the
        // reduced-motion setting literally reduces the motion rendered.
        if (u.progress < 0.5) {
            let streakY = pixel.y % 7.0;
            let streakMove = (pixel.x - u.time * 6.0) % 20.0;
            let lineY = 1.0 - smoothstep(0.0, 1.2, abs(streakY - 3.5));
            let lineFade = 1.0 - smoothstep(0.0, 12.0, streakMove);
            col = col + vec3<f32>(0.85, 0.70, 1.0) * lineY * lineFade * 0.45;
        } else {
            let lineY = 1.0 - smoothstep(0.0, 1.2, abs(pixel.y - 14.0));
            col = col + vec3<f32>(0.95, 0.85, 1.0) * lineY * 0.60;
        }
    }

    // Inset bezel — a darkening ramp in the outer 2.5px of the track
    // (trackDist 11.5→14) that makes the pill read as inset glass.
    let bezel = smoothstep(11.5, 14.0, trackDist);
    col = mix(col, vec3<f32>(0.0), bezel * 0.35);

    // ── The knob ─────────────────────────────────────────────────────
    // A fake-3D sphere built from one distance field:
    //   kShadow — a 2.5px soft shadow ring just past the knob edge.
    //   kSphere — radial shading: `1 - (d/r)*0.15` darkens the rim 15% —
    //     cheap Lambert-ish shading (no normals needed at this size).
    //   dotLight — a 2.7px accent dot at knob center, tinted per context.
    let knobCenter = vec2<f32>(u.knob_x, 14.0);
    let knobDist = length(pixel - knobCenter);
    let knobRadius = 11.0;

    let kShadow = (1.0 - smoothstep(knobRadius, knobRadius + 2.5, knobDist)) * 0.35;
    col = mix(col, vec3<f32>(0.0), kShadow);

    let kSphere = 1.0 - clamp(knobDist / knobRadius, 0.0, 1.0) * 0.15;
    var knobColor = vec3<f32>(0.98, 0.98, 1.0) * kSphere;

    let dDot = length(pixel - knobCenter);
    let dotLight = 1.0 - smoothstep(2.0, 2.7, dDot);

    var dotCol: vec3<f32>;
    if (u.context < 0.5) {
        dotCol = mix(vec3<f32>(0.6, 0.65, 0.7), vec3<f32>(0.1, 0.85, 0.45), u.progress);
    } else if (u.context < 1.5) {
        dotCol = mix(vec3<f32>(0.6, 0.65, 0.7), vec3<f32>(0.1, 0.65, 0.95), u.progress);
    } else {
        dotCol = mix(vec3<f32>(0.6, 0.65, 0.7), vec3<f32>(0.65, 0.4, 0.95), u.progress);
    }

    knobColor = mix(knobColor, dotCol, dotLight);

    let knobAA = 1.0 - smoothstep(knobRadius - 0.6, knobRadius + 0.4, knobDist);
    col = mix(col, knobColor, knobAA);

    return vec4<f32>(col * borderAA, borderAA);
}
