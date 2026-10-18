// flag.wgsl — the locale flag's aerodynamic wind warp.
//
// WGSL port of core/utils/canvas/widgets/flag/shaders.ts — constants
// preserved 1:1 so pixels are identical under the same uniforms.
//
// ════ THE WIND MODEL (read this before touching constants) ════
//
// A real flag is constrained at the pole and free at the fly edge.
// The math mirrors that:
//   windPhase = uv.x * 9.5 - t * 3.2   → a traveling wave:
//     uv.x*9.5 = ~1.5 wavelengths across the flag's width,
//     t*3.2 = wave speed 3.2/9.5 ≈ 0.34 uv/s → ~3s per full ripple.
//   windAmp  = (0.20 + 0.80 * uv.x) * (0.016 + 0.010 * hover)
//     (0.20 + 0.80x) = the "constraint envelope": 20% motion at the
//     pole (it's never truly rigid — cloth), 100% at the fly edge.
//     (0.016 + 0.010h) = base amplitude 1.6% of flag height, rising to
//     2.6% on hover — hover is additive excitement, not a state switch.
//   Warp applies uv.y += sin(phase)*amp  (vertical ripple) and
//   uv.x += cos(phase)*amp*0.35 (a foreshortening term — fabric slides
//   horizontally 35% as much as it lifts, which is what makes it read
//   as cloth instead of a static wave image).
//
// Per-locale personality (anim_type 0..11) adds a second detuned
// `flutter` term + a `specular` sheen term per flag — each country's
// fabric weight reads differently. Specular is a luminance adder on the
// SAMPLED texture, scaled (0.4 + 0.6*hover) so sheen grows on hover.
//
// Uniform layout (f32s, flag.rs::UNIFORM_F32S = 12):
//   [0,1] res       widget px size (device px)
//   [2]   time      seconds
//   [3]   hover     0..1 eased hover amount
//   [4]   anim_type 0..11 (flag.rs::anims — EN/PT/ES/DE/HRK/CAS/…)
//   [5]   is_split  >0.5 → split-flag mode (paired-locale flags)
//   [6]   split_x   split seam position 0..1 (only when is_split)
//   [7]   pad       unused — keeps the block 16B-aligned per WGSL
//                   uniform rules (vec4 must start at a 16B boundary)
//   [8..11] rect    widget clip rect, canvas CSS px
//
// Bindings (must match lib.rs's flag bind-group construction):
//   0 uniform, 1 tex1, 2 tex2, 3 sampler — split flags sample tex1
//   for x<split_x and tex2 for x≥split_x; single flags bind tex1 twice.

struct Uniforms {
    res: vec2<f32>,
    time: f32,
    hover: f32,
    anim_type: f32,
    is_split: f32,
    split_x: f32,
    pad: f32,
    rect: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var tex1: texture_2d<f32>;
@group(0) @binding(2) var tex2: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // Normalize into the widget rect; y flips so top-left matches the
    // original image-coordinate convention.
    var uv = vec2<f32>(
        (frag.x - u.rect.x) / u.rect.z,
        (frag.y - u.rect.y) / u.rect.w,
    );

    let t = u.time;
    let h = u.hover;
    let anim = u.anim_type;

    var uvWarp = uv;
    var specular = 0.0;

    // Wind model: fixed at flagpole (x=0) → free flutter at edge (x=1).
    let windPhase = uv.x * 9.5 - t * 3.2;
    let windAmp = (0.20 + 0.80 * uv.x) * (0.016 + 0.010 * h);
    let baseWave = sin(windPhase);

    var flutter = 0.0;

    if (anim < 0.5) {
        // EN — stars shimmer
        flutter = sin(windPhase * 1.5) * 0.005 + cos(uv.y * 7.0 - t * 1.8) * 0.003;
        specular = (0.5 + 0.5 * sin(windPhase)) * 0.28;
    } else if (anim < 1.5) {
        // PT — rhombus solar glow
        flutter = sin(windPhase * 1.3) * 0.006 + sin(uv.y * 6.0 - t * 1.5) * 0.003;
        specular = (0.5 + 0.5 * cos(windPhase)) * 0.30;
    } else if (anim < 2.5) {
        // ES — flamenco silk
        flutter = sin(windPhase + uv.y * 4.0) * 0.007;
        specular = (0.5 + 0.5 * sin(windPhase)) * 0.32;
    } else if (anim < 3.5) {
        // DE — cross-ribbon
        flutter = cos(windPhase * 1.2) * 0.005 + sin(uv.y * 5.0 - t * 1.6) * 0.003;
        specular = (0.5 + 0.5 * sin(windPhase)) * 0.25;
    } else if (anim < 4.5) {
        // HRK — harmonic wave
        flutter = sin(windPhase * 1.4) * 0.006;
        specular = (0.5 + 0.5 * sin(windPhase)) * 0.28;
    } else if (anim < 5.5) {
        // CAS — Sol de Mayo rays
        flutter = sin(windPhase * 1.2) * 0.005 + cos(uv.y * 6.0 - t * 2.0) * 0.003;
        specular = (0.5 + 0.5 * cos(windPhase)) * 0.35;
    } else if (anim < 6.5) {
        // RIV — river reflection
        flutter = sin(windPhase * 1.3) * 0.007;
        specular = (0.5 + 0.5 * sin(windPhase)) * 0.30;
    } else if (anim < 7.5) {
        // GN — horizontal surge
        flutter = cos(windPhase * 1.1) * 0.006;
        specular = (0.5 + 0.5 * sin(windPhase)) * 0.28;
    } else if (anim < 8.5) {
        // IT — mediterranean silk
        flutter = sin(windPhase * 1.2) * 0.006 + sin(uv.y * 5.0 - t * 1.5) * 0.003;
        specular = (0.5 + 0.5 * sin(windPhase)) * 0.30;
    } else if (anim < 9.5) {
        // RU — aurora shimmer
        flutter = sin(windPhase * 1.1) * 0.007 + cos(uv.y * 7.0 - t * 1.8) * 0.003;
        specular = (0.5 + 0.5 * cos(windPhase)) * 0.35;
    } else if (anim < 10.5) {
        // FR — satin ripple
        flutter = sin(windPhase * 1.3) * 0.006;
        specular = (0.5 + 0.5 * sin(windPhase)) * 0.28;
    } else {
        // TLN — venetian pulse
        flutter = sin(windPhase * 1.2) * 0.006 + cos(uv.y * 6.0 - t * 1.9) * 0.003;
        specular = (0.5 + 0.5 * sin(windPhase)) * 0.30;
    }

    uvWarp.y = uvWarp.y + (baseWave * windAmp) + (flutter * windAmp);
    uvWarp.x = uvWarp.x + cos(windPhase) * (windAmp * 0.35);

    var col = vec4<f32>(0.0);

    if (u.is_split > 0.5) {
        // Split mode: the flag's left half shows tex1 remapped into
        // [0,0.5] and the right half tex2 into [0.5,1.0] — each half of
        // the OUTPUT samples the FULL texture but only its left/right
        // half in uv space. The 0.5 multipliers are the half-atlas
        // packing; `clamp(..., 0.005, 0.995)` is an edge guard: warp can
        // push samples outside [0,1], and unclamped wrapping would smear
        // the opposite edge's pixels (repeat addressing is off anyway,
        // but the clamp makes the intent explicit and driver-proof).
        if (uvWarp.x < u.split_x) {
            var uv1 = vec2<f32>(uvWarp.x / u.split_x * 0.5, uvWarp.y);
            uv1 = clamp(uv1, vec2<f32>(0.005), vec2<f32>(0.995));
            col = textureSample(tex1, samp, uv1);
        } else {
            var uv2 = vec2<f32>(0.5 + (uvWarp.x - u.split_x) / (1.0 - u.split_x) * 0.5, uvWarp.y);
            uv2 = clamp(uv2, vec2<f32>(0.005), vec2<f32>(0.995));
            col = textureSample(tex2, samp, uv2);
        }

        // The seam itself — a 1.5%-wide darkened dividing line so the
        // two flag halves read as stitched cloth, not a hard cut.
        // NOTE: computed on un-warped uv so the line stays straight
        // while the fabric ripples around it (a warped seam would wiggle).
        let lineDist = abs(uv.x - u.split_x);
        let lineEdge = 1.0 - smoothstep(0.0, 0.015, lineDist);
        col = mix(col, vec4<f32>(0.1, 0.1, 0.1, 1.0), lineEdge * 0.4);
    } else {
        let uvSingle = clamp(uvWarp, vec2<f32>(0.005), vec2<f32>(0.995));
        col = textureSample(tex1, samp, uvSingle);
    }

    // Specular sheen — ADDED luminance (not mixed): a moving highlight
    // across the fabric. (0.4 + 0.6*h): 40% sheen at rest, full on
    // hover. Values >1 are clamped by the output format — fine, the
    // specular coefficient ≤0.35 keeps rgb ≤ ~1.35 worst-case.
    col = vec4<f32>(col.rgb + vec3<f32>(specular * (0.4 + 0.6 * h)), col.a);

    // Rounded-corner mask — SDF of a rounded rect, 3px radius.
    // The math: `abs(pixel - res/2) - (res/2 - r)` is the classic
    // rounded-box SDF (distance outside the corner box, 0 inside);
    // `length` of that vector gives euclidean distance to the corner
    // arc; smoothstep over the last 1px of `r` produces a soft edge.
    // 3.0px is fixed (not a token) because flag corners are a physical
    // flag detail, not a theme dimension.
    let pixel = uv * vec2<f32>(u.rect.z, u.rect.w);
    let res = vec2<f32>(u.rect.z, u.rect.w);
    let cornerRadius = vec2<f32>(3.0, 3.0);
    let dCorner = max(abs(pixel - res * 0.5) - (res * 0.5 - cornerRadius), vec2<f32>(0.0));
    let cornerDist = length(dCorner);
    let alpha = 1.0 - smoothstep(cornerRadius.x - 1.0, cornerRadius.x, cornerDist);

    // Premultiplied output: rgb × a matches the pipeline's alpha-blend
    // state (src×1 + dst×(1-a)) — writing non-premultiplied colors with
    // premultiplied blending would double-darken semi-transparent texels.
    return vec4<f32>(col.rgb * (col.a * alpha), col.a * alpha);
}
