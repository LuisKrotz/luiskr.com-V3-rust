// skeleton.wgsl — the loading skeleton's "data decoding" shimmer.
//
// WGSL port of core/utils/canvas/loaders/skeleton/shaders.ts.
// Each placeholder rect hosts a grid of tiny morphing 0/1 glyphs —
// content "decoding" into place; on resolve the glyphs collapse into
// the base fill and the layer fades.
//
// DATA MODEL — the per-rect sheet (unlike the other widgets, skeleton
// needs per-RECT data, so it gets a storage buffer beside uniforms):
//   uniform `u`   — global timing/resolve/ink + widget rect
//   storage sheet — MAX_RECTS × 4 × vec4 rows per rect:
//     [i*4+0] r     = rect x,y,w,h in widget px
//     [i*4+1] meta  = (corner_radius, cell_size, row_size, unused)
//     [i*4+2] sbase = surface/base tint rgb + unused
//     [i*4+3] sink  = ink tint rgb + unused
//   JS writes it via `write_skel_sheet(id, f32s)` — 24×16 f32s max.
//
// Uniform layout (f32s, skeleton.rs::UNIFORM_F32S = 12):
//   [0,1] res        design size px
//   [2]   time       seconds — morph cadence + density drift
//   [3]   resolve    0..1 — 0=decoding … 1=resolved (content arrived)
//   [4]   ink_alpha  global glyph opacity
//   [5]   count      live rect count (≤ MAX_RECTS)
//   [6,7] pad        16B alignment pad
//   [8..11] rect     widget clip rect, canvas DEVICE px

const MAX_RECTS: u32 = 24u;

struct Uniforms {
    res: vec2<f32>,
    time: f32,
    resolve: f32,
    ink_alpha: f32,
    count: f32,
    pad0: f32,
    pad1: f32,
    rect: vec4<f32>,
}

// rects | meta | sbase | sink — 4 × vec4 per rect row.
struct Sheet {
    data: array<vec4<f32>, 96>, // MAX_RECTS * 4
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> sheet: Sheet;

// hash21 — a coordinate→[0,1) hash with better mixing than sin·fract
// (the x·y self-product spreads entropy through both channels). Used
// for per-cell glyph state + row raggedness — must be deterministic or
// cells would boil every frame.
fn hash21(p_in: vec2<f32>) -> f32 {
    var p = fract(p_in * vec2<f32>(123.34, 456.21));
    p = p + vec2<f32>(dot(p, p + vec2<f32>(45.32)));
    return fract(p.x * p.y);
}

// sdRoundBox — the standard rounded-rectangle SDF (iq). `q = abs(p) -
// b + r` exploits the rect's symmetry: fold to the +x+y quadrant, then
// `min(max(q.x,q.y),0)` is the interior distance and `length(max(q,0))`
// the exterior — the sum is signed distance to the rounded boundary.
fn sdRoundBox(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
}

// Glyph "0" — an elliptical ring SDF: |length(stretched q) − ringR| −
// thickness. The 1.25 x-stretch makes the ring oval (digits are taller
// than wide); s·0.30 radius, s·0.075 stroke — tuned to read as a "0" at
// 6–10px cell sizes.
fn glyphZero(q: vec2<f32>, s: f32) -> f32 {
    return abs(length(q * vec2<f32>(1.25, 1.0)) - s * 0.30) - s * 0.075;
}

// Glyph "1" — union (min) of two rounded boxes: the vertical bar + the
// diagonal flag serif top-left. Same "reads-at-tiny-size" tuning logic
// as glyphZero — proportions are fractions of cell size `s`.
fn glyphOne(q: vec2<f32>, s: f32) -> f32 {
    let bar = sdRoundBox(q - vec2<f32>(s * 0.04, 0.0), vec2<f32>(s * 0.075, s * 0.33), s * 0.05);
    let flag = sdRoundBox(q - vec2<f32>(-s * 0.07, -s * 0.22), vec2<f32>(s * 0.12, s * 0.06), s * 0.03);
    return min(bar, flag);
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
    // Rect-local device px, y down (same convention as the original).
    let p = vec2<f32>(frag.x - u.rect.x, frag.y - u.rect.y);
    var color = vec4<f32>(0.0);

    for (var i = 0u; i < MAX_RECTS; i = i + 1u) {
        if (f32(i) >= u.count) {
            break;
        }

        let r = sheet.data[i * 4u];
        let meta = sheet.data[i * 4u + 1u];
        let sbase = sheet.data[i * 4u + 2u];
        let sink = sheet.data[i * 4u + 3u];

        let hs = r.zw * 0.5;
        let c = r.xy + hs;
        let d = sdRoundBox(p - c, hs, min(meta.x, min(hs.x, hs.y)));

        if (d > 1.0) {
            continue;
        }

        let edge = 1.0 - smoothstep(-1.0, 1.0, d);

        let cell = meta.y;
        let row = meta.z;
        let local = p - r.xy;

        // Cell addressing: `local` is rect-local px; `cell` is cell size
        // (px). cellId = which cell; q = position inside the cell
        // centered at 0 — the SDF domain for the glyph.
        var rowMask = 1.0;
        var cellId = floor(local / vec2<f32>(cell));
        var q = (fract(local / vec2<f32>(cell)) - vec2<f32>(0.5)) * cell;
        var glyphSize = cell;

        // "Row" rects (text-line skeletons) replace the cell grid with
        // tall thin glyphs and a ragged right edge — `ragged`∈[0.72,1.0]
        // per row via hash → lines look like ragged-right text, not a
        // full-width block. Row-height caps glyph size at 62% so glyphs
        // stay proportionate to the line height.
        if (row > 0.0) {
            let rowIdx = floor(local.y / row);
            cellId.y = rowIdx;
            q.y = local.y - row * (rowIdx + 0.5);
            glyphSize = min(cell, row * 0.62);
            let ragged = 0.72 + 0.28 * hash21(vec2<f32>(rowIdx, f32(i) * 3.1));
            rowMask = 1.0 - smoothstep(ragged * r.z - cell, ragged * r.z, local.x);
        }

        // Per-cell hash — `i * 17.0` decorrelates cells across rects
        // (same cell id in two rects gets different seeds).
        let h = hash21(cellId + vec2<f32>(f32(i) * 17.0));

        // ── The morph state machine ─────────────────────────────────
        // Each cell flips 0↔1 on its own cadence (0.35–0.85 flips/s via
        // h). `phase` = time×cadence + per-cell offset (h*7.0). floor(
        // phase) is the flip index; hashing it gives that flip's target
        // digit — hashing floor(phase)-1 gives the PREVIOUS digit, and
        // `morph` (first 35% of each flip's interval) crossfades the two
        // SDFs — mix() between two distance fields produces a smooth
        // shape interpolation for free.
        let cadence = 0.35 + h * 0.5;
        let phase = u.time * cadence + h * 7.0;
        let state = step(0.5, fract(sin(floor(phase) * 12.9898 + h * 78.233) * 43758.5453));
        let morph = smoothstep(0.0, 0.35, fract(phase));
        let prevState = step(0.5, fract(sin((floor(phase) - 1.0) * 12.9898 + h * 78.233) * 43758.5453));
        let shape = mix(prevState, state, morph);

        let g0 = glyphZero(q, glyphSize);
        let g1 = glyphOne(q, glyphSize);
        let g = mix(g0, g1, shape);
        var glyph = 1.0 - smoothstep(0.0, 0.32, g); // EDGE_SOFTNESS

        // Density gate: a per-cell drop chance keeps the field sparse.
        // Text rows ~28% filled (breathing ±4%), media cells ~48%
        // (breathing ±12%) — rows are also dimmed to 42% alpha since
        // text lines should read lighter than media blocks.
        let textDensity = 0.28 + 0.04 * sin(u.time * 0.35 + f32(i));
        let mediaDensity = 0.48 + 0.12 * sin(u.time * 0.35 + f32(i));
        let density = select(mediaDensity, textDensity, row > 0.0);
        let textAlpha = select(1.0, 0.42, row > 0.0); // TEXT_ALPHA
        glyph = glyph * step(1.0 - density, hash21(cellId * 1.7 + vec2<f32>(f32(i))));
        glyph = glyph * rowMask * textAlpha;

        // Ink tones close to the rect's surface (INK_SURFACE_MIX_*).
        let inkA = mix(sink.rgb, sbase.rgb, vec3<f32>(0.68));
        let inkB = mix(sink.rgb, sbase.rgb, vec3<f32>(0.86));
        let drift = 0.5 + 0.5 * sin(u.time * 0.6 + local.x * 0.012 + h * 2.0);
        let ink = mix(inkA, inkB, vec3<f32>(drift));

        // Resolve choreography — two overlapping windows: glyphs merge
        // into the base fill during resolve 0→0.6 (collapse), then the
        // whole rect fades out across 0.35→1.0. The overlap means cells
        // "finish decoding" just before the layer dissolves — content
        // arriving mid-collapse still looks intentional.
        let collapse = 1.0 - smoothstep(0.0, 0.6, u.resolve);
        let fade = 1.0 - smoothstep(0.35, 1.0, u.resolve);

        let rgb = mix(sbase.rgb, ink, vec3<f32>(glyph * u.ink_alpha * collapse));
        let a = edge * fade;
        color = vec4<f32>(rgb * a, a);
    }

    return color;
}
