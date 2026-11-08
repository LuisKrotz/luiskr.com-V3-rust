// skeleton.wgsl — the loading skeleton's "data decoding" shimmer.
//
// WGSL port of core/utils/canvas/loaders/skeleton/shaders.ts.
// Each placeholder rect hosts a grid of tiny morphing 0/1 glyphs —
// content "decoding" into place; on resolve the glyphs collapse into
// the base fill and the layer fades.
//
// DATA MODEL — the per-rect sheet (unlike the other widgets, skeleton
// needs per-RECT data, so it gets a storage buffer beside uniforms):
//   uniform `uniforms` — global timing/resolve/ink + widget rect
//   storage sheet     — MAX_RECTS × 4 × vec4 rows per rect:
//     [i*4+0] rect_px    = rect x,y,w,h in widget px
//     [i*4+1] row_meta   = (corner_radius, cell_size, row_size, unused)
//     [i*4+2] base_rgba  = surface/base tint rgb + unused
//     [i*4+3] ink_rgba   = ink tint rgb + unused
//   JS writes it via `write_skel_sheet(id, f32s)` — 24×16 f32s max.
//
// Uniform layout (f32s, skeleton.rs::UNIFORM_F32S = 12):
//   [0,1] resolution_css design size px
//   [2]   time_seconds   seconds — morph cadence + density drift
//   [3]   resolve_amount 0..1 — 0=decoding … 1=resolved (content arrived)
//   [4]   ink_alpha      global glyph opacity
//   [5]   rect_count     live rect count (≤ MAX_RECTS)
//   [6,7] reserved       16B alignment pad
//   [8..11] clip_rect    widget clip rect, canvas DEVICE px
//
// Fragment-side chunk only — sd_round_box + vs_main come from
// gpu-mini's wgsl library. `hash_cell` stays local: its 123.34/456.21
// self-product field is this shader's own seed (a different seed
// re-rolls every glyph's cadence — a pixel change, not a dedup).

const MAX_RECTS: u32 = 24u;

struct Uniforms {
    resolution_css: vec2<f32>,
    time_seconds: f32,
    resolve_amount: f32,
    ink_alpha: f32,
    rect_count: f32,
    reserved0: f32,
    reserved1: f32,
    clip_rect: vec4<f32>,
}

// rect_px | row_meta | base_rgba | ink_rgba — 4 × vec4 per rect row.
struct Sheet {
    data: array<vec4<f32>, 96>, // MAX_RECTS * 4
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> sheet: Sheet;

// ── Tunables ─────────────────────────────────────────────────────────

// Rect edge AA — pixels past CULL_DIST are skipped (they can't
// contribute to this rect).
const CULL_DIST_PX: f32 = 1.0;
const EDGE_AA_PX: f32 = 1.0;

// Row-rect mode (text-line skeletons): tall thin glyphs + a ragged
// right edge — `ragged` ∈ [0.72,1.0] per row so lines read as ragged-
// right text. Row-height caps glyph size at 62% of the line height.
const RAGGED_LO: f32 = 0.72;
const RAGGED_SPAN: f32 = 0.28;
const ROW_GLYPH_CAP: f32 = 0.62;
const ROW_SEED_SALT: f32 = 3.1;

// Per-cell decorrelation — `i × CELL_SEED_STRIDE` decorrelates cells
// across rects (same cell id in two rects gets different seeds).
const CELL_SEED_STRIDE: f32 = 17.0;
const DENSITY_CELL_SPREAD: f32 = 1.7;

// The morph state machine — each cell flips 0↔1 on its own cadence.
// `phase` = time×cadence + per-cell offset; floor(phase) is the flip
// index; hashing it gives that flip's target digit and hashing
// floor(phase)−1 the previous digit; `morph` (first MORPH_WINDOW of
// each flip's interval) crossfades the two SDFs — mix() between two
// distance fields produces smooth shape interpolation for free.
const CADENCE_LO: f32 = 0.35;
const CADENCE_SPAN: f32 = 0.5;
const PHASE_OFFSET: f32 = 7.0;
const MORPH_WINDOW: f32 = 0.35;
const STATE_HASH_A: f32 = 12.9898;
const STATE_HASH_B: f32 = 78.233;
const STATE_HASH_C: f32 = 43758.5453;
const STATE_GATE: f32 = 0.5;

// Glyph silhouette AA — the SDF→alpha soft edge.
const EDGE_SOFTNESS: f32 = 0.32;

// Density gate — a per-cell drop chance keeps the field sparse: text
// rows ~28% filled (breathing ±4%), media cells ~48% (breathing
// ±12%); rows dim to 42% alpha since text lines should read lighter
// than media blocks.
const TEXT_DENSITY: f32 = 0.28;
const TEXT_DENSITY_SWAY: f32 = 0.04;
const MEDIA_DENSITY: f32 = 0.48;
const MEDIA_DENSITY_SWAY: f32 = 0.12;
const DENSITY_SWEEP_SPEED: f32 = 0.35;
const TEXT_ALPHA: f32 = 0.42;

// Ink tones — two mixes between the rect's surface and ink tints
// drifting by position+time so the field reads as liquid decode.
const INK_MIX_LO: f32 = 0.68;
const INK_MIX_HI: f32 = 0.86;
const DRIFT_SPEED: f32 = 0.6;
const DRIFT_SPATIAL: f32 = 0.012;
const DRIFT_PHASE: f32 = 2.0;

// Resolve choreography — two overlapping windows: glyphs merge into
// the base fill during resolve 0→0.6 (collapse), then the whole rect
// fades out across 0.35→1.0. The overlap means cells "finish
// decoding" just before the layer dissolves.
const COLLAPSE_HI: f32 = 0.6;
const FADE_LO: f32 = 0.35;
const FADE_HI: f32 = 1.0;

// Glyph "0" — an elliptical ring SDF: |length(stretched point) −
// ring_radius| − stroke. The X_STRETCH makes the ring oval (digits
// are taller than wide); radius/stroke are fractions of cell size —
// tuned to read as a "0" at 6–10px cell sizes.
const GLYPH_ZERO_X_STRETCH: f32 = 1.25;
const GLYPH_ZERO_RADIUS: f32 = 0.30;
const GLYPH_ZERO_STROKE: f32 = 0.075;

// Glyph "1" — union of two rounded boxes: the vertical bar + the
// diagonal flag serif top-left. Same "reads-at-tiny-size" tuning —
// proportions are fractions of cell size.
const GLYPH_ONE_BAR_X: f32 = 0.04;
const GLYPH_ONE_BAR_HALF_W: f32 = 0.075;
const GLYPH_ONE_BAR_HALF_H: f32 = 0.33;
const GLYPH_ONE_BAR_RADIUS: f32 = 0.05;
const GLYPH_ONE_FLAG_X: f32 = -0.07;
const GLYPH_ONE_FLAG_Y: f32 = -0.22;
const GLYPH_ONE_FLAG_HALF_W: f32 = 0.12;
const GLYPH_ONE_FLAG_HALF_H: f32 = 0.06;
const GLYPH_ONE_FLAG_RADIUS: f32 = 0.03;

/// `hash_cell` — a coordinate→[0,1) hash with better mixing than
/// sin·fract (the x·y self-product spreads entropy through both
/// channels). Used for per-cell glyph state + row raggedness — must
/// be deterministic or cells would boil every frame. THIS shader's
/// own seed — the shared library's hash field would re-roll every
/// glyph's cadence (a pixel change, not a dedup).
fn hash_cell(point: vec2<f32>) -> f32 {
    var folded = fract(point * vec2<f32>(123.34, 456.21));
    folded = folded + vec2<f32>(dot(folded, folded + vec2<f32>(45.32)));
    return fract(folded.x * folded.y);
}

/// `glyph_zero` — the elliptical-ring "0" SDF (see constants above).
fn glyph_zero(point: vec2<f32>, size: f32) -> f32 {
    return abs(length(point * vec2<f32>(GLYPH_ZERO_X_STRETCH, 1.0)) - size * GLYPH_ZERO_RADIUS) - size * GLYPH_ZERO_STROKE;
}

/// `glyph_one` — the bar+flag "1" SDF (see constants above).
fn glyph_one(point: vec2<f32>, size: f32) -> f32 {
    let bar = sd_round_box(
        point - vec2<f32>(size * GLYPH_ONE_BAR_X, 0.0),
        vec2<f32>(size * GLYPH_ONE_BAR_HALF_W, size * GLYPH_ONE_BAR_HALF_H),
        size * GLYPH_ONE_BAR_RADIUS,
    );
    let flag = sd_round_box(
        point - vec2<f32>(size * GLYPH_ONE_FLAG_X, size * GLYPH_ONE_FLAG_Y),
        vec2<f32>(size * GLYPH_ONE_FLAG_HALF_W, size * GLYPH_ONE_FLAG_HALF_H),
        size * GLYPH_ONE_FLAG_RADIUS,
    );
    return min(bar, flag);
}

@fragment
fn fs_main(@builtin(position) pixel_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // Rect-local device px, y down (same convention as the original).
    let pixel = vec2<f32>(pixel_coord.x - uniforms.clip_rect.x, pixel_coord.y - uniforms.clip_rect.y);
    var color = vec4<f32>(0.0);

    for (var rect_i = 0u; rect_i < MAX_RECTS; rect_i = rect_i + 1u) {
        if (f32(rect_i) >= uniforms.rect_count) {
            break;
        }

        let rect_px = sheet.data[rect_i * 4u];
        // `meta` is a WGSL reserved keyword — the row's second vec4
        // carries (corner_radius, cell_size, row_size, unused).
        let row_meta = sheet.data[rect_i * 4u + 1u];
        let base_rgba = sheet.data[rect_i * 4u + 2u];
        let ink_rgba = sheet.data[rect_i * 4u + 3u];

        let half_extents = rect_px.zw * 0.5;
        let center_px = rect_px.xy + half_extents;
        let border_dist = sd_round_box(
            pixel - center_px,
            half_extents,
            min(row_meta.x, min(half_extents.x, half_extents.y)),
        );

        if (border_dist > CULL_DIST_PX) {
            continue;
        }

        let edge = 1.0 - smoothstep(-EDGE_AA_PX, EDGE_AA_PX, border_dist);

        let cell_size = row_meta.y;
        let row_size = row_meta.z;
        let local = pixel - rect_px.xy;

        // Cell addressing: `local` is rect-local px; `cell_size` is
        // cell size (px). cell_id = which cell; cell_local = position
        // inside the cell centered at 0 — the SDF domain for the glyph.
        var row_mask = 1.0;
        var cell_id = floor(local / vec2<f32>(cell_size));
        var cell_local = (fract(local / vec2<f32>(cell_size)) - vec2<f32>(0.5)) * cell_size;
        var glyph_size = cell_size;

        // "Row" rects (text-line skeletons) replace the cell grid with
        // tall thin glyphs and a ragged right edge.
        if (row_size > 0.0) {
            let row_index = floor(local.y / row_size);
            cell_id.y = row_index;
            cell_local.y = local.y - row_size * (row_index + 0.5);
            glyph_size = min(cell_size, row_size * ROW_GLYPH_CAP);
            let ragged = RAGGED_LO + RAGGED_SPAN * hash_cell(vec2<f32>(row_index, f32(rect_i) * ROW_SEED_SALT));
            row_mask = 1.0 - smoothstep(ragged * rect_px.z - cell_size, ragged * rect_px.z, local.x);
        }

        // Per-cell hash — decorrelated across rects.
        let cell_seed = hash_cell(cell_id + vec2<f32>(f32(rect_i) * CELL_SEED_STRIDE));

        // ── The morph state machine (see constants above) ────────────
        let cadence = CADENCE_LO + cell_seed * CADENCE_SPAN;
        let phase = uniforms.time_seconds * cadence + cell_seed * PHASE_OFFSET;
        let digit_now = step(STATE_GATE, fract(sin(floor(phase) * STATE_HASH_A + cell_seed * STATE_HASH_B) * STATE_HASH_C));
        let morph = smoothstep(0.0, MORPH_WINDOW, fract(phase));
        let digit_prev = step(STATE_GATE, fract(sin((floor(phase) - 1.0) * STATE_HASH_A + cell_seed * STATE_HASH_B) * STATE_HASH_C));
        let shape = mix(digit_prev, digit_now, morph);

        let dist_zero = glyph_zero(cell_local, glyph_size);
        let dist_one = glyph_one(cell_local, glyph_size);
        let dist_glyph = mix(dist_zero, dist_one, shape);
        var glyph = 1.0 - smoothstep(0.0, EDGE_SOFTNESS, dist_glyph);

        // Density gate — a per-cell drop chance keeps the field sparse.
        let text_density = TEXT_DENSITY + TEXT_DENSITY_SWAY * sin(uniforms.time_seconds * DENSITY_SWEEP_SPEED + f32(rect_i));
        let media_density = MEDIA_DENSITY + MEDIA_DENSITY_SWAY * sin(uniforms.time_seconds * DENSITY_SWEEP_SPEED + f32(rect_i));
        let density = select(media_density, text_density, row_size > 0.0);
        let text_alpha = select(1.0, TEXT_ALPHA, row_size > 0.0);
        glyph = glyph * step(1.0 - density, hash_cell(cell_id * DENSITY_CELL_SPREAD + vec2<f32>(f32(rect_i))));
        glyph = glyph * row_mask * text_alpha;

        // Ink tones close to the rect's surface.
        let ink_lo = mix(ink_rgba.rgb, base_rgba.rgb, vec3<f32>(INK_MIX_LO));
        let ink_hi = mix(ink_rgba.rgb, base_rgba.rgb, vec3<f32>(INK_MIX_HI));
        let drift = 0.5 + 0.5 * sin(uniforms.time_seconds * DRIFT_SPEED + local.x * DRIFT_SPATIAL + cell_seed * DRIFT_PHASE);
        let ink = mix(ink_lo, ink_hi, vec3<f32>(drift));

        // Resolve choreography — collapse then fade (see constants).
        let collapse = 1.0 - smoothstep(0.0, COLLAPSE_HI, uniforms.resolve_amount);
        let fade = 1.0 - smoothstep(FADE_LO, FADE_HI, uniforms.resolve_amount);

        let rgb = mix(base_rgba.rgb, ink, vec3<f32>(glyph * uniforms.ink_alpha * collapse));
        let alpha = edge * fade;
        color = vec4<f32>(rgb * alpha, alpha);
    }

    return color;
}
