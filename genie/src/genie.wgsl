// genie.wgsl — expand-modal genie warp: textured mesh deform + the
// dock-flash accent (one pipeline, one shared uniform).
//
// Field math mirrors `core/tokens/motion/genie.ts` + the 2D backend
// (`genie-canvas.ts`) exactly — the per-row staggered eased lerp the
// ui-layouts mac-genie reference implements. The field always warps the
// MODAL rect (S=`modal_rect`) toward the FIGURE rect (D=`figure_rect`);
// the open transition plays the identical funnel in reverse via the
// driver-flipped `progress`. Rows index in down-space (down_row_t 0 =
// far edge, 1 = figure-facing edge) and each row owns two staggered
// clocks:
//   x_stagger = (1−down_row_t)·STAGGER_X
//   x_eased   = easeInOutCubic(clamp((p−x_stagger)/(1−x_stagger)))
//   y_stagger = (1−down_row_t)·STAGGER_Y
//   y_eased   = easeInQuad(    clamp((p−y_stagger)/(1−y_stagger)))
//     x' = lerp(S.x + col·S.w, D.x + col·D.w, x_eased)
//     y' = farY + sgn·lerp(down_row_t·H, landing_far + down_row_t·iH, y_eased)
// landing_far is the far-edge → figure-far-edge distance, iH the figure
// thickness — every row lands on its proportional row inside the card
// at p=1 (the canonical 0-height mouth only suits a dock icon, so the
// warped sheet converges exactly onto the figure rect the real card
// then paints). The stagger is the whole effect: leading rows move
// first while far rows wait out their delay, so the funnel's S-curved
// walls emerge from the easing alone. Positions arrive in CSS px and
// are scaled to device px by `dpr` before the NDC projection.
//
// This file keeps its OWN vertex stage (vs_mesh synthesizes the grid
// from vertex_index), so it composes with the gpu-mini helpers prelude
// only — no shared fullscreen triangle.
//
// Uniform layout (24 f32s — mirrors GPU_WIDGET_UNIFORM_F32S[GENIE]):
//   [0,1]    resolution_dev  canvas DEVICE px
//   [2]      time_seconds    seconds
//   [3]      progress        funnel progress 0..1 (linear — row eases carry it)
//   [4..8]   modal_rect      modal rect (x,y,w,h) CSS px
//   [8..12]  figure_rect     figure rect
//   [12,13]  reserved        (kept for the 24-f32 ABI)
//   [14]     dpr             device/css ratio
//   [15]     flags           bit0 opening · bit1 nearBottom · bit2 hasTex
//   [16..20] ghost_tint      accent rgba (untex'd ghost fill)
//   [20..24] uv_rect         texture uv scale.xy + offset.xy (cover map)
//
// Bindings:  0 uniform · 1 texture · 2 sampler

const ROWS: u32 = 192u;       // mesh rows — ≈ the 2D backend's adaptive count on a 900px sheet
const QUADS: u32 = 191u;      // ROWS-1 triangle-list quads (6 verts each)
const STAGGER_X: f32 = 0.65;  // horizontal row stagger — GENIE_WARP.STAGGER_X
const STAGGER_Y: f32 = 0.2;   // vertical row stagger — GENIE_WARP.STAGGER_Y
const GLOW_START: f32 = 0.75; // dock-flash window — GENIE_WARP.GLOW_START
const GLOW_ALPHA: f32 = 0.3;  // dock-flash peak — GENIE_WARP.GLOW_ALPHA
const GLOW_RADIUS_FRAC: f32 = 0.5; // flash radius = min(D.w,D.h)·frac
const GHOST_ALPHA: f32 = 0.22; // untextured silhouette fill while the bitmap lags
const QUAD_VERTS: u32 = 6u;    // verts per quad cell (two triangles)
const FLAG_NEAR_BOTTOM: u32 = 2u; // config.w bit: figure sits below the modal
const FLAG_HAS_TEXTURE: u32 = 4u; // config.w bit: figure texture uploaded

struct Uniforms {
    resolution_dev: vec2<f32>,
    time_seconds: f32,
    progress: f32,
    modal_rect: vec4<f32>,
    figure_rect: vec4<f32>,
    config: vec4<f32>,
    ghost_tint: vec4<f32>,
    uv_rect: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var figure_texture: texture_2d<f32>;
@group(0) @binding(2) var figure_sampler: sampler;

struct MeshOut {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_uv: vec2<f32>,
    @location(1) shade: f32,
    // Fragment-space CSS-px position — the dock flash shades by distance
    // to the figure's center, not by texture coords.
    @location(2) frag_css: vec2<f32>,
}

// The genie field — returns (x, y) CSS px + the row's down-space
// down_row_t. `column` is the horizontal position 0..1 left→right,
// `row` the vertical position 0..1 in sheet order (top→bottom). See
// the module header for the staggered-clock math.
fn warp_field(column: f32, row: f32) -> vec3<f32> {
    let flags = u32(uniforms.config.w);
    let near_bottom = (flags & FLAG_NEAR_BOTTOM) != 0u;

    let progress = uniforms.progress;

    // Down-space row coordinate — 0 at the sheet's far edge, 1 at the
    // figure-facing edge (the bottom row when the figure sits below).
    let down_row_t = select(1.0 - row, row, near_bottom);

    // Staggered horizontal clock — leading rows start at progress=0
    // while the far edge waits out (1−down_row_t)·STAGGER_X of the run.
    let x_stagger = (1.0 - down_row_t) * STAGGER_X;
    let x_progress = clamp((progress - x_stagger) / (1.0 - x_stagger), 0.0, 1.0);
    let x_eased = ease_in_out_cubic(x_progress);

    // Staggered vertical clock — quadratic ease-in, the dock suck.
    let y_stagger = (1.0 - down_row_t) * STAGGER_Y;
    let y_progress = clamp((progress - y_stagger) / (1.0 - y_stagger), 0.0, 1.0);
    let y_eased = y_progress * y_progress;

    // modal_rect.xy = x,y — modal_rect.z is WIDTH, .w is HEIGHT.
    let modal_width = uniforms.modal_rect.z;
    let modal_height = uniforms.modal_rect.w;

    // Landing rect in down-space — landing_far is the far-edge →
    // figure far-edge distance, figure_thickness the figure's height.
    let landing_far = select(
        uniforms.modal_rect.y + uniforms.modal_rect.w - (uniforms.figure_rect.y + uniforms.figure_rect.w),
        uniforms.figure_rect.y - uniforms.modal_rect.y,
        near_bottom,
    );
    let figure_thickness = uniforms.figure_rect.w;

    // Row down_row_t lands on landing_far + down_row_t·figure_thickness
    // at y_eased=1 — the warped sheet converges exactly onto the card
    // it hands off to.
    let row_height = down_row_t * modal_height;
    let y_down = row_height + (landing_far + down_row_t * figure_thickness - row_height) * y_eased;
    let y_css = select(
        uniforms.modal_rect.y + uniforms.modal_rect.w - y_down,
        uniforms.modal_rect.y + y_down,
        near_bottom,
    );

    let x_css = uniforms.modal_rect.x + column * modal_width
        + (uniforms.figure_rect.x + column * uniforms.figure_rect.z - uniforms.modal_rect.x - column * modal_width) * x_eased;

    return vec3<f32>(x_css, y_css, down_row_t);
}

@vertex
fn vs_mesh(@builtin(vertex_index) vertex_index: u32) -> MeshOut {
    // 6 verts per quad: 0=(0,0) 1=(1,0) 2=(0,1) 3=(0,1) 4=(1,0) 5=(1,1)
    let quad_index = vertex_index / QUAD_VERTS;
    let corner_index = vertex_index % QUAD_VERTS;
    var corner_column = 0.0;
    var corner_row_offset = 0.0;

    if corner_index == 1u || corner_index == 4u || corner_index == 5u {
        corner_column = 1.0;
    }
    if corner_index == 2u || corner_index == 3u || corner_index == 5u {
        corner_row_offset = 1.0;
    }

    let row = (f32(quad_index) + corner_row_offset) / f32(QUADS);
    let field = warp_field(corner_column, row);
    let device_x = field.x * uniforms.config.z;
    let device_y = field.y * uniforms.config.z;

    var output: MeshOut;
    output.position = vec4<f32>(
        device_x / uniforms.resolution_dev.x * 2.0 - 1.0,
        1.0 - device_y / uniforms.resolution_dev.y * 2.0,
        0.0,
        1.0,
    );
    output.tex_uv = vec2<f32>(corner_column, row) * uniforms.uv_rect.xy + uniforms.uv_rect.zw;
    output.shade = field.z;
    output.frag_css = field.xy;

    return output;
}

@fragment
fn fs_mesh(input: MeshOut) -> @location(0) vec4<f32> {
    let has_texture = (u32(uniforms.config.w) & FLAG_HAS_TEXTURE) != 0u;

    var color: vec4<f32>;

    if has_texture {
        let texel = textureSample(figure_texture, figure_sampler, input.tex_uv);
        // Premultiply for the One/OneMinusSrcAlpha blend.
        color = vec4<f32>(texel.rgb * texel.a, texel.a);
    } else {
        // Untex'd ghost — the accent-tinted silhouette while the
        // bitmap lags behind the morph's first frames.
        color = vec4<f32>(uniforms.ghost_tint.rgb * GHOST_ALPHA, GHOST_ALPHA);
    }

    // Dock flash — the white radial bloom at the figure's center for
    // the run's last quarter (the reference's suck punctuation);
    // quadratic ease-out on the progress tail.
    let glow_progress = (uniforms.progress - GLOW_START) / (1.0 - GLOW_START);

    if glow_progress > 0.0 && glow_progress < 1.0 {
        let glow_center = vec2<f32>(
            uniforms.figure_rect.x + uniforms.figure_rect.z * 0.5,
            uniforms.figure_rect.y + uniforms.figure_rect.w * 0.5,
        );
        let glow_radius = min(uniforms.figure_rect.z, uniforms.figure_rect.w) * GLOW_RADIUS_FRAC;
        let glow_dist_t = clamp(length(input.frag_css - glow_center) / max(glow_radius, 0.1), 0.0, 1.0);
        let glow_alpha = (1.0 - (1.0 - glow_progress) * (1.0 - glow_progress)) * GLOW_ALPHA * (1.0 - glow_dist_t);

        color = vec4<f32>(mix(color.rgb, vec3<f32>(color.a), glow_alpha), color.a);
    }

    return color;
}
