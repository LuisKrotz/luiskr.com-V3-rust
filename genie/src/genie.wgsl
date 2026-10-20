// genie.wgsl — expand-modal genie warp: textured mesh deform + the
// dock-flash accent (one pipeline, one shared uniform).
//
// Field math mirrors `core/tokens/motion/genie.ts` + the 2D backend
// (`genie-canvas.ts`) exactly — the per-row staggered eased lerp the
// ui-layouts mac-genie reference implements. The field always warps the
// MODAL rect (S=`modal`) toward the FIGURE rect (D=`fig`); the open
// transition plays the identical funnel in reverse via the driver-
// flipped progress `p`. Rows index in down-space (ty 0 = far edge,
// 1 = figure-facing edge) and each row owns two staggered clocks:
//   xs = (1−ty)·STAGGER_X   xe = easeInOutCubic(clamp((p−xs)/(1−xs)))
//   ys = (1−ty)·STAGGER_Y   ye = easeInQuad(    clamp((p−ys)/(1−ys)))
//     x' = lerp(S.x + u·S.w, D.x + u·D.w, xe)
//     y' = farY + sgn·lerp(ty·H, iYfar + ty·iH, ye)
// iYfar is the far-edge → figure-far-edge distance, iH the figure
// thickness — every row lands on its proportional row inside the card
// at p=1 (the canonical 0-height mouth only suits a dock icon, so the
// warped sheet converges exactly onto the figure rect the real card
// then paints). The stagger is the whole effect: leading rows move
// first while far rows wait out their delay, so the funnel's S-curved
// walls emerge from the easing alone. Positions arrive in CSS px and
// are scaled to device px by `dpr` before the NDC projection.
//
// Uniform layout (24 f32s — mirrors GPU_WIDGET_UNIFORM_F32S[GENIE]):
//   [0,1]   res        canvas DEVICE px
//   [2]     time       seconds
//   [3]     prog       funnel progress 0..1 (linear — the row eases carry it)
//   [4..8]  modal      modal rect (x,y,w,h) CSS px
//   [8..12] fig        figure rect
//   [12,13] reserved   (kept for the 24-f32 ABI)
//   [14]    dpr        device/css ratio
//   [15]    flags      bit0 opening · bit1 nearBottom · bit2 hasTex
//   [16..20] tint      accent rgba (untex'd ghost fill)
//   [20..24] uv        texture uv scale.xy + offset.xy (cover map)
//
// Bindings:  0 uniform · 1 texture · 2 sampler

const ROWS: u32 = 192u;     // mesh rows — ≈ the 2D backend's adaptive count on a 900px sheet
const QUADS: u32 = 191u;    // ROWS-1 triangle-list quads (6 verts each)
const STAGGER_X: f32 = 0.65;  // horizontal row stagger — GENIE_WARP.STAGGER_X
const STAGGER_Y: f32 = 0.2;   // vertical row stagger — GENIE_WARP.STAGGER_Y
const GLOW_START: f32 = 0.75; // dock-flash window — GENIE_WARP.GLOW_START
const GLOW_ALPHA: f32 = 0.3;  // dock-flash peak — GENIE_WARP.GLOW_ALPHA
const GLOW_RADIUS_FRAC: f32 = 0.5; // flash radius = min(D.w,D.h)·frac

struct Uni {
  res: vec2f,
  time: f32,
  prog: f32,
  modal: vec4f,
  fig: vec4f,
  cfg: vec4f,
  tint: vec4f,
  uv: vec4f,
}

@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct MeshOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) shade: f32,
  // Fragment-space CSS-px position — the dock flash shades by distance
  // to the figure's center, not by texture coords.
  @location(2) px: vec2f,
}

// easeInOutCubic — the row clock for the horizontal pinch (zero slope
// at both ends so the funnel walls ease in and land softly).
fn eio_cubic(t: f32) -> f32 {
  return select(
    1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0,
    4.0 * t * t * t,
    t < 0.5
  );
}

// The genie field — returns (x, y) CSS px + the row's down-space ty.
// `u` is the column 0..1 left→right, `v` the row 0..1 in sheet order
// (top→bottom). See the module header for the staggered-clock math.
fn field(u: f32, v: f32) -> vec3f {
  let flags = u32(U.cfg.w);
  let near_bottom = (flags & 2u) != 0u;

  let p = U.prog;

  // Down-space ty — 0 at the sheet's far edge, 1 at the figure-facing
  // edge (the bottom row v=1 when the figure sits below the modal).
  let ty = select(1.0 - v, v, near_bottom);

  // Staggered horizontal clock — leading rows start at p=0 while the
  // far edge waits out (1−ty)·STAGGER_X of the run.
  let xs = (1.0 - ty) * STAGGER_X;
  let xp = clamp((p - xs) / (1.0 - xs), 0.0, 1.0);
  let xe = eio_cubic(xp);

  // Staggered vertical clock — quadratic ease-in, the dock suck.
  let ys = (1.0 - ty) * STAGGER_Y;
  let yp = clamp((p - ys) / (1.0 - ys), 0.0, 1.0);
  let ye = yp * yp;

  // modal.xy = x,y — modal.z is WIDTH, modal.w is HEIGHT.
  let W = U.modal.z;
  let H = U.modal.w;

  // Landing rect in down-space — iYfar is the far-edge → figure
  // far-edge distance, iH the figure's thickness.
  let iYfar = select(
    U.modal.y + U.modal.w - (U.fig.y + U.fig.w),
    U.fig.y - U.modal.y,
    near_bottom
  );
  let iH = U.fig.w;

  // Row ty lands on iYfar + ty·iH at ye=1 — the warped sheet converges
  // exactly onto the card it hands off to.
  let tyH = ty * H;
  let y_down = tyH + (iYfar + ty * iH - tyH) * ye;
  let y_css = select(U.modal.y + U.modal.w - y_down, U.modal.y + y_down, near_bottom);

  let x_css = U.modal.x + u * W + (U.fig.x + u * U.fig.z - U.modal.x - u * W) * xe;

  return vec3f(x_css, y_css, ty);
}

@vertex
fn vs_mesh(@builtin(vertex_index) vi: u32) -> MeshOut {
  // 6 verts per quad: 0=(0,0) 1=(1,0) 2=(0,1) 3=(0,1) 4=(1,0) 5=(1,1)
  let quad = vi / 6u;
  let corner = vi % 6u;
  var u = 0.0;
  var dy = 0.0;

  if corner == 1u || corner == 4u || corner == 5u {
    u = 1.0;
  }
  if corner == 2u || corner == 3u || corner == 5u {
    dy = 1.0;
  }

  let v = (f32(quad) + dy) / f32(QUADS);
  let f = field(u, v);
  let dx = f.x * U.cfg.z;
  let dy = f.y * U.cfg.z;

  var out: MeshOut;
  out.pos = vec4f(dx / U.res.x * 2.0 - 1.0, 1.0 - dy / U.res.y * 2.0, 0.0, 1.0);
  out.uv = vec2f(u, v) * U.uv.xy + U.uv.zw;
  out.shade = f.z;
  out.px = f.xy;

  return out;
}

@fragment
fn fs_mesh(in: MeshOut) -> @location(0) vec4f {
  let has_tex = (u32(U.cfg.w) & 4u) != 0u;

  var c: vec4f;

  if has_tex {
    let t = textureSample(tex, samp, in.uv);
    // Premultiply for the One/OneMinusSrcAlpha blend.
    c = vec4f(t.rgb * t.a, t.a);
  } else {
    // Untex'd ghost — the accent-tinted silhouette while the bitmap lags.
    let a = 0.22;
    c = vec4f(U.tint.rgb * a, a);
  }

  // Dock flash — the white radial bloom at the figure's center for the
  // run's last quarter (the reference's suck punctuation); ease-out.
  let gp = (U.prog - GLOW_START) / (1.0 - GLOW_START);

  if gp > 0.0 && gp < 1.0 {
    let gc = vec2f(U.fig.x + U.fig.z * 0.5, U.fig.y + U.fig.w * 0.5);
    let gr = min(U.fig.z, U.fig.w) * GLOW_RADIUS_FRAC;
    let rr = clamp(length(in.px - gc) / max(gr, 0.1), 0.0, 1.0);
    let a = (1.0 - (1.0 - gp) * (1.0 - gp)) * GLOW_ALPHA * (1.0 - rr);

    c = vec4f(mix(c.rgb, vec3f(c.a), a), c.a);
  }

  return c;
}
