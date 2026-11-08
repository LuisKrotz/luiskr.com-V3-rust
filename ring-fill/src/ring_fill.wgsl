// ring_fill.wgsl — instanced border-fill sweep for the route-search
// field (rounded-rect) and circular buttons.
//
// Ports the CSS stroke-dashoffset sweep (the image-3 "border draws
// itself" affordance):
//
//   track — faint full hairline outline, always visible.
//   fill  — an arc that travels the perimeter from the top-left
//           clockwise as progress runs 0 → 1; the JS side eases it
//           over ~350 ms, matching the CSS transition it replaces.
//   head  — a brighter dot riding the arc tip.
//
// This file keeps its OWN instanced-quad vertex stage, so it composes
// with the gpu-mini helpers prelude (sd_round_box, quad_corner) —
// never the shared fullscreen triangle.
//
// Tile-sheet layout (8 f32s — lib.rs doc):
//   [0,1] center_px · [2] half_width · [3] half_height ·
//   [4] corner_radius · [5] fill_progress 0..1 ·
//   [6] shape_kind (0=circle, 1=rounded rect) · [7] reserved
//
// Uniform layout (f32s=16):
//   [0,1] resolution_dev · [2] time_seconds · [3] hairline_half ·
//   [4..8) track_rgba · [8..12) fill_rgba · [12..16) head_rgba

struct Uniforms {
    resolution_dev: vec2<f32>,
    time_seconds: f32,
    hairline_half: f32,
    track_rgba: vec4<f32>,
    fill_rgba: vec4<f32>,
    head_rgba: vec4<f32>,
};

struct TileRecord {
    center_px: vec2<f32>,
    half_width: f32,
    half_height: f32,
    corner_radius: f32,
    fill_progress: f32,
    shape_kind: f32,
    reserved: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> tiles: array<TileRecord>;

// ── Tunables ─────────────────────────────────────────────────────────

// Geometry constants (kept as written — the shader's arc tables were
// tuned to these exact truncations, not the f64 literals).
const HALF_PI: f32 = 1.5707963;
const PI: f32 = 3.1415927;
const THREE_HALF_PI: f32 = 4.712389;
const TAU: f32 = 6.2831853;

// Vertex stage — each instance's quad covers the tile bounds plus a
// hairline slack (the glow must paint OUTSIDE the border line).
const SLACK_HAIR_MULT: f32 = 4.0;
const SLACK_PX: f32 = 1.0;

// Shape discriminant — shape_kind is an f32 in the tile record;
// <0.5 is the circle branch.
const SHAPE_RECT_MIN: f32 = 0.5;

// The ring band's AA edge (±1px around the half-hairline).
const RING_AA_PX: f32 = 1.0;
// The head dot is 3× the hairline half-width.
const HEAD_RADIUS_MULT: f32 = 3.0;
const HEAD_AA_PX: f32 = 1.0;
// Pixels this faint are culled before the lighting pass runs.
const LIVE_EPSILON: f32 = 0.001;

// Light physics — the fill is a beam sweeping the border:
//   drawn    — already-swept perimeter stays lit (the stroke remains)
//   trail    — the hot comet tail decaying behind the tip
//   fringes  — chromatic refraction split at the trail's live edge
//   cast     — inverse halo the beam throws onto the surface
const DRAWN_SOFT: f32 = 0.003;      // drawn-edge AA band (perimeter t)
const TRAIL_DECAY: f32 = 14.0;      // comet-tail falloff rate
const FRINGE_SPLIT: f32 = 0.02;     // chromatic split width
const FRINGE_RED_DECAY: f32 = 8.0;  // red fringe reach
const FRINGE_BLUE_DECAY: f32 = 30.0;// blue fringe falloff
const FRINGE_RED_GAIN: f32 = 30.0;  // red fringe strength
const FRINGE_BLUE_GAIN: f32 = 6.0;  // blue fringe strength
const CAST_ARC_DECAY: f32 = 26.0;   // cast falloff along the perimeter
const CAST_BAND_DECAY: f32 = 0.55;  // cast falloff away from the border
const CAST_RGB: f32 = 0.35;         // cast strength into rgb
const CAST_ALPHA: f32 = 0.22;       // cast strength into alpha
const STROKE_DIM: f32 = 0.55;       // persistent drawn stroke vs tail

// Head — hot core + inverse-square halo + orbiting sparks.
const HEAD_CORE_FRAC: f32 = 0.5;   // core radius = head_r × this
const HEAD_HALO_SCALE: f32 = 2.5;  // inverse-square halo reach
const HEAD_HALO_FLOOR: f32 = 0.4;  // halo minimum (keeps it additive)
const HEAD_HALO_GAIN: f32 = 0.5;   // halo strength multiplier
const PROGRESS_EPSILON: f32 = 0.001; // head hides at sweep ends

// Sparks — deterministic orbiters flickering with time: 2π/3 phase
// spacing, per-spark detuned angular speeds and orbit radii.
const SPARK_COUNT: u32 = 3u;
const SPARK_PHASE_STEP: f32 = 2.0943951; // 2π/3
const SPARK_SPEED_BASE: f32 = 1.7;
const SPARK_SPEED_STEP: f32 = 0.9;
const SPARK_ORBIT_BASE: f32 = 1.6;
const SPARK_ORBIT_AMP: f32 = 0.5;
const SPARK_ORBIT_SPEED: f32 = 3.0;
const SPARK_ORBIT_PHASE: f32 = 7.0;
const SPARK_INNER_PX: f32 = 1.0;
const SPARK_OUTER_PX: f32 = 2.2;
const SPARK_TWINKLE_SPEED: f32 = 9.0;
const SPARK_TWINKLE_PHASE: f32 = 13.0;

// Output rgb clamps at 1.5 — headroom for the additive hot core to
// bloom through the premult blend without clipping to a flat disc.
const MAX_RGB: f32 = 1.5;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) pixel_px: vec2<f32>,
    @location(1) instance_t: f32,
};

/// `vs_main` — each instance covers the tile's bounding box plus
/// hairline slack; geometry is synthesized from the shared quad table,
/// no vertex buffers.
@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOut {
    var output: VertexOut;
    let corner_uv = quad_corner(vertex_index);
    let tile = tiles[instance_index];
    let bleed = uniforms.hairline_half * SLACK_HAIR_MULT + SLACK_PX;
    let origin_px = tile.center_px - vec2<f32>(tile.half_width, tile.half_height) - vec2<f32>(bleed);
    let size_px = vec2<f32>(tile.half_width * 2.0 + bleed * 2.0, tile.half_height * 2.0 + bleed * 2.0);

    let pixel_px = origin_px + corner_uv * size_px;
    let clip_xy = vec2<f32>(
        pixel_px.x / uniforms.resolution_dev.x * 2.0 - 1.0,
        1.0 - pixel_px.y / uniforms.resolution_dev.y * 2.0,
    );

    output.position = vec4<f32>(clip_xy, 0.0, 1.0);
    output.pixel_px = pixel_px;
    output.instance_t = f32(instance_index);

    return output;
}

/// `rrect_perimeter_t` — perimeter parameter t∈[0,1] of the closest
/// rounded-rect border point, walking the 8 segments clockwise from
/// the top edge's left end (straight→arc alternating).
fn rrect_perimeter_t(point: vec2<f32>, half_width: f32, half_height: f32, radius: f32) -> f32 {
    let straight_w = 2.0 * (half_width - radius);
    let straight_h = 2.0 * (half_height - radius);
    let arc_len = HALF_PI * radius;
    let perimeter = 2.0 * (straight_w + straight_h) + 4.0 * arc_len;

    var best_dist = 1e9;
    var best_t = 0.0;
    let corner_center = vec2<f32>(half_width - radius, half_height - radius);

    // top edge
    var proj = clamp(point.x, -corner_center.x, corner_center.x);
    var dist = abs(point.y + half_height) + abs(point.x - proj);
    var length_t = proj + corner_center.x;

    best_dist = dist;
    best_t = length_t;

    // TR arc (θ −π/2 → 0)
    var angle = atan2(point.y + corner_center.y, point.x - corner_center.x) + HALF_PI;

    if (angle < 0.0) { angle = angle + TAU; }

    angle = clamp(angle, 0.0, HALF_PI);
    dist = abs(length(point - vec2<f32>(corner_center.x, -corner_center.y)) - radius);
    length_t = straight_w + angle * radius;

    if (dist < best_dist) { best_dist = dist; best_t = length_t; }

    // right edge
    proj = clamp(point.y, -corner_center.y, corner_center.y);
    dist = abs(point.x - half_width) + abs(point.y - proj);
    length_t = straight_w + arc_len + (proj + corner_center.y);

    if (dist < best_dist) { best_dist = dist; best_t = length_t; }

    // BR arc (θ 0 → π/2)
    angle = atan2(point.y - corner_center.y, point.x - corner_center.x);

    if (angle < 0.0) { angle = angle + TAU; }

    angle = clamp(angle, 0.0, HALF_PI);
    dist = abs(length(point - corner_center) - radius);
    length_t = straight_w + arc_len + straight_h + angle * radius;

    if (dist < best_dist) { best_dist = dist; best_t = length_t; }

    // bottom edge (right → left)
    proj = clamp(point.x, -corner_center.x, corner_center.x);
    dist = abs(point.y - half_height) + abs(point.x - proj);
    length_t = straight_w + arc_len + straight_h + arc_len + (corner_center.x - proj);

    if (dist < best_dist) { best_dist = dist; best_t = length_t; }

    // BL arc (θ π/2 → π)
    angle = atan2(point.y - corner_center.y, point.x + corner_center.x);

    if (angle < 0.0) { angle = angle + TAU; }

    angle = clamp(angle, HALF_PI, PI);
    dist = abs(length(point + vec2<f32>(corner_center.x, -corner_center.y)) - radius);
    length_t = straight_w + arc_len + straight_h + arc_len + straight_w + (angle - HALF_PI) * radius;

    if (dist < best_dist) { best_dist = dist; best_t = length_t; }

    // left edge (bottom → top)
    proj = clamp(point.y, -corner_center.y, corner_center.y);
    dist = abs(point.x + half_width) + abs(point.y - proj);
    length_t = 2.0 * (straight_w + arc_len) + straight_h + (corner_center.y - proj);

    if (dist < best_dist) { best_dist = dist; best_t = length_t; }

    // TL arc (θ π → 3π/2)
    angle = atan2(point.y + corner_center.y, point.x + corner_center.x);

    if (angle < 0.0) { angle = angle + TAU; }

    angle = clamp(angle, PI, THREE_HALF_PI);
    dist = abs(length(point + corner_center) - radius);
    length_t = 2.0 * (straight_w + arc_len + straight_h) + (angle - PI) * radius;

    if (dist < best_dist) { best_dist = dist; best_t = length_t; }

    return clamp(best_t / perimeter, 0.0, 1.0);
}

/// `rrect_perimeter_point` — point on the rounded-rect perimeter at
/// arc-length `arc_length` — mirrors rrect_perimeter_t's segment walk.
fn rrect_perimeter_point(arc_length: f32, half_width: f32, half_height: f32, radius: f32) -> vec2<f32> {
    let straight_w = 2.0 * (half_width - radius);
    let straight_h = 2.0 * (half_height - radius);
    let arc_len = HALF_PI * radius;
    let corner_center = vec2<f32>(half_width - radius, half_height - radius);
    var remaining = arc_length;

    if (remaining < straight_w) { return vec2<f32>(-corner_center.x + remaining, -half_height); }
    remaining = remaining - straight_w;

    if (remaining < arc_len) {
        let angle = remaining / radius - HALF_PI;

        return vec2<f32>(corner_center.x + radius * cos(angle), -corner_center.y + radius * sin(angle));
    }
    remaining = remaining - arc_len;

    if (remaining < straight_h) { return vec2<f32>(half_width, -corner_center.y + remaining); }
    remaining = remaining - straight_h;

    if (remaining < arc_len) {
        let angle = remaining / radius;

        return vec2<f32>(corner_center.x + radius * cos(angle), corner_center.y + radius * sin(angle));
    }
    remaining = remaining - arc_len;

    if (remaining < straight_w) { return vec2<f32>(corner_center.x - remaining, half_height); }
    remaining = remaining - straight_w;

    if (remaining < arc_len) {
        let angle = remaining / radius + HALF_PI;

        return vec2<f32>(-corner_center.x + radius * cos(angle), corner_center.y + radius * sin(angle));
    }
    remaining = remaining - arc_len;

    if (remaining < straight_h) { return vec2<f32>(-half_width, corner_center.y - remaining); }
    remaining = remaining - straight_h;

    let angle = remaining / radius + PI;

    return vec2<f32>(-corner_center.x + radius * cos(angle), -corner_center.y + radius * sin(angle));
}

/// `fs_main` — track ring + progress arc + head dot, per shape.
@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    let tile = tiles[u32(input.instance_t + 0.5)];
    let to_pixel = input.pixel_px - tile.center_px;

    var border_dist = 0.0;
    var perimeter_t = 0.0;
    var tip_px = tile.center_px;

    if (tile.shape_kind < SHAPE_RECT_MIN) {
        // circle — radius = half_width; angle sweep from 12 o'clock.
        let center_dist = length(to_pixel);

        border_dist = abs(center_dist - tile.half_width);

        var angle_t = atan2(to_pixel.x, -to_pixel.y) / TAU;

        if (angle_t < 0.0) { angle_t = angle_t + 1.0; }

        perimeter_t = angle_t;

        let tip_angle = tile.fill_progress * TAU;

        tip_px = tile.center_px + tile.half_width * vec2<f32>(sin(tip_angle), -cos(tip_angle));
    } else {
        // rounded rect — perimeter sweep from the top edge's left end.
        border_dist = abs(sd_round_box(to_pixel, vec2<f32>(tile.half_width, tile.half_height), tile.corner_radius));
        perimeter_t = rrect_perimeter_t(to_pixel, tile.half_width, tile.half_height, tile.corner_radius);

        let straight_w = 2.0 * (tile.half_width - tile.corner_radius);
        let straight_h = 2.0 * (tile.half_height - tile.corner_radius);
        let arc_len = HALF_PI * tile.corner_radius;
        let perimeter = 2.0 * (straight_w + straight_h) + 4.0 * arc_len;

        tip_px = tile.center_px + rrect_perimeter_point(
            tile.fill_progress * perimeter,
            tile.half_width,
            tile.half_height,
            tile.corner_radius,
        );
    }

    let ring_mask = 1.0 - smoothstep(
        uniforms.hairline_half - RING_AA_PX,
        uniforms.hairline_half + RING_AA_PX,
        border_dist,
    );

    let head_radius = uniforms.hairline_half * HEAD_RADIUS_MULT;
    let head_mask = 1.0 - smoothstep(head_radius - HEAD_AA_PX, head_radius + HEAD_AA_PX, length(input.pixel_px - tip_px));

    if (ring_mask <= LIVE_EPSILON && head_mask <= LIVE_EPSILON) {
        discard;
    }

    // ── Light physics ────────────────────────────────────────────────
    // The fill is a light beam sweeping the border: chromatic refraction
    // fringes at the arc edge, an inverse-square halo cast onto the
    // surface (reflection), and spark particles riding the head.

    // Wrap-around distance along the perimeter to the beam tail.
    let wrapped_dist = abs(perimeter_t - tile.fill_progress);
    let arc_dist = min(wrapped_dist, 1.0 - wrapped_dist);

    // The light tip draws the border: already-drawn stays lit (stroke
    // remains), and a bright trail decays behind the head like a comet
    // tail — reads as light traveling, not a filling gauge.
    let drawn_mask = 1.0 - smoothstep(
        tile.fill_progress - DRAWN_SOFT,
        tile.fill_progress + DRAWN_SOFT,
        perimeter_t,
    );

    let trail_glow = drawn_mask * exp(-arc_dist * TRAIL_DECAY);

    var color = uniforms.track_rgba * ring_mask;

    // Persistent drawn stroke — dimmer than the track so the tail leads.
    color = color + uniforms.fill_rgba * ring_mask * drawn_mask * STROKE_DIM;

    // Comet tail — the hot part just behind the tip.
    color = color + uniforms.fill_rgba * ring_mask * trail_glow;

    // Refraction fringes — chromatic split at the trail's live edge.
    let fringe_red_edge = trail_glow * exp(-arc_dist * FRINGE_RED_DECAY);
    let fringe_blue_edge = drawn_mask * (1.0 - exp(-arc_dist * FRINGE_BLUE_DECAY));

    // Whole-vec4 writes — multi-component swizzle assignment
    // (`color.rgb = …`) is rejected by strict naga validators.
    color = vec4<f32>(
        color.rgb + vec3<f32>(
            uniforms.fill_rgba.r * ring_mask * fringe_red_edge * FRINGE_SPLIT * FRINGE_RED_GAIN,
            0.0,
            uniforms.fill_rgba.b * ring_mask * fringe_blue_edge * FRINGE_SPLIT * FRINGE_BLUE_GAIN,
        ),
        color.a,
    );

    // Reflection — light cast off the ring onto the surface, falling
    // off with distance-to-beam (perimeter arc) × distance-to-border.
    // (`cast` is a reserved WGSL keyword — name it cast_light.)
    let cast_light = exp(-arc_dist * CAST_ARC_DECAY) * exp(-border_dist * CAST_BAND_DECAY) * uniforms.fill_rgba.a;

    color = vec4<f32>(
        color.rgb + uniforms.fill_rgba.rgb * cast_light * CAST_RGB,
        max(color.a, cast_light * CAST_ALPHA),
    );

    // Head — hot core + inverse-square halo + sparks.
    let head_dist = length(input.pixel_px - tip_px);
    let core_mask = 1.0 - smoothstep(head_radius * HEAD_CORE_FRAC, head_radius, head_dist);
    let halo_strength = uniforms.head_rgba.a * head_radius * HEAD_HALO_SCALE
        / max(head_dist, head_radius * HEAD_CORE_FRAC) - uniforms.head_rgba.a * HEAD_HALO_FLOOR;

    if (tile.fill_progress > PROGRESS_EPSILON && tile.fill_progress < 1.0 - PROGRESS_EPSILON) {
        color = mix(color, uniforms.head_rgba, core_mask);
        color = vec4<f32>(
            color.rgb + uniforms.head_rgba.rgb * clamp(halo_strength, 0.0, 1.0) * HEAD_HALO_GAIN,
            max(color.a, clamp(halo_strength, 0.0, 1.0) * HEAD_HALO_GAIN),
        );

        // Sparks — deterministic orbiters flickering with time.
        for (var spark_i = 0u; spark_i < SPARK_COUNT; spark_i = spark_i + 1u) {
            let spark_phase = f32(spark_i) * SPARK_PHASE_STEP;
            let spark_angle = spark_phase + uniforms.time_seconds * (SPARK_SPEED_BASE + f32(spark_i) * SPARK_SPEED_STEP);
            let spark_orbit = head_radius * (SPARK_ORBIT_BASE + SPARK_ORBIT_AMP * sin(uniforms.time_seconds * SPARK_ORBIT_SPEED + spark_phase * SPARK_ORBIT_PHASE));
            let spark_px = tip_px + spark_orbit * vec2<f32>(cos(spark_angle), sin(spark_angle));
            let spark_mask = 1.0 - smoothstep(SPARK_INNER_PX, SPARK_OUTER_PX, length(input.pixel_px - spark_px));
            let spark_twinkle = 0.5 + 0.5 * sin(uniforms.time_seconds * SPARK_TWINKLE_SPEED + spark_phase * SPARK_TWINKLE_PHASE);

            color = mix(color, uniforms.head_rgba, spark_mask * spark_twinkle);
        }
    }

    let alpha = max(color.a, ring_mask * uniforms.track_rgba.a * (1.0 - drawn_mask));

    return vec4<f32>(clamp(color.rgb, vec3<f32>(0.0), vec3<f32>(MAX_RGB)), clamp(alpha, 0.0, 1.0));
}
