// sdf.wgsl — the shared signed-distance-field primitives every widget
// used to paste into its own source (under three different names —
// `sdCapsule`, `distCapsule`, `distSegment` were the same function).
//
// A signed distance field answers "how far is this pixel from the
// nearest edge": negative = inside the shape, 0 = on the edge,
// positive = outside. Widgets draw by evaluating the field per pixel
// and converting the distance to alpha — no geometry ever exists.
//
// Param names follow the literate convention (`point`, `segment_start`)
// rather than the SDF papers' single letters — see AGENTS.md naming.

/// dist_to_segment — euclidean distance from `point` to the closest
/// point on segment `segment_start`→`segment_end` (always ≥ 0 — this
/// is a distance, not a signed field: segments have no interior).
///
/// The math, step by step:
///   to_point = point - segment_start → pixel relative to segment start
///   segment  = segment_end - segment_start → the segment as a vector
///   dot(to_point,segment)/dot(segment,segment) → parametric position
///     of the closest point on the INFINITE line (0 at start, 1 at end)
///   clamp(t,0,1) → restrict to the SEGMENT — the closest point ON it
///   length(to_point - segment*t) → euclidean distance to that point
fn dist_to_segment(point: vec2<f32>, segment_start: vec2<f32>, segment_end: vec2<f32>) -> f32 {
    let to_point = point - segment_start;
    let segment = segment_end - segment_start;
    let closest_t = clamp(dot(to_point, segment) / dot(segment, segment), 0.0, 1.0);

    return length(to_point - segment * closest_t);
}

/// sd_capsule — signed distance to a capsule: the segment thickened by
/// `radius` (a "stadium" — rectangle with semicircular caps). Negative
/// inside, 0 on the edge. The fundamental bar/stroke primitive: UI
/// hairlines are capsules, not rectangles, so their ends read round.
fn sd_capsule(point: vec2<f32>, segment_start: vec2<f32>, segment_end: vec2<f32>, radius: f32) -> f32 {
    return dist_to_segment(point, segment_start, segment_end) - radius;
}

/// sd_round_box — signed distance to an origin-centered box with
/// `corner_radius` rounded corners and `half_extents` half-size.
/// The standard Inigo Quilez formula: mirror the point into the first
/// quadrant (`abs`), measure the excess beyond the corner-reduced box
/// (`abs(point) - half_extents + radius`), then combine the outside
/// euclidean part with the inside Chebyshev part and subtract the
/// radius back out — one expression handles inside AND outside.
fn sd_round_box(point: vec2<f32>, half_extents: vec2<f32>, corner_radius: f32) -> f32 {
    let excess = abs(point) - half_extents + vec2<f32>(corner_radius);

    return length(max(excess, vec2<f32>(0.0))) + min(max(excess.x, excess.y), 0.0) - corner_radius;
}

/// smooth_min — polynomial smooth minimum: blends two signed distance
/// fields across a `blend_width`-wide band instead of meeting at a hard
/// union seam. This is what makes two approaching shapes pool together
/// like liquid droplets rather than touching at a crease — the metaball
/// trick every "liquid merge" widget uses.
fn smooth_min(dist_a: f32, dist_b: f32, blend_width: f32) -> f32 {
    let blend_t = clamp(0.5 + 0.5 * (dist_b - dist_a) / blend_width, 0.0, 1.0);

    return mix(dist_b, dist_a, blend_t) - blend_width * blend_t * (1.0 - blend_t);
}
