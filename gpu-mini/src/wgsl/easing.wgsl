// easing.wgsl — the shared easing curves.
//
// Curves live in the shader (not computed CPU-side per frame) because
// each widget's uniform only carries raw progress 0..1 — the shape of
// the motion is a rendering detail, and keeping it in WGSL means the
// WebGL fallback's GLSL easings stay the sole thing to match.

/// ease_smooth_step — the smoothstep polynomial t²(3−2t): zero
/// velocity at both ends. Identical to builtin `smoothstep(0,1,t)` for
/// t in 0..1 — spelled out so callers read intent ("this is an ease")
/// rather than an edge-shaping call.
fn ease_smooth_step(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

/// ease_in_out_cubic — the Penner cubic in/out: accelerate through the
/// first half, decelerate through the second. `select` picks the branch
/// without divergence — GPUs prefer it over `if` for scalar picks.
fn ease_in_out_cubic(t: f32) -> f32 {
    return select(
        1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0,
        4.0 * t * t * t,
        t < 0.5,
    );
}
