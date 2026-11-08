// hash.wgsl — the shared deterministic hash helpers.
//
// Sine-hash family: `fract(sin(dot(x, K)) * M)` — cheap, stateless,
// and identical on every GPU, which is what keeps procedural noise
// pixel-stable frame to frame. The constants are the classic Book of
// Shaders pair (127.1 / 311.7 lattice vector, 43758.5453 mantissa
// scrambler) — tuned once in 2015 and never improved on for 2D work.
//
// Widgets needing a DIFFERENT pattern instance keep a private hash
// (changing these constants re-rolls the noise field — same quality,
// different arrangement — which is a visual change, not a bug fix).

/// hash_sin_2d — scalar hash of a 2D lattice point, 0..1. Used by
/// value_noise for corner values and by widgets hashing pixel cells.
fn hash_sin_2d(point: vec2<f32>) -> f32 {
    return fract(sin(dot(point, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

/// hash_sin_1d — scalar hash of a scalar seed (glyph index, cell id).
fn hash_sin_1d(seed: f32) -> f32 {
    return fract(sin(seed * 127.1 + 311.7) * 43758.5453);
}

/// value_noise — smooth 2D value noise: hash the four lattice corners
/// of the cell containing `point`, then bilinear-mix them with a
/// smoothstep weight (`f*f*(3-2f)` — the ease curve, so the blend has
/// zero derivative at corners and tiles without visible seams).
fn value_noise(point: vec2<f32>) -> f32 {
    let cell = floor(point);
    var local = fract(point);

    local = local * local * (3.0 - 2.0 * local);

    return mix(
        mix(hash_sin_2d(cell), hash_sin_2d(cell + vec2<f32>(1.0, 0.0)), local.x),
        mix(hash_sin_2d(cell + vec2<f32>(0.0, 1.0)), hash_sin_2d(cell + vec2<f32>(1.0, 1.0)), local.x),
        local.y,
    );
}
