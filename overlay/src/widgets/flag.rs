//! Flag widget — the locale flag in nav/dialogs.
//!
//! WGSL port of `core/utils/canvas/widgets/flag/shaders.ts`: aerodynamic
//! wind warp (fixed at the pole, free at the fly edge), per-flag flutter
//! + specular personality (12 variants), split-flag transition between
//! two textures, rounded-corner mask.
//!
//! Texture model: `upload_flag_textures(id, rgba_a, w_a, h_a, rgba_b, w_b, h_b)`
//! stages both halves; single-flag widgets reuse tex1 for both bindings.
//!
//! Uniform layout (f32s):
//!   [0..1] res px      [2] time s     [3] hover 0..1
//!   [4] anim_type 0..11  [5] is_split  [6] split_x   [7] pad
//!   [8..11] widget rect (x, y, w, h) CSS px

/// f32 slots in the widget uniform block.
pub const UNIFORM_F32S: usize = 12;

/// Flag animation personalities — mirrors `u_anim_type` ranges in the
/// GLSL original and `FLAG_ANIMS` in core/tokens (kept in lockstep).
/// These are a JS↔Rust CONTRACT table, not code the crate executes —
/// the shader's `u_anim_type` branches index by these values.
#[allow(dead_code)]
pub mod anims {
    /// EN (US — stars shimmer).
    pub const EN: f32 = 0.0;
    /// PT (BR — rhombus solar glow).
    pub const PT: f32 = 1.0;
    /// ES (Spain — flamenco silk).
    pub const ES: f32 = 2.0;
    /// DE (CH+DE split — cross-ribbon).
    pub const DE: f32 = 3.0;
    /// HRK (DE+BR — harmonic wave).
    pub const HRK: f32 = 4.0;
    /// CAS (AR+UY — Sol de Mayo rays).
    pub const CAS: f32 = 5.0;
    /// RIV (UY+BR — river reflection).
    pub const RIV: f32 = 6.0;
    /// GN (Paraguay — horizontal surge).
    pub const GN: f32 = 7.0;
    /// IT (Italy — mediterranean silk).
    pub const IT: f32 = 8.0;
    /// RU (Russia — aurora shimmer).
    pub const RU: f32 = 9.0;
    /// FR (France — satin ripple).
    pub const FR: f32 = 10.0;
    /// TLN (IT+BR — venetian pulse).
    pub const TLN: f32 = 11.0;
}

/// WGSL — ported from FLAG_FS.
pub const WGSL: &str = include_str!("flag.wgsl");
