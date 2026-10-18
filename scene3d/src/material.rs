//! Scene3d materials — the four material kinds the scenes use,
//! reduced to a compact registry the renderer turns into bind groups.
//!
//! # Why an enum, not classes
//!
//! three.js materials are classes with dozens of fields — most
//! irrelevant to these scenes (env maps, normal maps, aoMap, …). The
//! actual surface used: MeshBasicMaterial, MeshStandardMaterial,
//! SpriteMaterial, PointsMaterial — differing ONLY in: lighting on/off,
//! texture or none, point size vs quad. An enum keeps the memory flat
//! (one discriminant + a fixed-size payload) and the match in the
//! pipeline switch exhaustive — adding a fifth kind is a compile-time
//! audit, not a runtime fallback.
//!
//! # Uniform layout
//!
//! Each material packs to a fixed 32-f32 (128-byte) uniform — the same
//! block size for every kind so ONE bind-group layout serves all
//! pipelines (uniform buffers cap at 64KB; 128B leaves the material
//! table at a comfortable per-draw dynamic-offset size):
//!   [0..4]  color.rgba        — base tint (diffuse/albedo)
//!   [4..8]  emissive.rgba     — self-lit term (atmosphere glows)
//!   [8..12] params            — kind-specific: (size, roughness,
//!                               metalness, opacity-threshold)
//!   [12..16] flags            — (texture_id, side, blending, fog-on)
//!   [16..32] reserved         — padding to 128B + per-kind extras
//!                               (sprite rotation, point attenuation)

/// `Side` — face culling mode, three.js FrontSide/BackSide/DoubleSide.
/// `Double` disables the cull-mode state on the pipeline rather than
/// flipping winding — GPU rasterizers treat both cull=none and
/// cull=back+draw-twice; one-sided drawing is always cheaper.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Side {
    /// Draw only front (CCW) faces — default, fastest.
    #[default]
    Front,
    /// Draw only back faces (inside-out shells — atmosphere interior).
    Back,
    /// Draw both — thin geometry (rings, planes) visible from either.
    Double,
}

/// `Blending` — the alpha compositing mode. The scenes use only two:
/// Normal (source-over) and Additive (glow accumulations — atmosphere,
/// embers, bloom layers). Additive needs NO depth-sort (a+b == b+a) —
/// the renderer's draw-order sort skips those draw calls entirely.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Blending {
    /// Source-over alpha blend: C = src·α + dst·(1−α).
    #[default]
    Normal,
    /// Additive glow: C = src + dst — order-independent, no sorting.
    Additive,
}

/// `Material` — one material record in the registry. Kind-specific
/// params live in the variant; shared presentation flags (side,
/// blending, transparency, depth-write) sit alongside so the pipeline
/// state key is computable without a second match.
#[derive(Clone, Copy, Debug)]
pub enum Material {
    /// Unlit flat color/texture — the workhorse (UI sprites in space,
    /// star billboards, ring bands). `texture` is a registry id,
    /// `u32::MAX` for none (the shader multiplies color·texture when
    /// bound — keeping the multiply unconditional avoids a pipeline
    /// per texture-presence).
    Basic {
        /// Base RGBA tint.
        color: [f32; 4],
        /// Texture registry id or u32::MAX.
        texture: u32,
    },
    /// Lambertian-lit surface — needs normals + one PointLight the
    /// scene passes via the camera uniform. `roughness`/`metalness`
    /// feed the simplified BRDF in the WGSL (a Blinn-Phong specular
    /// scaled by (1−roughness)² — cheap, and visually identical for
    /// the matte bodies these scenes render).
    Standard {
        /// Base RGBA tint.
        color: [f32; 4],
        /// Roughness 0..1 — specular spread inverse.
        roughness: f32,
        /// Metalness 0..1 — specular tint toward base color.
        metalness: f32,
        /// Texture registry id or u32::MAX.
        texture: u32,
    },
    /// Camera-facing billboard — the renderer substitutes the shared
    /// billboard quad for geometry; per-sprite offset/rotation folds
    /// into the instance data.
    Sprite {
        /// Base RGBA tint.
        color: [f32; 4],
        /// Texture registry id or u32::MAX.
        texture: u32,
    },
    /// Point sprite — each vertex rasterizes as a `size`-px point.
    /// `size` is in NDC scale (points size is pixel-denominated in
    /// WebGL; the shader divides by canvas height to stay DPR-stable).
    Points {
        /// Base RGBA tint.
        color: [f32; 4],
        /// Point size in pixels at unit distance.
        size: f32,
        /// Texture registry id or u32::MAX.
        texture: u32,
    },
}

/// `Flags` — the presentation bits the pipeline key reads. Packed as
/// a byte-field struct (not bitflags — 4 u8s pack into one f32 slot of
/// the uniform block with zero endianness drama).
#[derive(Clone, Copy, Debug, Default)]
pub struct Flags {
    /// Face culling — see `Side`.
    pub side: Side,
    /// Alpha compositing — see `Blending`.
    pub blending: Blending,
    /// `transparent` — participates in the alpha-sorted pass (opaque
    /// draws skip the sort entirely).
    pub transparent: bool,
    /// `depth_write` — off for additive glows (they must not occlude
    /// the particles behind them).
    pub depth_write: bool,
}

/// `Record` — one registry slot: the material + its flags + a dirty
/// stamp so `pack` only rewrites what changed between frames.
#[derive(Clone, Copy, Debug)]
pub struct Record {
    /// The kind-specific payload.
    pub material: Material,
    /// Pipeline-state flags.
    pub flags: Flags,
    /// Emissive tint — shared across kinds (basic/star materials use
    /// it for the self-lit halo term).
    pub emissive: [f32; 4],
}

impl Record {
    /// `Record::new` — defaults: front-side, normal blend, opaque,
    /// depth-write on, zero emissive.
    ///
    /// @param material The kind-specific payload.
    /// @returns A presentable default record.
    pub fn new(material: Material) -> Record {
        Record {
            material,
            flags: Flags {
                side: Side::Front,
                blending: Blending::Normal,
                transparent: false,
                depth_write: true,
            },
            emissive: [0.0, 0.0, 0.0, 0.0],
        }
    }

    /// `pack` — write the record into the fixed 32-f32 uniform block
    /// described in the module doc. Color/emissive carry across every
    /// kind; `params`/`flags` switch on the discriminant.
    ///
    /// @param out The 32-f32 destination slice (must be ≥32).
    pub fn pack(&self, out: &mut [f32]) {
        out.fill(0.0);

        let (color, params, texture) = match self.material {
            Material::Basic { color, texture } => (color, [0.0, 0.0, 0.0, 0.0], texture),
            Material::Standard {
                color,
                roughness,
                metalness,
                texture,
            } => (color, [0.0, roughness, metalness, 0.0], texture),
            Material::Sprite { color, texture } => (color, [0.0, 0.0, 0.0, 0.0], texture),
            Material::Points {
                color,
                size,
                texture,
            } => (color, [size, 0.0, 0.0, 0.0], texture),
        };

        out[0..4].copy_from_slice(&color);
        out[4..8].copy_from_slice(&self.emissive);
        out[8..12].copy_from_slice(&params);

        // flags row — texture id as raw f32 bits (registry ids < 2²⁴
        // so integer round-trip through f32 is exact).
        out[12] = texture as f32;
        out[13] = self.flags.side as u32 as f32;
        out[14] = self.flags.blending as u32 as f32;
        out[15] = if self.flags.transparent { 1.0 } else { 0.0 };
    }
}
