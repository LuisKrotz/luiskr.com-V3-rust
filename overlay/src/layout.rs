//! Scene layout + hit-testing — the CPU-side data model every frame
//! assembles before the GPU draw.
//!
//! # Why this exists (vs. a traditional DOM-driven frame)
//!
//! A traditional app lets the browser own layout: elements have rects,
//! the compositor knows what's under the cursor, hit-testing is free.
//! Our overlay is ONE canvas — the browser sees a single opaque quad.
//! Everything the DOM did for free (what's under the pointer, what order
//! things paint in) must be reimplemented here. This module is that
//! reimplementation, kept deliberately flat:
//!
//! - `Vec<Quad>` is the paint list — one struct of `#[repr(C)]` f32
//!   fields per drawable, uploaded to the GPU as ONE `queue.write_buffer`
//!   (a single memcpy into mapped VRAM). No per-element marshaling, no
//!   draw call per element — instancing does it all.
//! - `Vec<Hit>` is a parallel slim list of ONLY the interactive rects.
//!   Duplicating data beats deriving it: a hit-test scan reads 24 bytes
//!   per entry instead of the Quad's 56 — more targets per cacheline,
//!   and the pointer path never touches the paint array.
//!
//! Both vectors are `clear()`ed and refilled each frame — capacity is
//! retained, so steady-state frames perform ZERO heap allocations
//! (frame-budget rule: no allocs inside rAF).
//!
//! # Coordinate space
//!
//! Everything here lives in **canvas CSS px** — `clientX/clientY` units,
//! origin at the canvas top-left, +x right, +y down. Device-pixel
//! conversion happens at the very end (scissor rect computation in
//! `lib.rs`) so this layer never thinks about `devicePixelRatio`. If a
//! number here looks too big/small, measure it in DevTools CSS px first.

use bytemuck::{Pod, Zeroable};

/// One textured quad instance — the atom of the instanced draw.
///
/// `#[repr(C)]` + `Pod` make the byte layout fixed and memcpy-able
/// straight into the GPU vertex/instance buffer. Field order matters:
/// floats are 4B so any order aligns, but keep the 4-f32 groups together
/// (`tint`) so a cacheline holds coherent fields.
///
/// ## Size math
///
/// 14 × f32 + 1 × u32 = 60 bytes… but `#[repr(C)]` pads to alignment:
/// actual size is 60B (all fields 4B-aligned, no padding needed).
/// SAFE RANGE: fields may be added — keep total ≤ 128B or instance
/// upload bandwidth per frame (~10² quads × size) becomes measurable.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct Quad {
    /// Top-left corner in CSS px (origin = canvas top-left, +y DOWN —
    /// the vertex shader flips to NDC's +y-up, see media.wgsl).
    pub pos: [f32; 2],
    /// Size in CSS px. Zero-size quads are legal — the fragment shader
    /// degenerates to nothing (waste of one instance; skip pushing them
    /// instead when you know size is 0).
    pub size: [f32; 2],
    /// Normalized UV top-left into the texture atlas, 0..1.
    /// [0,0] + uv_size [0,0] = "no texture" — the media shader treats a
    /// zero-area UV as untextured (flat tint).
    pub uv: [f32; 2],
    /// UV extent, 0..1. For full-image sampling use [1,1] with uv [0,0].
    pub uv_size: [f32; 2],
    /// Straight-alpha RGBA multiplier over the texture sample —
    /// [1,1,1,1] is identity. NOT premultiplied; the pipeline's blend
    /// state (alpha-over) handles the compositing math. Values outside
    /// 0..1 are legal for HDR-ish glow effects but clamp the alpha —
    /// >1 alpha wraps through the blend equation on some drivers.
    pub tint: [f32; 4],
    /// Sort key for painter's-order z: higher draws on top. Kept as f32
    /// (not integer z-index) so procedural layering can slot between
    /// integers without renumbering. Has NO effect on the depth buffer —
    /// ordering is resolved CPU-side before upload.
    pub z: f32,
    /// Hit-group id; 0 = non-interactive (not mirrored into `hits`).
    /// Non-zero values come from the widget registry id space — see
    /// `lib.rs::register_widget`.
    pub group: u32,
}

/// Frame-owned scene data — reused across frames (clear + refill, the
/// vectors keep their heap capacity → zero allocations in steady state).
#[derive(Default)]
pub struct Scene {
    /// Quad instances in push order. The instanced pass draws them in
    /// array order — painter's algorithm — so callers MUST push
    /// back-to-front, or pre-sort by `z` when order is data-driven.
    /// Practical count: 10²–10³ quads/frame is comfortable; 10⁵ starts
    /// to notice on the write_buffer (60B × N).
    pub quads: Vec<Quad>,
    /// Interactive rects — a compact mirror of quads pushed with a
    /// non-zero `group`. Kept separate deliberately: hit-tests scan this
    /// 24B-per-entry list instead of 60B Quads → ~2.5× more targets per
    /// cacheline on the pointer path.
    pub hits: Vec<Hit>,
}

/// Hit-test rect — quad-derived but trimmed to exactly what the pointer
/// path reads (rect + group + z). 28 bytes… padded to 28? no — [f32;4]
/// + u32 + f32 = 24 bytes, natural alignment, no padding.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Hit {
    /// [x0, y0, x1, y1] in CSS px — half-open: `x0 ≤ x < x1`, matching
    /// DOM rect conventions (a pointer on the shared edge belongs to the
    /// right-hand/bottom rect, not both).
    pub rect: [f32; 4],
    /// Group id mirrored from `Quad::group` — what `hit_test` returns.
    pub group: u32,
    /// Sort order — front-most (highest z) wins the scan.
    pub z: f32,
}

impl Scene {
    /// Registers a quad; when `group` is non-zero also mirrors it into the
    /// hit list for input drains. Called during the per-frame rebuild —
    /// `self.quads.clear()`/`hits.clear()` ran first so pushes stay
    /// allocation-free in steady state.
    ///
    /// The mirror is built HERE (not derived later) because the Quad
    /// fields are already in registers at push time — one branch +
    /// 24B write beats a second pass over the quad array.
    /// (Consumer lands with the static-content scene builder — burn
    /// sims emit straight into `quads` since particles aren't hitable.)
    #[allow(dead_code)]
    pub fn push(&mut self, quad: Quad) {
        if quad.group != 0 {
            self.hits.push(Hit {
                // Quad stores pos+size; Hit stores min+max edges.
                // x1 = x0 + w — width must be ≥0 or the rect is empty
                // and nothing can hit it (a degenerate quad is silently
                // non-interactive — intended).
                rect: [
                    quad.pos[0],
                    quad.pos[1],
                    quad.pos[0] + quad.size[0],
                    quad.pos[1] + quad.size[1],
                ],
                group: quad.group,
                z: quad.z,
            });
        }

        self.quads.push(quad);
    }

    /// Resolves (x, y) in CSS px to the front-most registered group, or
    /// `None` when the pointer is over dead space.
    ///
    /// # Algorithm: z-max linear scan
    ///
    /// Every hit that CONTAINS the point is a candidate; the highest-z
    /// candidate wins — the same rule the DOM compositor applies for
    /// overlapping elements. Complexity is O(n) with a tiny constant:
    /// each candidate costs one 4-way float compare. Interactive targets
    /// on a page number in the tens — a spatial tree (quadtree/BVH) would
    /// cost more in build time per frame than this scan costs total.
    /// If a future surface pushes past ~500 targets, bucket `hits` by
    /// coarse grid cell instead of switching to a real tree.
    ///
    /// The comparison `x >= h.rect[2]` makes the rect right/bottom edge
    /// EXCLUSIVE — two abutting rects never double-hit, and a pointer on
    /// the seam belongs to the later rect. Matches CSS hit-testing.
    pub fn hit_test(&self, x: f32, y: f32) -> Option<u32> {
        let mut best: Option<&Hit> = None;

        for h in &self.hits {
            // Reject fast: 4 ordered compares, all false → inside.
            // `x >= x1`/`y >= y1` implement the half-open edge above.
            if x < h.rect[0] || y < h.rect[1] || x >= h.rect[2] || y >= h.rect[3] {
                continue;
            }

            // Inside → keep only if it outranks the current best.
            // Equal z → FIRST pushed wins (stable painter's order —
            // later elements in DOM order sit on top, mirroring the
            // paint list's own semantics).
            if best.map_or(true, |b| h.z > b.z) {
                best = Some(h);
            }
        }

        best.map(|h| h.group)
    }
}
