//! Scene3d geometry — procedural mesh generators replacing three.js's
//! `*Geometry` classes for the nine shapes the scenes instantiate.
//!
//! # Output contract
//!
//! Every generator fills the same `Geometry` record — interleaved-free
//! SoA buffers (`positions`, `normals`, `uvs`, `indices`) that the
//! renderer uploads verbatim as GPU vertex/index buffers. Keeping
//! attributes in separate typed arrays (rather than three.js's
//! BufferAttribute objects) means a `bytemuck::cast_slice` is the whole
//! upload — zero marshaling between the crate and the vertex stage.
//!
//! # Conventions (must match the shaders + math.rs)
//!
//! - **Winding**: counter-clockwise when viewed from the front face —
//!   wgpu's default `FrontFace::Ccw`. Reversing an index triplet flips
//!   the face the same way `material.side` does in three.js.
//! - **Normals**: unit-length, pointing AWAY from the surface interior
//!   (outward for closed solids). The lighting shaders normalize again
//!   after interpolation — authoring unit vectors keeps the interpolated
//!   direction correct before that renormalize.
//! - **UVs**: `u` runs left→right, `v` runs BOTTOM→TOP (wgpu/WebGPU
//!   texture convention — `textureLoad` row 0 is the top of the image,
//!   so uv.y=0 samples the BOTTOM row). three.js OpenGL uses the same
//!   bottom-left origin, so ported UV math carries over unchanged.
//!
//! # Tessellation parameters
//!
//! `segments` arguments mean the same thing as in three.js: the number
//! of EDGES around/along a dimension (vertices = edges+1 per line).
//! More segments ⇒ smoother silhouette + more vertices/quadratic index
//! growth. The scenes' values (sphere 32×16, ring 64, cylinder 24) were
//! chosen for 60fps on integrated GPUs — keep ratios, not absolutes.

use crate::math::Vec3;
use std::f32::consts::{PI, TAU};

/// `Geometry` — one mesh's CPU-side vertex data. `positions`/`normals`
/// are xyz triplets (`len == 3·vertex_count`), `uvs` is uv pairs
/// (`len == 2·vertex_count`), `indices` addresses triplets of vertices
/// per triangle. Empty `indices` means non-indexed (draw the vertex
/// buffer linearly — used for point clouds and debug lines).
#[derive(Clone, Debug, Default)]
pub struct Geometry {
    /// Flat xyz positions, 3 f32 per vertex.
    pub positions: Vec<f32>,
    /// Flat xyz unit normals, 3 f32 per vertex — same count as positions.
    pub normals: Vec<f32>,
    /// Flat uv texture coordinates, 2 f32 per vertex — same vertex count.
    pub uvs: Vec<f32>,
    /// Triangle indices into the vertex arrays, 3 per face.
    pub indices: Vec<u32>,
}

impl Geometry {
    /// `vertex_count` — positions.len()/3, the canonical vertex total
    /// every other attribute must match (the renderer asserts this —
    /// mismatched attribute lengths are the classic "one black face"
    /// bug and are rejected at upload time rather than drawn wrong).
    ///
    /// @returns Number of vertices in the mesh.
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    /// `push_vertex` — appends one vertex's three attributes in one
    /// place so every generator stays a sequence of
    /// `push_vertex`+`push_tri` calls and can't drift attributes out
    /// of sync by construction.
    ///
    /// @param p Position (object space, pre-world-transform).
    /// @param n Outward unit normal at that position.
    /// @param uv Texture coordinate (u left→right, v bottom→top).
    pub(crate) fn push_vertex(&mut self, p: Vec3, n: Vec3, uv: [f32; 2]) {
        self.positions.extend_from_slice(&[p.x, p.y, p.z]);
        self.normals.extend_from_slice(&[n.x, n.y, n.z]);
        self.uvs.extend_from_slice(&uv);
    }
}

// ── Sphere ──────────────────────────────────────────────────────────

/// `sphere` — the lat/long sphere: `width_segments` meridians ×
/// `height_segments` parallels over the unit sphere scaled by `radius`.
///
/// Parameterization: vertex (i,j) sits at
///   φ = π · j/h            (polar angle, top pole → bottom pole)
///   θ = 2π · i/w           (azimuth around +Y, right-handed)
///   p = r · (sinφ·cosθ? no —
///   p = r · (−cosθ·sinφ, cosφ, sinθ·sinφ)  — see below.
///
/// We use the same convention three.js ships so ported texture
/// atlases keep their seam positions:
///   x = −r·cosθ·sinφ,  y = r·cosφ,  z = r·sinθ·sinφ
/// i.e. θ=0 faces −X; the seam runs at u=0/1 through −X.
/// The normal of a sphere at p is simply p̂ (center → surface), which
/// is why spheres are the cheapest curved surface to light — the
/// normal field is free, no tangent-frame math needed.
///
/// UVs: u = θ/2π (azimuth wraps the texture), v = 1 − φ/π so v=1 is
/// the top pole — images land right-side-up under the bottom-origin
/// wgpu convention.
///
/// Vertex count: (w+1)·(h+1) — the seam column duplicates so the
/// texture doesn't smear between u=1→0. Index count: w·h·6.
///
/// @param radius Sphere radius in object units.
/// @param width_segments Meridian edges around the equator (≥3).
/// @param height_segments Parallel edges pole→pole (≥2).
/// @returns Indexed triangle mesh of the sphere.
pub fn sphere(radius: f32, width_segments: u32, height_segments: u32) -> Geometry {
    let w = width_segments.max(3) as usize;
    let h = height_segments.max(2) as usize;
    let mut g = Geometry::default();

    g.positions.reserve((w + 1) * (h + 1) * 3);
    g.normals.reserve((w + 1) * (h + 1) * 3);
    g.uvs.reserve((w + 1) * (h + 1) * 2);
    g.indices.reserve(w * h * 6);

    for j in 0..=h {
        // φ measured from the +Y pole downward.
        let phi = PI * (j as f32) / (h as f32);
        let (sp, cp) = phi.sin_cos();

        for i in 0..=w {
            let theta = TAU * (i as f32) / (w as f32);
            let (st, ct) = theta.sin_cos();

            // Unit-sphere direction — IS the normal for a sphere.
            let n = Vec3::new(-ct * sp, cp, st * sp);
            let p = n.scale(radius);

            g.push_vertex(p, n, [i as f32 / w as f32, 1.0 - j as f32 / h as f32]);
        }
    }

    // Two triangles per quad cell, CCW seen from outside:
    //   a=(i,j)  b=(i,j+1)  c=(i+1,j)  d=(i+1,j+1)
    // → (a,b,c) + (b,d,c). The seam column (i=w) reuses index w·row —
    // the duplicated seam vertex keeps UV continuity.
    for j in 0..h {
        for i in 0..w {
            let a = (j * (w + 1) + i) as u32;
            let b = ((j + 1) * (w + 1) + i) as u32;
            let c = (j * (w + 1) + i + 1) as u32;
            let d = ((j + 1) * (w + 1) + i + 1) as u32;

            g.indices.extend_from_slice(&[a, b, c]);
            g.indices.extend_from_slice(&[b, d, c]);
        }
    }

    g
}

// ── Plane ───────────────────────────────────────────────────────────

/// `plane` — an axis-aligned quad in the XY plane, z=0, facing +Z.
/// `width`/`height` are full extents (half-extents ±w/2, ±h/2) so the
/// mesh is centred on its own origin — the same convention three.js
/// PlaneGeometry uses, which keeps `mesh.position` placement math
/// identical when porting scenes.
///
/// Normals: every vertex is (0,0,1) — the face is flat, interpolation
/// is trivially exact. UVs: (0,0) bottom-left → (1,1) top-right.
///
/// Vertex count: 4 (corners). Indices: 2 triangles, CCW from +Z:
/// (bl, tr, tl) + (bl, br, tr).
///
/// @param width Full x extent.
/// @param height Full y extent.
/// @returns Indexed quad mesh.
pub fn plane(width: f32, height: f32) -> Geometry {
    let mut g = Geometry::default();
    let hw = width * 0.5;
    let hh = height * 0.5;
    let n = Vec3::new(0.0, 0.0, 1.0);

    // Corner order: 0=bl, 1=br, 2=tl, 3=tr — chosen so the first
    // triangle (0,3,2) reads counter-clockwise when viewed from +Z.
    g.push_vertex(Vec3::new(-hw, -hh, 0.0), n, [0.0, 0.0]);
    g.push_vertex(Vec3::new(hw, -hh, 0.0), n, [1.0, 0.0]);
    g.push_vertex(Vec3::new(-hw, hh, 0.0), n, [0.0, 1.0]);
    g.push_vertex(Vec3::new(hw, hh, 0.0), n, [1.0, 1.0]);

    g.indices.extend_from_slice(&[0, 3, 2, 0, 1, 3]);

    g
}

// ── Ring ────────────────────────────────────────────────────────────

/// `ring` — a flat annulus in the XY plane (planet rings, halo discs):
/// `seg+1` concentric vertex PAIRS (inner+outer) swept around the
/// circle, yielding `seg` quad cells = `seg`·2 triangles.
///
/// Vertex k sits at θ=2πk/seg on BOTH radii:
///   inner = (ri·cosθ, ri·sinθ, 0),  outer = (ro·cosθ, ro·sinθ, 0)
/// Normal: +Z everywhere (flat). UVs: radial sweep mapped so the
/// texture samples around the ring — u = θ/2π, v = 0 at inner rim,
/// v = 1 at outer rim. A band texture therefore reads inner→outer,
/// which is how the star-field ring gradient atlas is authored.
///
/// @param inner_radius Hole radius — the empty centre.
/// @param outer_radius Rim radius.
/// @param segments Radial edges around the circle (≥3).
/// @returns Indexed annulus mesh.
pub fn ring(inner_radius: f32, outer_radius: f32, segments: u32) -> Geometry {
    let seg = segments.max(3) as usize;
    let mut g = Geometry::default();
    let n = Vec3::new(0.0, 0.0, 1.0);

    // Duplicate vertex pair at θ=2π (k=seg) so the seam doesn't fold.
    for k in 0..=seg {
        let theta = TAU * (k as f32) / (seg as f32);
        let (st, ct) = theta.sin_cos();
        let u = k as f32 / seg as f32;

        g.push_vertex(
            Vec3::new(inner_radius * ct, inner_radius * st, 0.0),
            n,
            [u, 0.0],
        );
        g.push_vertex(
            Vec3::new(outer_radius * ct, outer_radius * st, 0.0),
            n,
            [u, 1.0],
        );
    }

    // Pair k = (2k inner, 2k+1 outer). Cell (k,k+1):
    //   a=inner_k  b=outer_k  c=inner_k+1  d=outer_k+1
    // → (a,d,b) + (a,c,d) — CCW from +Z.
    for k in 0..seg {
        let a = (2 * k) as u32;
        let b = a + 1;
        let c = a + 2;
        let d = a + 3;

        g.indices.extend_from_slice(&[a, d, b]);
        g.indices.extend_from_slice(&[a, c, d]);
    }

    g
}

// ── Box ─────────────────────────────────────────────────────────────

/// `box_mesh` — the axis-aligned box: 6 faces × 4 vertices × 2
/// triangles = 24 verts, 36 indices. Per-face vertex duplication is
/// deliberate — a cube's normals are discontinuous at edges (each face
/// needs its own +axis normal), so sharing corners would average
/// normals into a blob-shaded mess. This is THE reason three.js
/// BoxGeometry carries 24 vertices for 8 corners.
///
/// `w`,`h`,`d` are full extents, centred on origin.
///
/// @param w Full x extent. @param h Full y extent. @param d Full z extent.
/// @returns Indexed box mesh.
pub fn box_mesh(w: f32, h: f32, d: f32) -> Geometry {
    let mut g = Geometry::default();
    let (hw, hh, hd) = (w * 0.5, h * 0.5, d * 0.5);

    // (normal axis, half-extent sign, tangent axes (u,v)) — six faces.
    // For each face the four corners are swept in face-local uv space;
    // winding is fixed CCW seen from OUTSIDE by ordering tl→tr→bl→br
    // along the face's local axes and emitting (0,1,2)+(1,3,2).
    let faces: [(Vec3, [f32; 3], [f32; 3]); 6] = [
        // n, u-axis extents, v-axis extents — face centre = n·half.
        (Vec3::new(0.0, 0.0, 1.0), [hw, 0.0, 0.0], [0.0, hh, 0.0]), // +Z
        (Vec3::new(0.0, 0.0, -1.0), [-hw, 0.0, 0.0], [0.0, hh, 0.0]), // -Z
        (Vec3::new(0.0, 1.0, 0.0), [hw, 0.0, 0.0], [0.0, 0.0, -hd]), // +Y
        (Vec3::new(0.0, -1.0, 0.0), [hw, 0.0, 0.0], [0.0, 0.0, hd]), // -Y
        (Vec3::new(1.0, 0.0, 0.0), [0.0, 0.0, -hd], [0.0, hh, 0.0]), // +X
        (Vec3::new(-1.0, 0.0, 0.0), [0.0, 0.0, hd], [0.0, hh, 0.0]), // -X
    ];

    for (n, ua, va) in faces {
        let base = g.vertex_count() as u32;

        // Face centre sits one half-extent along the normal.
        let half = Vec3::new(
            n.x * (if n.x != 0.0 { hw } else { 0.0 })
                + n.y * (if n.y != 0.0 { hh } else { 0.0 })
                + n.z * (if n.z != 0.0 { hd } else { 0.0 }),
            n.y * (if n.y != 0.0 { hh } else { 0.0 })
                + n.x * (if n.x != 0.0 { hw } else { 0.0 })
                + n.z * (if n.z != 0.0 { hd } else { 0.0 }),
            n.z * (if n.z != 0.0 { hd } else { 0.0 })
                + n.x * (if n.x != 0.0 { hw } else { 0.0 })
                + n.y * (if n.y != 0.0 { hh } else { 0.0 }),
        );

        // Corners at centre ± u-axis ± v-axis, in uv order (0,0),(1,0),(0,1),(1,1).
        for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            let p = Vec3::new(
                half.x + ua[0] * su + va[0] * sv,
                half.y + ua[1] * su + va[1] * sv,
                half.z + ua[2] * su + va[2] * sv,
            );
            let uv = [(su + 1.0) * 0.5, (sv + 1.0) * 0.5];

            g.push_vertex(p, n, uv);
        }

        g.indices
            .extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
    }

    g
}

// ── Cylinder ────────────────────────────────────────────────────────

/// `cylinder` — a tube between two rings: `radial_segments` edges
/// around the circle, side wall only (open-ended — the scenes use
/// cylinders for orbit lines and shells, never capped rods; caps would
/// add `2·seg` vertices for geometry the camera never sees).
///
/// Vertex ring t ∈ {0,1}: y = ∓h/2, radius = {r_bottom, r_top} so
/// r_top≠r_bottom draws a cone/frustum (orbit cone shells).
///
/// Side normal at θ: the surface direction ⊥ both the meridian slope
/// and the tangent — for a frustum the normal tilts with slope
/// s = (r_bottom − r_top)/h:
///   n = normalize( cosθ, s, sinθ )
/// i.e. the circle direction lifted by the slope — a pure cone
/// (r_top=0) leans further up, a true cylinder (s=0) stays radial.
///
/// UVs: u = θ/2π wraps; v = 0 bottom → 1 top.
///
/// @param radius_top Upper ring radius.
/// @param radius_bottom Lower ring radius.
/// @param height Full y extent.
/// @param radial_segments Edges around (≥3).
/// @returns Indexed open-ended tube mesh.
pub fn cylinder(
    radius_top: f32,
    radius_bottom: f32,
    height: f32,
    radial_segments: u32,
) -> Geometry {
    let seg = radial_segments.max(3) as usize;
    let mut g = Geometry::default();
    // Meridian slope — vertical rise per unit radial run (see doc).
    let slope = (radius_bottom - radius_top) / height.max(f32::EPSILON);

    for t in 0..2usize {
        let y = if t == 0 { -height * 0.5 } else { height * 0.5 };
        let r = if t == 0 { radius_bottom } else { radius_top };
        let v = t as f32;

        for k in 0..=seg {
            let theta = TAU * (k as f32) / (seg as f32);
            let (st, ct) = theta.sin_cos();
            let n = Vec3::new(ct, slope, st).normalized();

            g.push_vertex(Vec3::new(r * ct, y, r * st), n, [k as f32 / seg as f32, v]);
        }
    }

    // Same row-pair walk as `ring`, but between t=0 and t=1 rings.
    let row = seg + 1;
    for k in 0..seg {
        let a = k as u32; // bottom k
        let b = (k + row) as u32; // top k
        let c = (k + 1) as u32; // bottom k+1
        let d = (k + 1 + row) as u32; // top k+1

        g.indices.extend_from_slice(&[a, b, c]);
        g.indices.extend_from_slice(&[c, b, d]);
    }

    g
}

// ── Icosahedron ─────────────────────────────────────────────────────

/// `icosahedron` — a subdivided regular icosahedron projected onto the
/// sphere of `radius`: the classic "geodesic ball" — every triangle is
/// near-equilateral so detail distributes evenly (unlike the lat/long
/// sphere which bunches at the poles). `detail` subdivisions split each
/// edge into `2^detail` segments.
///
/// The 12 base vertices of an icosahedron are the rectangle corners of
/// three mutually perpendicular golden rectangles — with φ=(1+√5)/2:
///   (0,±1,±φ)  (±1,±φ,0)  (±φ,0,±1)
/// Each is projected onto the unit sphere at generation time.
///
/// detail=0 ⇒ the 20-face platonic solid; each extra detail level
/// quadruples face count (every triangle splits into 4 by inserting
/// edge midpoints, all re-projected to the sphere).
///
/// Normals: as a projected sphere, n = p̂ — free again.
/// UVs: equirectangular (u=azimuth/2π, v=elevation/π+½) — enough for
/// the atmospheric shells the scenes apply this to.
///
/// @param radius Sphere radius the icosahedron is projected onto.
/// @param detail Subdivision exponent — 0..3 is the sane range
///               (detail=3 ⇒ 20·4³ = 1280 faces).
/// @returns Indexed geodesic sphere mesh.
pub fn icosahedron(radius: f32, detail: u32) -> Geometry {
    const PHI: f32 = 1.618_034;

    // The 12 corner directions of the three golden rectangles.
    let base: [[f32; 3]; 12] = [
        [0.0, 1.0, PHI],
        [0.0, -1.0, PHI],
        [0.0, 1.0, -PHI],
        [0.0, -1.0, -PHI],
        [1.0, PHI, 0.0],
        [-1.0, PHI, 0.0],
        [1.0, -PHI, 0.0],
        [-1.0, -PHI, 0.0],
        [PHI, 0.0, 1.0],
        [-PHI, 0.0, 1.0],
        [PHI, 0.0, -1.0],
        [-PHI, 0.0, -1.0],
    ];

    // The 20 faces — index triplets chosen CCW-outward.
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 1, 8],
        [0, 8, 4],
        [0, 4, 5],
        [0, 5, 9],
        [0, 9, 1],
        [1, 6, 8],
        [8, 6, 10],
        [8, 10, 4],
        [4, 10, 2],
        [4, 2, 5],
        [5, 2, 11],
        [5, 11, 9],
        [9, 11, 7],
        [9, 7, 1],
        [1, 7, 6],
        [6, 7, 3],
        [6, 3, 10],
        [10, 3, 2],
        [2, 3, 11],
        [11, 3, 7],
    ];

    // Base vertex table — projected to unit sphere for normals.
    let mut verts: Vec<Vec3> = base
        .iter()
        .map(|c| Vec3::new(c[0], c[1], c[2]).normalized())
        .collect();

    // Subdivide: split every triangle (a,b,c) into 4 by edge midpoints
    // m_ab, m_bc, m_ca — each re-projected to the sphere. A HashMap on
    // the sorted edge key dedupes midpoints shared between faces so
    // vertex count stays 10·4^d+2 instead of 3× per face.
    for _ in 0..detail.min(4) {
        let mut next: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
        let mut mid_cache: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();

        // midpoint(i,j) — cached by sorted edge key (i<<32|j).
        let mut midpoint = |i: u32, j: u32, verts: &mut Vec<Vec3>| -> u32 {
            let key = ((i.min(j) as u64) << 32) | (i.max(j) as u64);

            if let Some(&m) = mid_cache.get(&key) {
                return m;
            }

            let m = Vec3::new(
                (verts[i as usize].x + verts[j as usize].x) * 0.5,
                (verts[i as usize].y + verts[j as usize].y) * 0.5,
                (verts[i as usize].z + verts[j as usize].z) * 0.5,
            )
            .normalized();

            verts.push(m);

            let idx = (verts.len() - 1) as u32;

            mid_cache.insert(key, idx);

            idx
        };

        for [a, b, c] in faces {
            let ab = midpoint(a, b, &mut verts);
            let bc = midpoint(b, c, &mut verts);
            let ca = midpoint(c, a, &mut verts);

            next.push([a, ab, ca]);
            next.push([ab, b, bc]);
            next.push([ca, bc, c]);
            next.push([ab, bc, ca]);
        }

        faces = next;
    }

    let mut g = Geometry::default();

    for v in &verts {
        let p = v.scale(radius);
        // Equirectangular UV — azimuth around +Y, elevation −π/2..π/2.
        let u = (v.z.atan2(v.x) / TAU) + 0.5;
        let vt = v.y.asin() / PI + 0.5;

        g.push_vertex(p, *v, [u, vt]);
    }

    g.indices = faces.concat();

    g
}

// ── Torus ───────────────────────────────────────────────────────────

/// `torus` — a ring with tube thickness (accretion discs, gyro rings):
/// `tubular` segments around the big ring × `radial` segments around
/// the tube cross-section.
///
/// Parameterization — big angle θ around the torus axis (+Z-facing
/// ring lies in the XY plane, matching three.js), small angle φ around
/// the tube:
///   centre(θ) = R·(cosθ, sinθ, 0)            — spine of the torus
///   p(θ,φ) = centre(θ) + tube·(cosφ·cosθ, cosφ·sinθ, sinφ)
/// The normal is the offset direction p − centre normalized:
///   n = (cosφ·cosθ, cosφ·sinθ, sinφ)         — already unit-length.
/// i.e. the tube direction IS the normal — like the sphere, the torus
/// gets its normal field for free from its own parameterization.
///
/// UVs: u = θ/2π around the ring, v = φ/2π around the tube.
///
/// @param radius Distance origin→tube centre (R, the "big" radius).
/// @param tube Tube cross-section radius.
/// @param radial_segments Edges around the tube (≥3).
/// @param tubular_segments Edges around the ring (≥3).
/// @returns Indexed torus mesh.
pub fn torus(radius: f32, tube: f32, radial_segments: u32, tubular_segments: u32) -> Geometry {
    let rs = radial_segments.max(3) as usize;
    let ts = tubular_segments.max(3) as usize;
    let mut g = Geometry::default();

    for j in 0..=rs {
        let phi = TAU * (j as f32) / (rs as f32);
        let (sph, cph) = phi.sin_cos();

        for i in 0..=ts {
            let theta = TAU * (i as f32) / (ts as f32);
            let (st, ct) = theta.sin_cos();

            // Tube direction = normal (see doc).
            let n = Vec3::new(cph * ct, cph * st, sph);
            let p = Vec3::new(
                (radius + tube * cph) * ct,
                (radius + tube * cph) * st,
                tube * sph,
            );

            g.push_vertex(p, n, [i as f32 / ts as f32, j as f32 / rs as f32]);
        }
    }

    let row = ts + 1;
    for j in 0..rs {
        for i in 0..ts {
            let a = (j * row + i) as u32;
            let b = ((j + 1) * row + i) as u32;
            let c = (j * row + i + 1) as u32;
            let d = ((j + 1) * row + i + 1) as u32;

            // (a,c,b)+(b,c,d) — ring direction × tube direction reads
            // CCW from outside the tube (verified by signed volume).
            g.indices.extend_from_slice(&[a, c, b]);
            g.indices.extend_from_slice(&[b, c, d]);
        }
    }

    g
}

// ── Tetrahedron ─────────────────────────────────────────────────────

/// `tetrahedron` — the 4-vertex platonic solid (debris fields, shard
/// markers): the minimum closed solid — 4 faces, 4 corners.
///
/// The four corners of a regular tetrahedron inscribed in the unit
/// sphere can be written as alternate cube vertices:
///   (1,1,1)  (1,−1,−1)  (−1,1,−1)  (−1,−1,1)
/// normalized then scaled by `radius`. Face normals are the normalized
/// sum of the face's two edge vectors — or equivalently the direction
/// OPPOSITE the vertex not in the face (each face stares away from its
/// opposite corner). We use per-face flat normals (each face gets its
/// own three duplicate vertices — like the box, sharp shading needs
/// non-shared corners).
///
/// UVs: each face maps its triangle into the lower-left unit triangle
/// (0,0),(1,0),(0,1) — a flat face has no canonical unwrap; the shards
/// this generates are tinted flat anyway.
///
/// @param radius Circumscribed sphere radius.
/// @returns Non-shared-corner tetrahedron mesh (12 verts, 4 tris).
pub fn tetrahedron(radius: f32) -> Geometry {
    let mut g = Geometry::default();
    let inv = 1.0 / 3f32.sqrt();

    let v: [Vec3; 4] = [
        Vec3::new(inv, inv, inv),
        Vec3::new(inv, -inv, -inv),
        Vec3::new(-inv, inv, -inv),
        Vec3::new(-inv, -inv, inv),
    ];

    // Each face omits one corner — its normal points opposite that
    // corner (face 0 omits v0, etc).
    for omit in 0..4 {
        let n = v[omit].scale(-1.0).normalized();
        let base = g.vertex_count() as u32;

        let uvs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mut k = 0usize;

        for corner in 0..4 {
            if corner == omit {
                continue;
            }

            g.push_vertex(v[corner].scale(radius), n, uvs[k]);
            k += 1;
        }

        // Ascending corner order winds outward only when the OMITTED
        // index is odd — even omits need the last two indices swapped
        // (the signed-volume test below verifies all four faces).
        if omit % 2 == 0 {
            g.indices.extend_from_slice(&[base, base + 2, base + 1]);
        } else {
            g.indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
    }

    g
}

// ── Tests ───────────────────────────────────────────────────────────
//
// Host-target unit tests — geometry is pure CPU math, so these run
// under `cargo test` without a GPU. The key invariant for every
// generator: indices stay in bounds, normals are unit-length, and
// closed solids report POSITIVE signed volume (winding truly is
// CCW-outward — an inward face list would invert it and render
// inside-out under backface culling).
#[cfg(test)]
mod tests {
    use super::*;

    /// Signed volume via the divergence theorem:
    ///   V = (1/6) Σ_faces a·(b×c)
    /// positive ⇒ triangles wind CCW seen from outside.
    fn signed_volume(g: &Geometry) -> f32 {
        let p = &g.positions;
        let mut v = 0.0f32;

        for t in g.indices.chunks_exact(3) {
            let (a, b, c) = (t[0] as usize * 3, t[1] as usize * 3, t[2] as usize * 3);
            let (ax, ay, az) = (p[a], p[a + 1], p[a + 2]);
            let (bx, by, bz) = (p[b], p[b + 1], p[b + 2]);
            let (cx, cy, cz) = (p[c], p[c + 1], p[c + 2]);

            v += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
        }

        v / 6.0
    }

    /// Asserts indices in-bounds + every normal unit-length.
    fn check_basics(g: &Geometry) {
        let n = g.vertex_count();

        assert_eq!(g.normals.len(), n * 3);
        assert_eq!(g.uvs.len(), n * 2);

        for &i in &g.indices {
            assert!((i as usize) < n);
        }

        for c in g.normals.chunks_exact(3) {
            let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();

            assert!((l - 1.0).abs() < 1e-4, "normal length {l}");
        }
    }

    #[test]
    fn sphere_is_closed_and_outward() {
        let g = sphere(2.0, 16, 8);

        check_basics(&g);
        assert_eq!(g.vertex_count(), 17 * 9);
        assert!(signed_volume(&g) > 30.0); // ~4/3·π·8 = 33.5
    }

    #[test]
    fn box_is_closed_and_outward() {
        let g = box_mesh(2.0, 2.0, 2.0);

        check_basics(&g);
        assert_eq!(g.vertex_count(), 24);
        assert!(signed_volume(&g) > 7.9); // 8 within float noise
    }

    #[test]
    fn icosahedron_is_outward_at_all_details() {
        for d in 0..=3 {
            let g = icosahedron(1.0, d);

            check_basics(&g);
            assert!(signed_volume(&g) > 0.0, "detail {d} wound inward");
        }
    }

    #[test]
    fn tetrahedron_is_outward() {
        let g = tetrahedron(1.0);

        check_basics(&g);
        assert_eq!(g.vertex_count(), 12);
        assert!(signed_volume(&g) > 0.0);
    }

    #[test]
    fn torus_is_closed_and_outward() {
        let g = torus(2.0, 0.5, 24, 32);

        check_basics(&g);
        // 2π²·R·r² ≈ 9.87 for R=2,r=0.5 — signed volume verifies the
        // torus's winding even though the hole makes "inside"
        // ambiguous geometrically. The polyhedral estimate
        // underestimates by ~1/N² chord error, so assert at 9.5.
        let v = signed_volume(&g);

        assert!(v > 9.5, "torus volume {v}");
    }

    #[test]
    fn cylinder_and_ring_and_plane_have_valid_tables() {
        check_basics(&cylinder(0.4, 0.6, 2.0, 12));
        check_basics(&ring(0.5, 1.0, 24));
        check_basics(&plane(2.0, 1.0));
    }
}
