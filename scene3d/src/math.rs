//! Scene3d math — the linear algebra every 3D renderer is built on.
//!
//! # Why this exists instead of glam/three-math
//!
//! Three.js ships a general-purpose math library (~60KB of the bundle)
//! covering matrices, quaternions, splines, and a dozen vector sizes.
//! The scenes this crate replaces use exactly: 3-vectors, 4×4 matrices,
//! TRS composition, one perspective projection, and ray/sphere picking.
//! Hand-rolling that subset keeps the wasm small AND keeps every number
//! auditable — each formula below carries its derivation so a reader
//! can change a constant and predict the result.
//!
//! # Conventions (must match what the shaders expect)
//!
//! - **Right-handed**: +X right, +Y up, +Z toward the viewer. Objects
//!   sit at negative Z in front of the camera — same as three.js.
//! - **Column vectors**: `v' = M · v` — a point transforms by
//!   multiplying the matrix on its LEFT. Matrices are stored
//!   column-major (`m[col][row]`, i.e. `m[0]` is the first COLUMN) so a
//!   `bytemuck` cast of the flat array lands in WGSL's `mat4x4<f32>`
//!   with zero reshuffling.
//! - **Clip space**: perspective output maps eye-space to x,y∈[-1,1],
//!   z∈[0,1] — the wgpu (DX12/Metal) convention, NOT three.js/OpenGL's
//!   z∈[-1,1]. Getting this wrong renders nothing or mirrors depth.
//!
//! # Performance note
//!
//! Every function here is a leaf of arithmetic — no allocation, no
//! branching beyond early-outs. `Mat4::mul` is 64 multiplies + 48 adds
//! in a fixed loop; on wasm32 that lowers to ~10ns. Transform
//! propagation for a 5k-node scene ≈ 50µs/frame of CPU — inside the
//! 1.5s/total-main-thread budget with orders of magnitude to spare.

// ── Vec3 ────────────────────────────────────────────────────────────

/// `Vec3` — a point or direction in 3D space. `repr(C)` so a slice of
/// these can be `bytemuck`-cast straight into a vertex buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3 {
    /// X component — +right in the scene's right-handed frame.
    pub x: f32,
    /// Y component — +up.
    pub y: f32,
    /// Z component — +toward viewer (objects in front of the camera
    /// have NEGATIVE z, matching three.js convention).
    pub z: f32,
}

impl Vec3 {
    /// `Vec3::new` — constructor kept as a const fn so compile-time
    /// constants (axis vectors, origins) can be `const` values.
    ///
    /// @param x Right-axis component.
    /// @param y Up-axis component.
    /// @param z Forward-axis component.
    /// @returns The assembled vector.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// `dot` — the scalar projection of `self` onto `o`:
    ///   dot = x·o.x + y·o.y + z·o.z = |self|·|o|·cos θ
    /// Ranges from +|a||b| (parallel) through 0 (perpendicular) to
    /// -|a||b| (antiparallel). Used for: ray-plane tests, back-face
    /// culls, orbit-camera spherical math. Cost: 3 mul + 2 add.
    ///
    /// @param o The vector to project onto.
    /// @returns Signed cosine-scaled projection length.
    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// `cross` — the vector PERPENDICULAR to both inputs:
    ///   c = (y·o.z − z·o.y,  z·o.x − x·o.z,  x·o.y − y·o.x)
    /// |c| = |a|·|b|·sin θ — zero when inputs are parallel (that case
    /// means "no unique perpendicular"; callers must guard).
    /// Handedness: `x̂ × ŷ = ẑ` — reversing argument order flips the
    /// sign (a×b = −b×a). Used to build camera basis vectors.
    ///
    /// @param o Second operand (order matters — sign flips if swapped).
    /// @returns Unit-scaled perpendicular, or the zero vector for
    ///          parallel inputs.
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// `length` — Euclidean norm √(x²+y²+z²). One sqrt (~4ns on
    /// modern cores, a single `f32.sqrt` instruction in wasm). Avoid
    /// when a SQUARED comparison suffices (`length_squared` — culling
    /// against a radius² needs no sqrt).
    ///
    /// @returns The vector's magnitude, ≥ 0.
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// `length_squared` — x²+y²+z². The cheap form for comparisons:
    /// `v.length_squared() < r*r` is true exactly when `|v| < r`.
    ///
    /// @returns Squared magnitude.
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// `normalized` — self ÷ |self|, mapping the vector onto the unit
    /// sphere. Direction preserved, magnitude becomes 1.
    ///
    /// Guard: returns `self` UNCHANGED for |v| < 1e-9 — dividing a
    /// near-zero vector would produce NaN/Inf and poison every
    /// downstream dot product. A 1e-9 threshold is far below any real
    /// scene dimension yet far above f32 denormals.
    ///
    /// @returns Unit-length copy, or the input if degenerate.
    pub fn normalized(self) -> Vec3 {
        let len = self.length();

        if len < 1e-9 {
            return self;
        }

        self.scale(1.0 / len)
    }

    /// `scale` — component-wise multiply by a scalar: (sx, sy, sz).
    /// Positive s preserves direction; negative flips; 0 collapses
    /// to the origin.
    ///
    /// @param s Scalar multiplier.
    /// @returns Scaled vector.
    pub fn scale(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    /// `add` — component-wise sum: translating a point by a vector, or
    /// summing forces. `a + b = (a.x+b.x, a.y+b.y, a.z+b.z)`.
    ///
    /// @param o Vector to add.
    /// @returns Sum vector.
    pub fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }

    /// `sub` — component-wise difference: `a − b` is the vector that
    /// points FROM b TO a (direction of `a` seen from `b`).
    ///
    /// @param o Vector to subtract.
    /// @returns Difference vector.
    pub fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }

    /// `lerp` — linear interpolation toward `o` by `t`:
    ///   r = self + (o − self)·t
    /// t=0 → self, t=1 → o. Values outside [0,1] extrapolate (allowed —
    /// spring overshoot uses t>1 deliberately).
    ///
    /// @param o Target vector.
    /// @param t Interpolation fraction.
    /// @returns Interpolated vector.
    pub fn lerp(self, o: Vec3, t: f32) -> Vec3 {
        self.add(o.sub(self).scale(t))
    }
}

// ── Mat4 ────────────────────────────────────────────────────────────

/// `Mat4` — a 4×4 column-major transform matrix. 64 bytes, one
/// cacheline — exactly the GPU's uniform-buffer record for a model
/// matrix. The `_[col][row]` interpretation: `m[0..4]` is column 0,
/// `m[12..15]` is the translation column (position offset lives in the
/// LAST column under the column-vector convention).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Mat4 {
    /// Flat column-major storage: element (row r, col c) is `m[c*4+r]`.
    pub m: [f32; 16],
}

impl Mat4 {
    /// `IDENTITY` — the do-nothing transform: diag(1,1,1,1), zeros
    /// elsewhere. `I·M = M` for every M — the neutral element of
    /// composition and the default for nodes with no local transform.
    pub const IDENTITY: Mat4 = Mat4 {
        m: [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ],
    };

    /// `mul` — matrix composition `self · o`.
    ///
    /// Math: element (r,c) of the product is row r of `self` dotted
    /// with column c of `o`: `out[c*4+r] = Σ_k self[k*4+r] · o[c*4+k]`.
    /// Composition order: `A·B` applies B FIRST, then A — so a node's
    /// world matrix is `parent_world · local`. Swapping the order
    /// applies transforms inside-out (rotate-around-self becomes
    /// rotate-around-world) — the classic scene-graph bug.
    ///
    /// Cost: 64 mul + 48 add — flat, branch-free, monomorphic.
    ///
    /// @param o Right-hand matrix (applied FIRST to vectors).
    /// @returns The composed transform.
    pub fn mul(self, o: Mat4) -> Mat4 {
        let mut out = [0.0f32; 16];

        for c in 0..4 {
            for r in 0..4 {
                out[c * 4 + r] = self.m[r]
                    * o.m[c * 4]
                    + self.m[4 + r] * o.m[c * 4 + 1]
                    + self.m[8 + r] * o.m[c * 4 + 2]
                    + self.m[12 + r] * o.m[c * 4 + 3];
            }
        }

        Mat4 { m: out }
    }

    /// `transform_point` — apply to a POINT (w=1, so translation
    /// applies): `v' = M · (x, y, z, 1)`. The w component of the result
    /// is kept: perspective matrices produce w ≠ 1 and the caller does
    /// the `÷w` perspective divide where needed (GPU does it in
    /// hardware after the vertex shader).
    ///
    /// @param p Point to transform.
    /// @returns `(transformed_xyz, w)` — w is 1 for affine transforms.
    pub fn transform_point(self, p: Vec3) -> (Vec3, f32) {
        let x = self.m[0] * p.x + self.m[4] * p.y + self.m[8] * p.z + self.m[12];
        let y = self.m[1] * p.x + self.m[5] * p.y + self.m[9] * p.z + self.m[13];
        let z = self.m[2] * p.x + self.m[6] * p.y + self.m[10] * p.z + self.m[14];
        let w = self.m[3] * p.x + self.m[7] * p.y + self.m[11] * p.z + self.m[15];

        (Vec3::new(x, y, z), w)
    }

    /// `translation` — build a pure-translate matrix: I with the
    /// offset in the last column. Points pick up +t; directions (w=0)
    /// are unaffected — the defining property of translation.
    ///
    /// @param t Offset vector in parent space.
    /// @returns Translation matrix.
    pub fn translation(t: Vec3) -> Mat4 {
        let mut m = Mat4::IDENTITY.m;

        m[12] = t.x;
        m[13] = t.y;
        m[14] = t.z;
        Mat4 { m }
    }

    /// `scale` — build a pure-scale matrix: diag(sx, sy, sz, 1).
    /// Uniform `s` preserves shape; per-axis values squash/stretch.
    /// A ZERO scale collapses the axis — degenerate triangles with
    /// zero-area normals; the shader still runs but lighting breaks
    /// (normalizes a zero vector). Keep every axis > ε.
    ///
    /// @param s Per-axis scale factors.
    /// @returns Scale matrix.
    pub fn scale(s: Vec3) -> Mat4 {
        let mut m = Mat4::IDENTITY.m;

        m[0] = s.x;
        m[5] = s.y;
        m[10] = s.z;
        Mat4 { m }
    }

    /// `rotation_y` — rotate about the UP axis by `a` radians (CCW when
    /// viewed from +Y looking down — right-hand rule).
    ///
    /// The y-rotation matrix:
    ///   ⎡  cos a   0   sin a   0 ⎤
    ///   ⎢    0     1     0     0 ⎥
    ///   ⎢ −sin a   0   cos a   0 ⎥
    ///   ⎣    0     0     0     1 ⎦
    /// Y stays fixed (row/col 1 is identity); the X/Z plane rotates.
    /// `a` is RADIANS — pass degrees × π/180. One full orbit is 2π ≈
    /// 6.2832; at 60fps `a += 0.001` is ~6°/s — a slow stately spin.
    ///
    /// @param a Angle in radians.
    /// @returns Rotation matrix.
    pub fn rotation_y(a: f32) -> Mat4 {
        let (s, c) = a.sin_cos();
        let mut m = Mat4::IDENTITY.m;

        m[0] = c;
        m[2] = -s;
        m[8] = s;
        m[10] = c;
        Mat4 { m }
    }

    /// `rotation_x` — rotate about the RIGHT axis by `a` radians
    /// (tilt up/down). The x-rotation matrix leaves column 0 fixed and
    /// rotates the Y/Z plane:
    ///   ⎡ 1    0       0      0 ⎤
    ///   ⎢ 0  cos a  −sin a    0 ⎥
    ///   ⎢ 0  sin a   cos a    0 ⎥
    ///   ⎣ 0    0       0      1 ⎦
    ///
    /// @param a Angle in radians.
    /// @returns Rotation matrix.
    pub fn rotation_x(a: f32) -> Mat4 {
        let (s, c) = a.sin_cos();
        let mut m = Mat4::IDENTITY.m;

        m[5] = c;
        m[6] = s;
        m[9] = -s;
        m[10] = c;
        Mat4 { m }
    }

    /// `perspective` — the wgpu-convention perspective projection.
    ///
    /// Derivation: a pinhole camera maps eye-space (x, y, z<0 in front)
    /// to clip space. With `f = 1/tan(fov_y/2)` (the focal length in
    /// half-height units):
    ///   ⎡ f/aspect   0      0            0      ⎤
    ///   ⎢    0       f      0            0      ⎥
    ///   ⎢    0       0   far/(near−far)  −1     ⎥  ← w = −z_eye
    ///   ⎣    0       0  (near·far)/(near−far) 0 ⎦
    /// After the hardware divide by w=−z: z' = (far·(z−near))/
    /// (z·(near−far)) — mapping z∈[near,far] → [0,1] (wgpu/D3D/Metal
    /// depth, NOT OpenGL's [-1,1]). The −1 in row 2 of the w-column is
    /// what makes w = −z_eye, enabling the divide.
    ///
    /// fov_y in RADIANS: 60° = 1.0472 is the natural human-ish lens;
    /// smaller = telephoto/flat, larger = wide/fisheye. aspect = w/h —
    /// pass CSS-pixels aspect or the image stretches when the canvas
    /// isn't square. near must be > 0 (the 1/(near−far) term divides
    /// by zero at near=0); far should bound the scene — a huge far
    /// with small near wastes depth precision (z-fighting between
    /// close objects).
    ///
    /// @param fov_y Vertical field of view, radians.
    /// @param aspect Canvas width ÷ height.
    /// @param near Near clip distance (> 0).
    /// @param far Far clip distance (> near).
    /// @returns Perspective projection matrix.
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (fov_y * 0.5).tan();
        let mut m = [0.0f32; 16];

        m[0] = f / aspect;
        m[5] = f;
        m[10] = far / (near - far);
        m[11] = -1.0;
        m[14] = (near * far) / (near - far);
        Mat4 { m }
    }

    /// `look_at` — camera-to-world inverse: builds the VIEW matrix that
    /// places the world in front of a camera at `eye` facing `target`
    /// with `up` as the roll reference.
    ///
    /// Basis construction (Gram-Schmidt-lite, 3 axes):
    ///   fwd  = (target − eye).normalized()      — where the camera looks
    ///   right = fwd × up, normalized             — perpendicular to both
    ///   up'  = right × fwd                       — re-orthogonalized up
    ///     (up' absorbs any tilt in the caller's `up` — passing a true
    ///     vertical and a fwd that's almost-vertical still yields a
    ///     valid basis; parallel fwd/up is the ONLY failure case —
    ///     cross returns ~0 and the basis collapses).
    /// The view matrix is the basis transposed (rotation⁻¹ = rotationᵀ
    /// for orthonormal bases — the cheap inverse) with the eye position
    /// baked in via −(eye·axis) translation terms.
    ///
    /// @param eye Camera position in world space.
    /// @param target Point the camera looks at.
    /// @param up Roll reference (usually +Y).
    /// @returns View matrix (world → eye space).
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let fwd = target.sub(eye).normalized();
        let right = fwd.cross(up).normalized();
        let up2 = right.cross(fwd);
        let mut m = Mat4::IDENTITY.m;

        // Rows of the basis (transposed columns = the inverse rotation),
        // translation = −(eye · basis-axis) folds the camera offset in.
        m[0] = right.x;
        m[4] = right.y;
        m[8] = right.z;
        m[1] = up2.x;
        m[5] = up2.y;
        m[9] = up2.z;
        m[2] = -fwd.x;
        m[6] = -fwd.y;
        m[10] = -fwd.z;
        m[12] = -eye.dot(right);
        m[13] = -eye.dot(up2);
        m[14] = eye.dot(fwd);
        Mat4 { m }
    }
}

// ── Ray / picking ───────────────────────────────────────────────────

/// `Ray` — an infinite half-line for picking: `p(t) = origin + dir·t`,
/// t ≥ 0. dir is normalized so `t` is in world-space units (t=1 → one
/// unit along dir).
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    /// Start point in world space.
    pub origin: Vec3,
    /// Unit direction.
    pub dir: Vec3,
}

impl Ray {
    /// `intersect_sphere` — closest hit of the ray on a sphere, or
    /// `None`. Solves |p(t) − c|² = r² for t — a quadratic in t:
    ///   (dir·dir)t² + 2·dir·(o−c)·t + (o−c)·(o−c) − r² = 0
    /// With dir normalized (a=1): t = −b/2 ± √(b²/4 − c_term) where
    /// `b = dir·oc`, `c_term = oc·oc − r²`. The discriminant
    /// `h = b² − c_term` < 0 → ray misses. We take the NEAR root
    /// (−b − √h); if the near root is behind the eye the far root is
    /// checked (eye inside the sphere).
    ///
    /// three.js's Raycaster does the same math per-object through a
    /// virtual call chain — here it's 20 scalar ops, allocation-free.
    ///
    /// @param center Sphere center in world space.
    /// @param radius Sphere radius.
    /// @returns Hit distance `t` along the ray, if any.
    pub fn intersect_sphere(&self, center: Vec3, radius: f32) -> Option<f32> {
        let oc = self.origin.sub(center);
        let b = oc.dot(self.dir);
        let c = oc.dot(oc) - radius * radius;
        let h = b * b - c;

        if h < 0.0 {
            return None;
        }

        let h = h.sqrt();
        let t = -b - h;

        if t >= 0.0 {
            Some(t)
        } else if -b + h >= 0.0 {
            Some(-b + h)
        } else {
            None
        }
    }
}

/// `ray_from_screen` — unproject a 2D screen position into a world ray.
///
/// The inverse of the render path: a pixel at NDC (nx, ny) sits on the
/// near plane; inverting the projection turns it into an eye-space
/// point, and the eye-space→world inverse (view⁻¹) completes the ray.
/// For a perspective camera the ray's eye-space direction simplifies
/// to (nx/f·aspect, ny/f, −1) — divide by the focal terms, look down
/// −Z. We then transform ONLY the direction (w=0 → translation-free)
/// by view⁻¹'s rotation and add eye.
///
/// @param nx NDC x ∈ [-1,1] (pixel_x/w · 2 − 1).
/// @param ny NDC y ∈ [-1,1], +up (flip the DOM's +down y first).
/// @param inv_view Inverse of the camera's view matrix.
/// @param tan_half_fov tan(fov_y/2) — the projection's `f` reciprocal.
/// @param aspect Canvas w/h.
/// @param eye Camera world position (ray origin).
/// @returns World-space picking ray.
pub fn ray_from_screen(
    nx: f32,
    ny: f32,
    inv_view: Mat4,
    tan_half_fov: f32,
    aspect: f32,
    eye: Vec3,
) -> Ray {
    // Eye-space direction: the point on the projection plane.
    let dx = nx * tan_half_fov * aspect;
    let dy = ny * tan_half_fov;
    let eye_dir = Vec3::new(dx, dy, -1.0);

    // Rotate (not translate — direction, w=0) into world space by the
    // inverse view's basis columns.
    let world_dir = Vec3::new(
        inv_view.m[0] * eye_dir.x + inv_view.m[4] * eye_dir.y + inv_view.m[8] * eye_dir.z,
        inv_view.m[1] * eye_dir.x + inv_view.m[5] * eye_dir.y + inv_view.m[9] * eye_dir.z,
        inv_view.m[2] * eye_dir.x + inv_view.m[6] * eye_dir.y + inv_view.m[10] * eye_dir.z,
    )
    .normalized();

    Ray {
        origin: eye,
        dir: world_dir,
    }
}
