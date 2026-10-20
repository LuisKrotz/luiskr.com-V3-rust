//! Scene3d camera — the PerspectiveCamera replacement: projection +
//! view matrices, an orbit rig (the scenes' only camera control), and
//! the frame uniform the WGSL reads.
//!
//! # The frame uniform
//!
//! Every draw pipeline shares one 48-f32 camera uniform, written once
//! per frame into slot 0 of the uniform ring:
//!   [0..16]  view_proj    — proj·view, the clip-space transform
//!   [16..19] eye.xyz      — camera world position (specular vectors)
//!   [19]     time         — seconds (shader-side animation clocks)
//!   [20..23] fog          — (color.rgb packed? no — see below)
//!   [24..27] light_pos    — the scenes' single PointLight position
//!   [28..31] light_color  — point light rgb
//!   [32..35] light_range  — (intensity, range, 0, 0)
//!   [36..39] canvas       — (w, h, 1/h, pixel_ratio) for point sizes
//!   [40..48] reserved     — post-chain params (bloom threshold…)
//! One binding, one write, every draw — the alternative (per-pipeline
//! uniforms) multiplies both the buffer writes and the bind-group
//! count without buying anything at this scene size.

use crate::math::{Mat4, Quat, Vec3};

/// `Camera` — a perspective camera with an optional orbit rig.
///
/// The transform model matches the scene graph: `position`/`rotation`
/// are the camera's world TRS (scale is meaningless for a camera and
/// omitted); `view` is the INVERSE of that world transform — the
/// matrix that moves the world into eye space.
#[derive(Clone, Debug)]
pub struct Camera {
    /// Vertical field of view in radians (three.js stores degrees;
    /// storing radians kills a per-frame multiply — `fov°` sets once).
    pub fov_y: f32,
    /// Near/far clip distances. Keep near ≥ 0.1 — depth precision is
    /// ~1/near at the far end; a tiny near wastes the whole buffer on
    /// the first metre.
    pub near: f32,
    /// Far clip — the star-field uses ~4000.
    pub far: f32,
    /// World position of the eye.
    pub position: Vec3,
    /// World orientation.
    pub rotation: Quat,
    /// Camera layer mask — `collect` gates nodes against it.
    pub layers: u32,
    /// Orbit rig: when Some, `position`/`rotation` are derived each
    /// frame from (target, distance, yaw, pitch) — the only camera
    /// motion the experiments use (OrbitControls' spherical core).
    pub orbit: Option<Orbit>,
}

/// `Orbit` — spherical camera rig: the eye sits at
/// `target + R(yaw,pitch)·(0,0,distance)` and faces −Z back at the
/// target — OrbitControls' essential math without the zoom/damping
/// state machine (the engines drive yaw/pitch directly per frame).
#[derive(Clone, Copy, Debug)]
pub struct Orbit {
    /// The point the camera circles + looks at.
    pub target: Vec3,
    /// Eye distance from target.
    pub distance: f32,
    /// Rotation about +Y, radians.
    pub yaw: f32,
    /// Elevation above the equatorial plane, radians (clamped ±(π/2−ε)
    /// — at the poles the up vector degenerates to zero and look_at
    /// produces NaNs; 0.01 keeps a visible pole).
    pub pitch: f32,
}

impl Camera {
    /// `Camera::new` — defaults matching the scenes: 45° fov, near
    /// 0.1, far 4000, at origin looking down −Z.
    ///
    /// @param fov_y_deg Vertical field of view in DEGREES (converted
    ///        once at construction — three.js parity for ported calls).
    /// @param near Near clip. @param far Far clip.
    /// @returns The camera.
    pub fn new(fov_y_deg: f32, near: f32, far: f32) -> Camera {
        Camera {
            fov_y: fov_y_deg.to_radians(),
            near,
            far,
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            layers: 1,
            orbit: None,
        }
    }

    /// `apply_orbit` — resolve the rig into position+rotation when set.
    /// Position: spherical→cartesian —
    ///   x = d·cos(pitch)·sin(yaw)
    ///   y = d·sin(pitch)
    ///   z = d·cos(pitch)·cos(yaw)
    /// (yaw=0 faces the target from +Z — same as OrbitControls).
    /// Rotation: look_at(target) — the camera faces −Z at the target.
    fn apply_orbit(&mut self) {
        let Some(o) = self.orbit else { return };

        // Clamp pitch just off the poles — look_at's up-vector cross
        // degenerates exactly at ±90° (see Orbit.pitch doc).
        let pitch = o.pitch.clamp(-1.56, 1.56);
        let (sp, cp) = pitch.sin_cos();
        let (sy, cy) = o.yaw.sin_cos();

        self.position = Vec3::new(
            o.target.x + o.distance * cp * sy,
            o.target.y + o.distance * sp,
            o.target.z + o.distance * cp * cy,
        );

        let view = Mat4::look_at(self.position, o.target, Vec3::new(0.0, 1.0, 0.0));

        self.rotation = Quat::from_mat4(&view_rot_to_world(&view));
    }

    /// `view` — the eye-space transform: inverse of the camera's world
    /// TRS. For a rigid transform M = R·T, M⁻¹ = T⁻¹·R⁻¹ =
    /// translate(−p)·rotate(q̄) — two cheap matrices instead of a
    /// general 4×4 inverse.
    ///
    /// @returns The view matrix.
    pub fn view(&mut self) -> Mat4 {
        self.apply_orbit();

        // Inverse rotation: the conjugate quaternion (negate xyz).
        let qi = Quat::new(
            -self.rotation.x,
            -self.rotation.y,
            -self.rotation.z,
            self.rotation.w,
        );
        let rot = qi.to_mat4();
        let tra = Mat4::translation(Vec3::new(
            -self.position.x,
            -self.position.y,
            -self.position.z,
        ));

        rot.mul(tra)
    }

    /// `proj` — the wgpu-convention perspective matrix (z→[0,1]) for
    /// the current canvas aspect.
    ///
    /// @param aspect Canvas width/height.
    /// @returns The projection matrix.
    pub fn proj(&self, aspect: f32) -> Mat4 {
        Mat4::perspective(self.fov_y, aspect, self.near, self.far)
    }

    /// `write_frame_uniform` — fill the 48-f32 camera block (see the
    /// module doc for the field map). `view_proj` is proj·view (proj
    /// applied LAST — the vector multiplies on the left in WGSL:
    /// `out = view_proj * vec4(pos,1)`).
    ///
    /// @param out The 48-f32 destination.
    /// @param canvas_w Canvas backing width px. @param canvas_h Height.
    /// @param dpr Device pixel ratio (point-size compensation).
    /// @param time Elapsed seconds (shader clocks).
    /// @param light (position, rgb, intensity) — the scene's PointLight.
    pub fn write_frame_uniform(
        &mut self,
        out: &mut [f32],
        canvas_w: f32,
        canvas_h: f32,
        dpr: f32,
        time: f32,
        light: (Vec3, Vec3, f32),
    ) {
        self.apply_orbit();

        let aspect = canvas_w / canvas_h.max(1.0);
        let vp = self.proj(aspect).mul(self.view());

        out.fill(0.0);
        out[0..16].copy_from_slice(&vp.m);
        out[16] = self.position.x;
        out[17] = self.position.y;
        out[18] = self.position.z;
        out[19] = time;
        out[24] = light.0.x;
        out[25] = light.0.y;
        out[26] = light.0.z;
        out[28] = light.1.x;
        out[29] = light.1.y;
        out[30] = light.1.z;
        out[32] = light.2;
        out[36] = canvas_w;
        out[37] = canvas_h;
        out[38] = 1.0 / canvas_h.max(1.0);
        out[39] = dpr;
    }
}

/// `view_rot_to_world` — extract the world-orientation basis from a
/// look_at view matrix: the view's 3×3 rotation transposed (inverse of
/// an orthonormal rotation is its transpose — no general inverse
/// needed). Shared by `look_at` in the scene graph and the orbit rig.
///
/// @param view The view matrix (rotation in m[0..10]).
/// @returns The world-orientation rotation matrix.
pub(crate) fn view_rot_to_world(view: &Mat4) -> Mat4 {
    let mut m = Mat4::IDENTITY.m;

    m[0] = view.m[0];
    m[1] = view.m[4];
    m[2] = view.m[8];
    m[4] = view.m[1];
    m[5] = view.m[5];
    m[6] = view.m[9];
    m[8] = view.m[2];
    m[9] = view.m[6];
    m[10] = view.m[10];

    Mat4 { m }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_places_eye_at_distance_and_faces_target() {
        let mut cam = Camera::new(45.0, 0.1, 100.0);

        cam.orbit = Some(Orbit {
            target: Vec3::new(0.0, 0.0, 0.0),
            distance: 10.0,
            yaw: 0.0,
            pitch: 0.0,
        });

        let v = cam.view();
        let (p, _) = v.transform_point(Vec3::new(0.0, 0.0, 0.0));

        // yaw=0, pitch=0 → eye sits at +Z by `distance`; the origin
        // lands at (0,0,−distance) in eye space.
        assert!((p.z + 10.0).abs() < 1e-4, "z={}", p.z);
        assert!(p.x.abs() < 1e-4);
        assert!(cam.position.z > 9.9);
    }

    #[test]
    fn orbit_pitch_raises_the_eye() {
        let mut cam = Camera::new(45.0, 0.1, 100.0);

        cam.orbit = Some(Orbit {
            target: Vec3::new(0.0, 0.0, 0.0),
            distance: 10.0,
            yaw: 0.0,
            pitch: std::f32::consts::FRAC_PI_2 * 0.5,
        });

        cam.apply_orbit();

        // 45° pitch → y ≈ d·sin(π/4) ≈ 7.07.
        assert!((cam.position.y - 7.07).abs() < 0.05, "y={}", cam.position.y);
    }

    #[test]
    fn view_is_the_inverse_of_world() {
        let mut cam = Camera::new(45.0, 0.1, 100.0);

        cam.position = Vec3::new(3.0, 4.0, 5.0);
        cam.rotation = Quat::from_euler(0.3, -0.6, 0.2);

        let v = cam.view();

        // The camera's own position maps to eye-space origin.
        let (p, _) = v.transform_point(cam.position);

        assert!(p.x.abs() < 1e-4 && p.y.abs() < 1e-4 && p.z.abs() < 1e-4);
    }
}
