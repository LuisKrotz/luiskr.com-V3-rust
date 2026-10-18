//! wasm.rs — the `#[wasm_bindgen]` surface JS actually calls.
//!
//! The module boundary mirrors the widget crates: one exported handle
//! (`Surface3d`) that owns Scene + Camera + Renderer, plus flat f32/u32
//! argument lists — no serde, no JsValue structs, every call is a
//! typed-array or scalar write straight into the engine. That keeps
//! the JS/Wasm boundary at ~64 bytes/call (the wasm-bindgen overhead
//! floor) and lets the JS bridge drive a whole frame with ~10 calls.
//!
//! # Method families
//!
//!   Lifecycle : mount → resize/set_size → frame → destroy
//!   Resources : gen_*/upload_geometry → geo_id
//!               add_material_*        → mat_id
//!               upload_texture        → tex_id
//!   Nodes     : add_* → node_id; set_position/set_euler/set_scale;
//!               look_at; set_visible/set_layers/set_render_order/
//!               set_user_data/set_lod_range; remove; set_instance
//!   Camera    : set_camera/set_fov/set_clip/set_orbit (None via
//!               clear_orbit)/camera_look_at/set_camera_layers
//!   Frame     : set_light/set_clear/set_post (state), frame(t,dpr)
//!   Picking   : ray_at → [ox,oy,oz,dx,dy,dz] (JS loops bodies —
//!               per-pick, not per-frame, so JS-side cost is fine)
//!
//! Node/geo/mat/tex ids are u32 registry indices — JS never sees a
//! Rust pointer; stale ids no-op inside the engine.

use crate::camera::{Camera, Orbit};
use crate::geometry as geo;
use crate::material::{Blending, Flags, Material, Record, Side};
use crate::math::{Mat4, Vec3};
use crate::math::Quat;
use crate::renderer::Renderer;
use crate::scene::{Kind, NodeId, Scene};
use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, ImageBitmap};

/// `Surface3d` — the exported scene handle: the engine state (scene,
/// camera, renderer) plus the per-frame scalar state JS pushes through
/// set_* calls (light, clear color, post amounts). Kept as one struct
/// so the JS handle has exactly ONE drop point (`destroy`).
#[wasm_bindgen]
pub struct Surface3d {
    /// The flat scene graph.
    scene: Scene,
    /// The camera (orbit rig lives inside it).
    cam: Camera,
    /// The WebGPU renderer (resource registries + pass builder).
    r: Renderer,
    /// Scene point light — (pos, rgb, intensity) pushed once per
    /// state change, not per frame.
    light: (Vec3, Vec3, f32),
    /// Clear color (usually transparent black — the canvas
    /// composites over the page background).
    clear: [f64; 4],
    /// Post-chain amounts — (film, chroma, bloom); all-zero skips
    /// the post pass.
    post: [f32; 3],
}

/// `mount` — the async constructor (WebGPU init is a promise):
/// adapter → device → surface configure → renderer. Mirrors the
/// widget crates' `mount(canvas)` contract so the shared
/// `webgpu-widget.ts` bridge can drive it with the same mount-first,
/// fallback-on-failure flow.
///
/// @param canvas The host canvas element.
/// @param w @param h Backing size in DEVICE pixels.
/// @returns The scene handle, or a JS error string on init failure.
#[wasm_bindgen]
pub async fn mount(canvas: HtmlCanvasElement, w: u32, h: u32) -> Result<Surface3d, JsValue> {
    let surface = gpu_mini::init_surface(&canvas).await?;

    Ok(Surface3d {
        scene: Scene::new(),
        cam: Camera::new(45.0, 0.1, 4000.0),
        r: Renderer::new(surface, w, h),
        light: (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 1.0),
        clear: [0.0, 0.0, 0.0, 0.0],
        post: [0.0, 0.0, 0.0],
    })
}

#[wasm_bindgen]
impl Surface3d {
    /// `resize` — backing-store size in DEVICE px (DPR-aware): depth
    /// attachment + surface configure re-issue. Called from the JS
    /// ResizeObserver arm.
    ///
    /// @param w @param h Device-pixel size.
    pub fn resize(&mut self, w: u32, h: u32) {
        self.r.resize(w, h);
    }

    /// `set_size` — CSS-pixel size for layout-derived math (sprite
    /// attenuation, point sizes in CSS units). Separate channel from
    /// `resize` per the crate-wide contract: `set_size`/`res` = CSS,
    /// `resize`/`rect` = device.
    ///
    /// @param _w @param _h CSS-pixel size — folded into the camera
    ///        aspect at frame time via the device size; kept for
    ///        contract parity (future CSS-space effects).
    pub fn set_size(&mut self, _w: f32, _h: f32) {}

    // ── Geometry uploads ───────────────────────────────────────────
    // Each generator returns a geo_id — JS holds it on the wrapper
    // object it creates (the Geo handle). `as_points` expands the
    // vertex stream into billboard quads at upload (see renderer.rs).

    /// `gen_sphere` — unit generator passthrough (docs in geometry.rs).
    ///
    /// @param radius Sphere radius.
    /// @param ws @param hs Width/height segments.
    /// @param as_points True → billboard-quad expansion (Points).
    /// @returns The geometry registry id.
    pub fn gen_sphere(&mut self, radius: f32, ws: u32, hs: u32, as_points: bool) -> u32 {
        self.r.upload_geometry(&geo::sphere(radius, ws, hs), as_points)
    }

    /// `gen_plane` — XY plane generator passthrough.
    ///
    /// @param w @param h Plane extents.
    /// @param as_points Points-expansion flag.
    /// @returns The geometry registry id.
    pub fn gen_plane(&mut self, w: f32, h: f32, as_points: bool) -> u32 {
        self.r.upload_geometry(&geo::plane(w, h), as_points)
    }

    /// `gen_ring` — flat ring generator passthrough.
    ///
    /// @param inner @param outer Ring radii.
    /// @param segments Radial segment count.
    /// @param as_points Points-expansion flag.
    /// @returns The geometry registry id.
    pub fn gen_ring(&mut self, inner: f32, outer: f32, segments: u32, as_points: bool) -> u32 {
        self.r
            .upload_geometry(&geo::ring(inner, outer, segments), as_points)
    }

    /// `gen_box` — box generator passthrough.
    ///
    /// @param w @param h @param d Edge lengths.
    /// @param as_points Points-expansion flag.
    /// @returns The geometry registry id.
    pub fn gen_box(&mut self, w: f32, h: f32, d: f32, as_points: bool) -> u32 {
        self.r.upload_geometry(&geo::box_mesh(w, h, d), as_points)
    }

    /// `gen_cylinder` — cylinder generator passthrough.
    ///
    /// @param rt @param rb Top/bottom radii.
    /// @param h Height. @param segs Radial segments.
    /// @param as_points Points-expansion flag.
    /// @returns The geometry registry id.
    pub fn gen_cylinder(&mut self, rt: f32, rb: f32, h: f32, segs: u32, as_points: bool) -> u32 {
        self.r
            .upload_geometry(&geo::cylinder(rt, rb, h, segs), as_points)
    }

    /// `gen_torus` — torus generator passthrough.
    ///
    /// @param radius Ring radius. @param tube Tube radius.
    /// @param radial @param tubular Segment counts.
    /// @param as_points Points-expansion flag.
    /// @returns The geometry registry id.
    pub fn gen_torus(
        &mut self,
        radius: f32,
        tube: f32,
        radial: u32,
        tubular: u32,
        as_points: bool,
    ) -> u32 {
        self.r
            .upload_geometry(&geo::torus(radius, tube, radial, tubular), as_points)
    }

    /// `gen_icosahedron` — icosahedron generator passthrough.
    ///
    /// @param radius Circumscribed radius.
    /// @param detail Subdivision detail.
    /// @param as_points Points-expansion flag.
    /// @returns The geometry registry id.
    pub fn gen_icosahedron(&mut self, radius: f32, detail: u32, as_points: bool) -> u32 {
        self.r
            .upload_geometry(&geo::icosahedron(radius, detail), as_points)
    }

    /// `gen_tetrahedron` — tetrahedron generator passthrough.
    ///
    /// @param radius Circumscribed radius.
    /// @param as_points Points-expansion flag.
    /// @returns The geometry registry id.
    pub fn gen_tetrahedron(&mut self, radius: f32, as_points: bool) -> u32 {
        self.r
            .upload_geometry(&geo::tetrahedron(radius), as_points)
    }

    /// `upload_geometry` — arbitrary vertex data (the custom
    /// BufferGeometry path: star-point clouds, hand-built shells).
    /// Typed arrays in, geo_id out — zero JsValue marshalling.
    ///
    /// @param positions xyz triples (len = 3n).
    /// @param normals xyz triples (len = 3n — zeros allowed for
    ///        unlit materials).
    /// @param uvs uv pairs (len = 2n).
    /// @param indices Triangle indices (len = 3k, u32).
    /// @param as_points Points-expansion flag.
    /// @returns The geometry registry id.
    pub fn upload_geometry(
        &mut self,
        positions: &[f32],
        normals: &[f32],
        uvs: &[f32],
        indices: &[u32],
        as_points: bool,
    ) -> u32 {
        let g = geo::Geometry {
            positions: positions.to_vec(),
            normals: normals.to_vec(),
            uvs: uvs.to_vec(),
            indices: indices.to_vec(),
        };

        self.r.upload_geometry(&g, as_points)
    }

    /// `upload_texture` — register a decoded ImageBitmap (the
    /// GPU→GPU `copyExternalImageToTexture` path — no CPU pixel
    /// readback).
    ///
    /// @param bitmap The JS-decoded bitmap.
    /// @param w @param h Pixel size.
    /// @returns The texture registry id.
    pub fn upload_texture(&mut self, bitmap: &ImageBitmap, w: u32, h: u32) -> u32 {
        self.r.upload_texture(bitmap, w, h)
    }

    // ── Materials ──────────────────────────────────────────────────
    // One add_* per Material variant; `flags`/`emissive` apply to the
    // LAST-added material id (JS builds a material then tunes it —
    // mirrors the constructor+setters shape of the three.js materials
    // being replaced).

    /// `add_material_basic` — MeshBasicMaterial (unlit texel·color).
    ///
    /// @param r @param g @param b @param a Color rgba 0-1.
    /// @param tex Texture id or `u32::MAX` for none.
    /// @returns The material registry id.
    pub fn add_material_basic(&mut self, r: f32, g: f32, b: f32, a: f32, tex: u32) -> u32 {
        self.r.add_material(Record::new(Material::Basic {
            color: [r, g, b, a],
            texture: tex,
        }))
    }

    /// `add_material_standard` — MeshStandardMaterial (Blinn-Phong —
    /// see shader_standard.wgsl for the lighting model).
    ///
    /// @param r @param g @param b @param a Color rgba 0-1.
    /// @param roughness Surface roughness 0-1 (→ spec exponent).
    /// @param metalness Metallic factor 0-1 (→ spec tint).
    /// @param tex Texture id or `u32::MAX`.
    /// @returns The material registry id.
    pub fn add_material_standard(
        &mut self,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
        roughness: f32,
        metalness: f32,
        tex: u32,
    ) -> u32 {
        self.r.add_material(Record::new(Material::Standard {
            color: [r, g, b, a],
            roughness,
            metalness,
            texture: tex,
        }))
    }

    /// `add_material_sprite` — SpriteMaterial (billboard texel·color).
    ///
    /// @param r @param g @param b @param a Color rgba 0-1.
    /// @param tex Texture id or `u32::MAX`.
    /// @returns The material registry id.
    pub fn add_material_sprite(&mut self, r: f32, g: f32, b: f32, a: f32, tex: u32) -> u32 {
        self.r.add_material(Record::new(Material::Sprite {
            color: [r, g, b, a],
            texture: tex,
        }))
    }

    /// `add_material_points` — PointsMaterial (expanded-quad point
    /// sprites).
    ///
    /// @param r @param g @param b @param a Color rgba 0-1.
    /// @param size World-space point half-size.
    /// @param tex Texture id or `u32::MAX`.
    /// @returns The material registry id.
    pub fn add_material_points(
        &mut self,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
        size: f32,
        tex: u32,
    ) -> u32 {
        self.r.add_material(Record::new(Material::Points {
            color: [r, g, b, a],
            size,
            texture: tex,
        }))
    }

    /// `set_material_flags` — pipeline-state flags on a registered
    /// material. REBUILDS the material's bind group slot cheaply
    /// (flags live in the uniform pack — the bind group itself is
    /// texture-only and untouched).
    ///
    /// @param mat Material id.
    /// @param side 0=front 1=back 2=double (three.js *Side).
    /// @param blending 0=normal 1=additive.
    /// @param transparent Alpha-sort pass membership.
    /// @param depth_write Depth-buffer writes on/off.
    pub fn set_material_flags(
        &mut self,
        mat: u32,
        side: u32,
        blending: u32,
        transparent: bool,
        depth_write: bool,
    ) {
        self.r.set_flags(
            mat,
            Flags {
                side: match side {
                    1 => Side::Back,
                    2 => Side::Double,
                    _ => Side::Front,
                },
                blending: match blending {
                    1 => Blending::Additive,
                    _ => Blending::Normal,
                },
                transparent,
                depth_write,
            },
        );
    }

    /// `set_emissive` — the additive self-lit tint on a material.
    ///
    /// @param mat Material id.
    /// @param r @param g @param b @param a Emissive rgba.
    pub fn set_emissive(&mut self, mat: u32, r: f32, g: f32, b: f32, a: f32) {
        self.r.set_emissive(mat, [r, g, b, a]);
    }

    /// `set_color` — the base rgba on a registered material. Alpha is
    /// the opacity slot — the per-frame label-fade/highlight arms
    /// write this once per visible material (one uniform-pack row —
    /// cheaper than rebuilding the record).
    ///
    /// @param mat Material id.
    /// @param r @param g @param b @param a rgba 0-1.
    pub fn set_color(&mut self, mat: u32, r: f32, g: f32, b: f32, a: f32) {
        self.r.set_color(mat, [r, g, b, a]);
    }

    // ── Nodes ──────────────────────────────────────────────────────
    // `parent` = u32::MAX attaches to the scene root.

    /// `add_mesh` — Mesh(geo, mat).
    ///
    /// @param parent Parent node id or `u32::MAX` for root.
    /// @param geo Geometry id. @param mat Material id.
    /// @returns The new node id.
    pub fn add_mesh(&mut self, parent: u32, geo: u32, mat: u32) -> u32 {
        self.scene
            .add(to_parent(parent), Kind::Mesh { geo, mat })
            .0
    }

    /// `add_points` — Points(geo, mat).
    ///
    /// @param parent Parent id or `u32::MAX`.
    /// @param geo Geometry id. @param mat Material id.
    /// @returns The new node id.
    pub fn add_points(&mut self, parent: u32, geo: u32, mat: u32) -> u32 {
        self.scene
            .add(to_parent(parent), Kind::Points { geo, mat })
            .0
    }

    /// `add_sprite` — Sprite(mat) — geometry is the shared billboard
    /// quad inside the renderer.
    ///
    /// @param parent Parent id or `u32::MAX`.
    /// @param mat Material id.
    /// @returns The new node id.
    pub fn add_sprite(&mut self, parent: u32, mat: u32) -> u32 {
        self.scene.add(to_parent(parent), Kind::Sprite { mat }).0
    }

    /// `add_instanced` — InstancedMesh(geo, mat).
    ///
    /// @param parent Parent id or `u32::MAX`.
    /// @param geo Geometry id. @param mat Material id.
    /// @param capacity Reserved instance slots.
    /// @returns The new node id.
    pub fn add_instanced(&mut self, parent: u32, geo: u32, mat: u32, capacity: u32) -> u32 {
        self.scene
            .add(to_parent(parent), Kind::Instanced { geo, mat, capacity })
            .0
    }

    /// `add_group` — a transform-only node (three.js Group/Object3D).
    ///
    /// @param parent Parent id or `u32::MAX`.
    /// @returns The new node id.
    pub fn add_group(&mut self, parent: u32) -> u32 {
        self.scene.add(to_parent(parent), Kind::Group).0
    }

    /// `add_lod` — an LOD switch node (three.js LOD): children
    /// self-select via `lod_range` against camera distance.
    ///
    /// @param parent Parent id or `u32::MAX`.
    /// @returns The new node id.
    pub fn add_lod(&mut self, parent: u32) -> u32 {
        self.scene.add(to_parent(parent), Kind::Lod).0
    }

    /// `set_lod_range` — a child's distance window under a Lod node:
    /// the child draws when camera distance ∈ [near, far).
    ///
    /// @param node Child node id.
    /// @param near @param far Distance window.
    pub fn set_lod_range(&mut self, node: u32, near: f32, far: f32) {
        if let Some(n) = self.scene.nodes.get_mut(node as usize) {
            n.lod_range = (near, far);
        }
    }

    /// `remove` — detach + drop a node (three.js remove/orphan
    /// semantics — children re-parent to the root).
    ///
    /// @param node Node id.
    pub fn remove(&mut self, node: u32) {
        self.scene.remove(NodeId(node));
    }

    /// `set_position` — object-space translation (three.js
    /// position.set — marks the node dirty for the next world pass).
    ///
    /// @param node Node id. @param x @param y @param z Position.
    pub fn set_position(&mut self, node: u32, x: f32, y: f32, z: f32) {
        self.scene.set_position(NodeId(node), Vec3::new(x, y, z));
    }

    /// `set_euler` — object-space Euler rotation, XYZ radians
    /// (three.js rotation.set — converts to the internal quaternion).
    ///
    /// @param node Node id. @param x @param y @param z Euler radians.
    pub fn set_euler(&mut self, node: u32, x: f32, y: f32, z: f32) {
        self.scene.set_euler(NodeId(node), x, y, z);
    }

    /// `set_scale` — object-space per-axis scale.
    ///
    /// @param node Node id. @param x @param y @param z Scale factors.
    pub fn set_scale(&mut self, node: u32, x: f32, y: f32, z: f32) {
        self.scene.set_scale(NodeId(node), Vec3::new(x, y, z));
    }

    /// `node_look_at` — orient a node toward a world point (three.js
    /// Object3D.lookAt — billboard moons, camera-chasing shells).
    ///
    /// @param node Node id. @param x @param y @param z World target.
    pub fn node_look_at(&mut self, node: u32, x: f32, y: f32, z: f32) {
        self.scene
            .look_at(NodeId(node), Vec3::new(x, y, z), Vec3::new(0.0, 1.0, 0.0));
    }

    /// `set_visible` — the visibility gate (hidden subtrees skip the
    /// whole draw walk, children included).
    ///
    /// @param node Node id. @param visible Gate flag.
    pub fn set_visible(&mut self, node: u32, visible: bool) {
        if let Some(n) = self.scene.nodes.get_mut(node as usize) {
            n.visible = visible;
        }
    }

    /// `set_layers` — the 32-bit layer mask (three.js Layers: node
    /// draws only when `layers & camera.layers != 0`).
    ///
    /// @param node Node id. @param mask Layer bitmask.
    pub fn set_layers(&mut self, node: u32, mask: u32) {
        if let Some(n) = self.scene.nodes.get_mut(node as usize) {
            n.layers = mask;
        }
    }

    /// `set_render_order` — the stable within-pass sort key (three.js
    /// renderOrder — higher draws later).
    ///
    /// @param node Node id. @param order Sort key.
    pub fn set_render_order(&mut self, node: u32, order: i32) {
        if let Some(n) = self.scene.nodes.get_mut(node as usize) {
            n.render_order = order;
        }
    }

    /// `set_user_data` — the opaque engine tag (body ids, tile keys —
    /// the graph never interprets it; JS round-trips it for picking).
    ///
    /// @param node Node id. @param data The tag.
    pub fn set_user_data(&mut self, node: u32, data: f64) {
        if let Some(n) = self.scene.nodes.get_mut(node as usize) {
            n.user_data = data as u64;
        }
    }

    /// `user_data` — read back the engine tag.
    ///
    /// @param node Node id.
    /// @returns The tag as f64 (exact for u53 ids).
    pub fn user_data(&self, node: u32) -> f64 {
        self.scene
            .nodes
            .get(node as usize)
            .map(|n| n.user_data as f64)
            .unwrap_or(0.0)
    }

    /// `set_instance` — write one instance transform on an Instanced
    /// node. Takes the 16 column-major f32s as a typed array — the
    /// same layout `Mat4.m` uses, no repacking.
    ///
    /// @param node Instanced node id.
    /// @param i Instance slot.
    /// @param m The 16-f32 matrix.
    pub fn set_instance(&mut self, node: u32, i: u32, m: &[f32]) {
        if m.len() < 16 {
            return;
        }

        let mut arr = [0.0f32; 16];

        arr.copy_from_slice(&m[0..16]);
        self.scene.set_instance(NodeId(node), i, Mat4 { m: arr });
    }

    /// `world_position` — a node's world-space translation
    /// (three.js getWorldPosition — engine-side pick helpers use it).
    ///
    /// @param node Node id.
    /// @returns [x, y, z] — allocated per call; pick paths only,
    ///          never per-frame.
    pub fn world_position(&mut self, node: u32) -> Vec<f32> {
        let p = self.scene.world_position(NodeId(node));

        vec![p.x, p.y, p.z]
    }

    // ── Camera ─────────────────────────────────────────────────────

    /// `set_camera` — direct TRS camera (orbit disabled). For orbiting
    /// scenes use `set_orbit` instead — the two modes are exclusive
    /// (orbit derives position/rotation each frame).
    ///
    /// @param x @param y @param z Eye position.
    pub fn set_camera(&mut self, x: f32, y: f32, z: f32) {
        self.cam.orbit = None;
        self.cam.position = Vec3::new(x, y, z);
    }

    /// `camera_look_at` — point the direct camera at a world target.
    ///
    /// @param x @param y @param z World target.
    pub fn camera_look_at(&mut self, x: f32, y: f32, z: f32) {
        let pos = self.cam.position;
        let target = Vec3::new(x, y, z);
        let view = Mat4::look_at(pos, target, Vec3::new(0.0, 1.0, 0.0));
        let world_rot = crate::camera::view_rot_to_world(&view);

        self.cam.rotation = Quat::from_mat4(&world_rot);
    }

    /// `set_orbit` — the spherical rig: eye = target + R(yaw,pitch)·
    /// (0,0,distance), facing the target (see camera.rs Orbit).
    ///
    /// @param tx @param ty @param tz Orbit target.
    /// @param distance Eye distance.
    /// @param yaw Rotation about +Y, radians.
    /// @param pitch Elevation, radians (clamped inside).
    pub fn set_orbit(&mut self, tx: f32, ty: f32, tz: f32, distance: f32, yaw: f32, pitch: f32) {
        self.cam.orbit = Some(Orbit {
            target: Vec3::new(tx, ty, tz),
            distance,
            yaw,
            pitch,
        });
    }

    /// `set_fov` — vertical field of view in DEGREES (three.js
    /// parity — stored as radians inside).
    ///
    /// @param deg Field of view.
    pub fn set_fov(&mut self, deg: f32) {
        self.cam.fov_y = deg.to_radians();
    }

    /// `set_clip` — near/far planes (see camera.rs for the depth-
    /// precision note on near).
    ///
    /// @param near Near clip. @param far Far clip.
    pub fn set_clip(&mut self, near: f32, far: f32) {
        self.cam.near = near;
        self.cam.far = far;
    }

    /// `set_camera_layers` — the camera's layer mask (only nodes
    /// whose `layers` intersect it draw).
    ///
    /// @param mask Layer bitmask.
    pub fn set_camera_layers(&mut self, mask: u32) {
        self.cam.layers = mask;
    }

    // ── Frame state + draw ─────────────────────────────────────────

    /// `set_light` — the scene's single point light.
    ///
    /// @param x @param y @param z Light position.
    /// @param r @param g @param b Light color rgb 0-1.
    /// @param intensity Scalar multiplier.
    pub fn set_light(&mut self, x: f32, y: f32, z: f32, r: f32, g: f32, b: f32, intensity: f32) {
        self.light = (Vec3::new(x, y, z), Vec3::new(r, g, b), intensity);
    }

    /// `set_clear` — the frame clear color (transparent black for
    /// page-composited canvases).
    ///
    /// @param r @param g @param b @param a Clear rgba 0-1.
    pub fn set_clear(&mut self, r: f64, g: f64, b: f64, a: f64) {
        self.clear = [r, g, b, a];
    }

    /// `set_post` — post-chain amounts; all-zero skips the pass.
    ///
    /// @param film Film grain 0-1.
    /// @param chroma Chromatic aberration 0-1.
    /// @param bloom Bloom strength 0-1.
    pub fn set_post(&mut self, film: f32, chroma: f32, bloom: f32) {
        self.post = [film, chroma, bloom];
    }

    /// `frame` — one full draw: refresh worlds → write camera block →
    /// draw list → submit. The JS loop calls this once per rAF
    /// (MANUAL drive — the engine never self-schedules).
    ///
    /// @param time Elapsed seconds (shader clocks).
    /// @param dpr Device pixel ratio (point-size compensation).
    pub fn frame(&mut self, time: f32, dpr: f32) {
        self.r
            .frame(&mut self.scene, &mut self.cam, time, dpr, self.light, self.clear, self.post);
    }

    /// `ray_at` — screen-space picking ray: NDC pixel → world-space
    /// (origin, direction) — JS loops bodies with it (sphere tests
    /// per body, per-pick only).
    ///
    /// `inv_view` is rebuilt from the camera's rotation + position —
    /// the view matrix's inverse IS the camera world transform (no
    /// general 4×4 inverse needed for a rigid transform).
    ///
    /// @param sx @param sy Screen position in CSS px.
    /// @param sw @param sh Canvas size in CSS px.
    /// @returns [ox, oy, oz, dx, dy, dz] — origin + normalized dir.
    pub fn ray_at(&mut self, sx: f32, sy: f32, sw: f32, sh: f32) -> Vec<f32> {
        let ndc_x = (sx / sw.max(1.0)) * 2.0 - 1.0;
        let ndc_y = 1.0 - (sy / sh.max(1.0)) * 2.0;
        let aspect = sw / sh.max(1.0);
        let view = self.cam.view();
        let mut inv_view = crate::camera::view_rot_to_world(&view);

        // Append the translation column — rot⁻¹ · t⁻¹ = world mat.
        inv_view.m[12] = self.cam.position.x;
        inv_view.m[13] = self.cam.position.y;
        inv_view.m[14] = self.cam.position.z;

        let ray = crate::math::ray_from_screen(
            ndc_x,
            ndc_y,
            inv_view,
            (self.cam.fov_y * 0.5).tan(),
            aspect,
            self.cam.position,
        );

        vec![ray.origin.x, ray.origin.y, ray.origin.z, ray.dir.x, ray.dir.y, ray.dir.z]
    }

    /// `destroy` — release the surface (GpuDevice drops with the
    /// struct; the canvas detaches when JS removes the element).
    pub fn destroy(self) {}
}

/// `to_parent` — the u32::MAX sentinel → `None` (scene root).
///
/// @param id The JS-side parent id.
/// @returns `Some(NodeId)` or `None` for root attachment.
fn to_parent(id: u32) -> Option<NodeId> {
    if id == u32::MAX { None } else { Some(NodeId(id)) }
}
