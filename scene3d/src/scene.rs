//! Scene3d scene graph — the flat-indexed node hierarchy replacing
//! three.js's Object3D/Mesh/Group/Sprite/Points tree.
//!
//! # Why flat indices instead of pointer trees
//!
//! three.js nodes are heap objects linked by parent/children pointers —
//! every `updateMatrixWorld` walks those pointers, which is cache-hostile
//! AND allocates through JS. Here the graph is one `Vec<Node>`: a node
//! is a 64-ish-byte record found by index, children are index lists.
//! Benefits that matter at 5k nodes:
//! - `update_world` iterates a contiguous array in parent-before-child
//!   order (parents always get a SMALLER index — `add` appends), so one
//!   linear pass propagates every matrix with zero recursion and zero
//!   pointer chasing.
//! - Node identity is a `u32` — the wasm/JS boundary ships scalars, not
//!   object graphs. A JS handle is just the index; lookups are O(1).
//! - Removal is a flag, not a splice: children keep their indices and
//!   tombstoned nodes are skipped (reusing slots would invalidate live
//!   handles — the pool trades a few dead records for correctness).
//!
//! # What a node carries
//!
//! TRS (position Vec3, rotation Quat, scale Vec3) + cached `local`/
//! `world` Mat4 + hierarchy links + render flags (`visible`, `layers`,
//! `frustum_culled`, `render_order`) + `kind` (what to draw) +
//! `user_data` (u64 tag the JS side uses for its own ids — the engine's
//! `userData.bodyId` maps to it). The dirty-flag model: any TRS write
//! marks the node dirty; `update_world` recomputes ONLY dirty subtrees
//! (a static scene pays ~nothing per frame).

use crate::math::{Mat4, Quat, Vec3};

/// `NodeId` — a handle into `Scene.nodes`. Wraps u32 so node lookups
/// can't be confused with geometry/material/texture ids (all separate
/// registries, all u32 — bare u32s would happily cross-contaminate).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(pub u32);

/// `Kind` — what a node renders as. `Group` draws nothing (transform
/// node only — three.js Group/Object3D). The payloads hold REGISTRY
/// ids, not data — geometries/materials/textures live in their own
/// tables on the renderer so a mesh and a sprite can share one.
#[derive(Clone, Copy, Debug, Default)]
pub enum Kind {
    /// Empty transform node — three.js Object3D/Group.
    #[default]
    Group,
    /// Indexed triangle mesh — geometry id + material id.
    Mesh { geo: u32, mat: u32 },
    /// Point cloud — geometry id + material id (drawn as point list).
    Points { geo: u32, mat: u32 },
    /// Camera-facing quad — material id only (geometry is the shared
    /// billboard quad the renderer owns).
    Sprite { mat: u32 },
    /// Instanced mesh — geometry + material + instance capacity; the
    /// per-instance matrices live in the node's `instances` buffer.
    Instanced { geo: u32, mat: u32, capacity: u32 },
    /// Level-of-detail switch — picks one of `children` by distance;
    /// the renderer tests each child's `lod_range` (near, far).
    Lod,
}

/// `Node` — one scene-graph record. `local`/`world` are cached
/// results, NOT authoritative state — the authoritative transform is
/// TRS; `update_world` rebuilds `local` from TRS whenever `dirty` is
/// set and composes `world = parent_world · local`. Reading `world`
/// before `update_world` yields stale data — every public accessor
/// runs the update first.
#[derive(Clone, Debug)]
pub struct Node {
    /// Parent index, `None` for the scene root. Roots are the direct
    /// `Scene::root` children — `parent` tracks the OWNER of a node,
    /// `Scene::root` is the implicit parent of parentless nodes.
    pub parent: Option<NodeId>,
    /// Child indices in insertion order (render order for groups with
    /// identical `render_order`).
    pub children: Vec<NodeId>,
    /// Object-space position (translation in parent space).
    pub position: Vec3,
    /// Rotation — authoritative orientation (Euler writes convert
    /// through `Quat::from_euler`).
    pub rotation: Quat,
    /// Per-axis scale. Keep every axis > ε (see Mat4::scale).
    pub scale: Vec3,
    /// Cached T·R·S — rebuilt by `update_world` when `dirty`.
    pub local: Mat4,
    /// Cached parent·local — valid after `update_world`.
    pub world: Mat4,
    /// TRS stale flag — set by every transform write, cleared by
    /// `update_world` for the node AND its whole subtree (child worlds
    /// depend on this node's `local`, so dirty always cascades down).
    dirty: bool,
    /// `visible` — render gate. Hidden subtrees are skipped entirely
    /// (children of a hidden node inherit — three.js semantics).
    pub visible: bool,
    /// `layers` — 32-bit camera-mask channel (three.js Layers): a node
    /// draws only when `node.layers & camera.layers != 0`.
    pub layers: u32,
    /// `frustum_culled` — per-node opt-out of sphere-vs-frustum tests
    /// (sprites/points whose bounds are per-frame dynamic).
    pub frustum_culled: bool,
    /// `render_order` — stable sort key within the draw list (higher
    /// draws later — three.js renderOrder semantics for overlays).
    pub render_order: i32,
    /// What to draw (Group for pure-transform nodes).
    pub kind: Kind,
    /// Engine tag — opaque u64 the JS side uses for its own ids
    /// (body ids, tile keys). The graph never interprets it.
    pub user_data: u64,
    /// LOD window for children of a `Lod` node: `lod_range = (near,
    /// far)` — the child draws when camera distance lands inside.
    pub lod_range: (f32, f32),
    /// Per-instance world matrices for `Kind::Instanced` — uploaded to
    /// the instance buffer each frame the node is dirty.
    pub instances: Vec<Mat4>,
    /// Instance-data version — bumped by `set_instance_data`; the
    /// renderer re-uploads the buffer only when the stamp differs
    /// from the last upload (per-frame rewrites of unchanged instance
    /// tables are the usual perf trap).
    pub inst_version: u64,
}

impl Node {
    /// Blank group node — identity transform, visible, all-layers.
    fn blank() -> Node {
        Node {
            parent: None,
            children: Vec::new(),
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(1.0, 1.0, 1.0),
            local: Mat4::IDENTITY,
            world: Mat4::IDENTITY,
            dirty: true,
            visible: true,
            layers: 1,
            frustum_culled: true,
            render_order: 0,
            kind: Kind::Group,
            user_data: 0,
            lod_range: (0.0, f32::MAX),
            instances: Vec::new(),
            inst_version: 0,
        }
    }
}

/// `Scene` — the node arena + the implicit root. `nodes[0]` is
/// RESERVED as the scene root itself (a Group node JS never sees —
/// every `add` without a parent attaches here), which keeps
/// `update_world` a single uniform walk with no special-casing.
#[derive(Debug, Default)]
pub struct Scene {
    /// The node arena. Index 0 is the implicit root.
    pub nodes: Vec<Node>,
    /// Draw-order scratch — `collect` fills it each frame; reused to
    /// stay allocation-free in the steady state (rule 27).
    draw_list: Vec<NodeId>,
}

impl Scene {
    /// `Scene::new` — allocates the arena with the reserved root at 0.
    ///
    /// @returns A scene whose only node is the implicit root group.
    pub fn new() -> Scene {
        let mut root = Node::blank();

        root.dirty = false; // The root never moves — skip its recompute.

        Scene {
            nodes: vec![root],
            draw_list: Vec::new(),
        }
    }

    /// `add` — append a node under `parent` (or the implicit root when
    /// `None`). Returns the new handle — indices are stable for the
    /// scene's lifetime (no slot reuse; see module doc).
    ///
    /// @param parent Owning node, None → scene root.
    /// @param kind What the node renders as.
    /// @returns The new node's id.
    pub fn add(&mut self, parent: Option<NodeId>, kind: Kind) -> NodeId {
        let parent = parent.unwrap_or(NodeId(0));
        let id = NodeId(self.nodes.len() as u32);
        let mut node = Node::blank();

        node.parent = Some(parent);
        node.kind = kind;

        self.nodes.push(node);
        self.nodes[parent.0 as usize].children.push(id);

        id
    }

    /// `remove` — detach a node from its parent (tombstone — the node
    /// record stays so live handles elsewhere don't alias; the slot is
    /// skipped by traversal/draw).
    ///
    /// @param id Node to detach — children are detached with it.
    pub fn remove(&mut self, id: NodeId) {
        // Detach children first so they don't dangle inside a
        // disconnected subtree — each becomes parentless (root-owned).
        let children = std::mem::take(&mut self.nodes[id.0 as usize].children);

        for child in children {
            self.nodes[child.0 as usize].parent = None;
            self.nodes[0].children.push(child);
        }

        if let Some(p) = self.nodes[id.0 as usize].parent.take() {
            self.nodes[p.0 as usize].children.retain(|&c| c != id);
        }

        // Tombstone: invisible + nonexistent kind so every query skips.
        self.nodes[id.0 as usize].visible = false;
        self.nodes[id.0 as usize].kind = Kind::Group;
    }

    /// `set_position` / `set_euler` / `set_scale` — the three.js-style
    /// transform setters. Each marks the node dirty; `update_world`
    /// recomputes on the next traversal (callers never pay per-write).
    ///
    /// @param id Node to move. @param v New position/euler/scale.
    pub fn set_position(&mut self, id: NodeId, v: Vec3) {
        self.nodes[id.0 as usize].position = v;
        self.nodes[id.0 as usize].dirty = true;
    }

    /// Sets rotation from XYZ-order Euler radians (three.js
    /// `rotation.set(x, y, z)` parity — see `Quat::from_euler`).
    ///
    /// @param id Node to rotate. @param x @param y @param z Radians.
    pub fn set_euler(&mut self, id: NodeId, x: f32, y: f32, z: f32) {
        self.nodes[id.0 as usize].rotation = Quat::from_euler(x, y, z);
        self.nodes[id.0 as usize].dirty = true;
    }

    /// Sets per-axis scale.
    ///
    /// @param id Node to scale. @param v Scale factors.
    pub fn set_scale(&mut self, id: NodeId, v: Vec3) {
        self.nodes[id.0 as usize].scale = v;
        self.nodes[id.0 as usize].dirty = true;
    }

    /// `look_at` — orient a node so its −Z axis points at `target`
    /// (three.js `Object3D.lookAt` semantics — cameras and most scene
    /// objects face −Z toward the point). Implemented by building the
    /// same basis `Mat4::look_at` produces, stripping translation, and
    /// converting to a quaternion — one code path for both views.
    ///
    /// @param id Node to orient (its current position is the eye).
    /// @param target World-space point to face.
    /// @param up World up reference (0,1,0 in the scenes).
    pub fn look_at(&mut self, id: NodeId, target: Vec3, up: Vec3) {
        self.update_world();

        // The node's CURRENT world position — the translation column
        // of its (now fresh) world matrix.
        let w = &self.nodes[id.0 as usize].world;
        let wp = Vec3::new(w.m[12], w.m[13], w.m[14]);
        let view = Mat4::look_at(wp, target, up);

        // The basis rows of view⁻¹'s rotation — for an orthonormal
        // view matrix the world orientation is its transpose; extracting
        // the quaternion from the rotation block is cleaner than
        // inverting a full affine.
        let basis = {
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
        };

        self.nodes[id.0 as usize].rotation = Quat::from_mat4(&basis);
        self.nodes[id.0 as usize].dirty = true;
    }

    /// `update_world` — recompute every dirty local+world matrix in a
    /// single index-order pass. Parents always index lower than their
    /// children (nodes append; `remove` never shifts), so a simple
    /// ascending scan guarantees parents' `world` is final before a
    /// child's compose reads it — no recursion, no visited sets.
    ///
    /// Dirty propagation: a dirty parent marks its children dirty even
    /// when their own TRS is untouched (their `local` is clean but
    /// `world = parent·local` still changes). Cost when static: one
    /// scan of a flag word per node — ~5k loads ≈ 5µs.
    pub fn update_world(&mut self) {
        // Ascending index order = parent-before-child (see module doc).
        for i in 1..self.nodes.len() {
            let node = &self.nodes[i];

            if !node.dirty {
                // Parent-dirty propagation — world must recompute even
                // when this node's own TRS is clean.
                let p_dirty = node
                    .parent
                    .map(|p| self.nodes[p.0 as usize].dirty)
                    .unwrap_or(false);

                if !p_dirty {
                    continue;
                }
            }

            // Rebuild local from TRS, then compose against the parent.
            let (pos, rot, scl, parent) = {
                let n = &self.nodes[i];

                (n.position, n.rotation, n.scale, n.parent)
            };
            let local = Mat4::compose(pos, rot, scl);
            let parent_world = parent
                .map(|p| self.nodes[p.0 as usize].world)
                .unwrap_or(Mat4::IDENTITY);

            // Cascade BEFORE clearing — children always index higher
            // than their parent (nodes append; reparenting via remove
            // only ever re-homes to the root), so marking them dirty
            // here guarantees they recompute later in THIS pass even
            // though the parent's flag is cleared below.
            for k in 0..self.nodes[i].children.len() {
                let c = self.nodes[i].children[k].0 as usize;

                self.nodes[c].dirty = true;
            }

            let n = &mut self.nodes[i];

            n.local = local;
            n.world = parent_world.mul(local);
            n.dirty = false;
        }
    }

    /// `world_position` — the node's world-space translation column —
    /// three.js `getWorldPosition` (world.m[12..14]).
    ///
    /// @param id Node to query.
    /// @returns World-space position.
    pub fn world_position(&mut self, id: NodeId) -> Vec3 {
        self.update_world();

        let w = self.nodes[id.0 as usize].world;

        Vec3::new(w.m[12], w.m[13], w.m[14])
    }

    /// `set_instance` — writes one instance transform on a
    /// `Kind::Instanced` node and bumps `inst_version` so the renderer
    /// re-uploads the instance buffer once per edit batch, not once
    /// per frame.
    ///
    /// The table grows to `i+1` on first use — instance tables are
    /// filled at init then updated sparsely (star-field tweens touch
    /// a few rows per frame), never cleared.
    ///
    /// @param id The instanced node.
    /// @param i  Instance slot index.
    /// @param m  The instance's world-space transform.
    pub fn set_instance(&mut self, id: NodeId, i: u32, m: Mat4) {
        let n = &mut self.nodes[id.0 as usize];

        if n.instances.len() <= i as usize {
            n.instances.resize(i as usize + 1, Mat4::IDENTITY);
        }

        n.instances[i as usize] = m;
        n.inst_version = n.inst_version.wrapping_add(1);
    }

    /// `instance_count` — the draw-time instance total for an
    /// `Instanced` node (0 for other kinds).
    ///
    /// @param id Node to query.
    /// @returns Populated instance slots.
    pub fn instance_count(&self, id: NodeId) -> u32 {
        self.nodes[id.0 as usize].instances.len() as u32
    }

    /// `collect` — fill the reusable draw list: every VISIBLE node
    /// whose `layers` intersect `camera_layers`, in stable render
    /// order. The walk is a DFS down the index tree — children arrays
    /// are small (a body's LOD set, a group's shells), so the stack
    /// stays in the caller's scratch and recursion depth mirrors real
    /// hierarchy depth (≤8 in the scenes — bounded, no stack risk).
    ///
    /// `lod_nodes` returned separately: a `Kind::Lod` node contributes
    /// nothing itself — its children self-select by `lod_range` at
    /// draw time against camera distance.
    ///
    /// @param camera_layers The camera's 32-bit layer mask.
    /// @returns (draw-list slice, lod node ids) — both views into the
    ///          scene's reusable scratch buffers.
    pub fn collect(&mut self, camera_layers: u32) -> (&[NodeId], Vec<NodeId>) {
        self.draw_list.clear();

        let mut lod_nodes = Vec::new();

        // Recursive DFS — visibility/layer gates prune the whole
        // subtree at its root, so a hidden parent's children are never
        // visited (three.js semantics). Recursion depth mirrors the
        // real hierarchy (≤8 in the scenes) — bounded by construction.
        self.walk(NodeId(0), camera_layers, &mut lod_nodes);

        // Stable render-order sort — overlay groups (render_order>0)
        // draw after the base pass; equal keys keep insertion order.
        self.draw_list
            .sort_by_key(|&id| self.nodes[id.0 as usize].render_order);

        (&self.draw_list, lod_nodes)
    }

    /// `walk` — the recursive collector behind `collect`: emits the
    /// node into the draw list when drawable, then recurses through
    /// children. A node failing the visible/layer gates returns
    /// WITHOUT recursing — that's how the prune propagates.
    ///
    /// @param id Node being visited.
    /// @param camera_layers Camera layer mask for the gate.
    /// @param lod_nodes Accumulator for `Kind::Lod` handles.
    fn walk(&mut self, id: NodeId, camera_layers: u32, lod_nodes: &mut Vec<NodeId>) {
        let node = &self.nodes[id.0 as usize];

        if !node.visible || (node.layers & camera_layers) == 0 {
            return;
        }

        if matches!(node.kind, Kind::Lod) {
            lod_nodes.push(id);
        } else if !matches!(node.kind, Kind::Group) {
            self.draw_list.push(id);
        }

        // Index-based child walk — re-borrows `self` per child so the
        // &mut walk call doesn't alias a held borrow (children never
        // mutate mid-walk — structural edits happen outside collect).
        for k in 0..self.nodes[id.0 as usize].children.len() {
            let child = self.nodes[id.0 as usize].children[k];

            self.walk(child, camera_layers, lod_nodes);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_world_composes_parent() {
        let mut s = Scene::new();
        let parent = s.add(None, Kind::Group);
        let child = s.add(Some(parent), Kind::Mesh { geo: 0, mat: 0 });

        s.set_position(parent, Vec3::new(10.0, 0.0, 0.0));
        s.set_position(child, Vec3::new(0.0, 5.0, 0.0));

        let w = s.world_position(child);

        assert!((w.x - 10.0).abs() < 1e-5);
        assert!((w.y - 5.0).abs() < 1e-5);
    }

    #[test]
    fn dirty_parent_recomputes_clean_child() {
        let mut s = Scene::new();
        let parent = s.add(None, Kind::Group);
        let child = s.add(Some(parent), Kind::Group);

        s.update_world();
        s.set_position(parent, Vec3::new(7.0, 0.0, 0.0));

        let w = s.world_position(child);

        assert!((w.x - 7.0).abs() < 1e-5, "child didn't follow parent");
    }

    #[test]
    fn remove_tombstones_the_node_and_orphans_children() {
        let mut s = Scene::new();
        let parent = s.add(None, Kind::Group);
        let child = s.add(Some(parent), Kind::Mesh { geo: 0, mat: 0 });

        s.remove(parent);

        assert!(!s.nodes[parent.0 as usize].visible);
        assert_eq!(s.nodes[child.0 as usize].parent, None);
        assert!(s.nodes[0].children.contains(&child));
    }

    #[test]
    fn collect_respects_visibility_layers_and_order() {
        let mut s = Scene::new();
        let a = s.add(None, Kind::Mesh { geo: 0, mat: 0 });
        let b = s.add(None, Kind::Sprite { mat: 0 });
        let hidden = s.add(None, Kind::Mesh { geo: 0, mat: 0 });

        s.nodes[a.0 as usize].render_order = 5;
        s.nodes[b.0 as usize].render_order = -1;
        s.nodes[hidden.0 as usize].visible = false;

        let (draw, _lod) = s.collect(1);

        assert_eq!(draw, &[b, a]);
    }

    #[test]
    fn euler_setter_matches_matrix_rotation() {
        let mut s = Scene::new();
        let id = s.add(None, Kind::Group);

        s.set_euler(id, 0.0, std::f32::consts::FRAC_PI_2, 0.0);
        s.update_world();

        // +Z rotated 90° about +Y lands on +X.
        let (p, _) = s.nodes[id.0 as usize]
            .local
            .transform_point(Vec3::new(0.0, 0.0, 1.0));

        assert!((p.x - 1.0).abs() < 1e-5, "x={}", p.x);
    }
}
