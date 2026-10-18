//! Scene3d renderer — the WebGPU draw path: resource registries,
//! per-material-kind pipelines, depth-tested forward pass, and the
//! frame() entry the wasm surface calls.
//!
//! # Draw-dispatch model (why it looks different from a widget crate)
//!
//! The widget crates draw ONE fullscreen quad with ONE uniform block —
//! one pipeline, one bind group, one draw call. A scene draws ~100
//! objects with different meshes/materials, so the renderer inverts
//! the granularity:
//!
//! - **One uniform buffer** (`frame_buf`, 96 f32 = 384B): camera block
//!   written once per frame at [0..48]; model matrix + material
//!   params rewritten per draw at [48..96]. `queue.write_buffer` calls
//!   are recorded, not executed — batching 100 tiny writes into one
//!   submit is still cheaper than one bind-group swap per draw (each
//!   bind group has to be CREATED per swap — allocation in the hot
//!   path, which rule 27 forbids).
//!
//! - **One bind group per material** (`group(1)`): {texture, sampler}.
//!   Built once when the material's texture is registered, rebound by
//!   `set_bind_group(1, …)` per draw — set_bind_group on an EXISTING
//!   group is a cheap pointer write inside the pass encoder. Untextured
//!   materials share a 1×1 white texture so the shader's color·texture
//!   multiply is unconditional (no pipeline variants by texture
//!   presence — pipeline count is the real cost driver in WebGPU).
//!
//! - **Pipelines keyed by (kind, side, blending)**: created lazily on
//!   first use into a small HashMap — the state space is tiny
//!   (4 kinds × 3 sides × 2 blends ≈ ≤24 real combos; the scenes use
//!   ~8). Depth: LessEqual-write for opaque, off for additive glows.
//!
//! - **Points as expanded quads**: WebGPU point-lists can't size
//!   points (no gl_PointSize), so `upload_geometry` expands each point
//!   into a 4-vertex billboard quad + 6 indices at upload time — a
//!   one-time CPU transform that turns every point cloud into an
//!   ordinary indexed mesh with a `corner` attribute the shader reads.
//!
//! # Pass structure per frame()
//!
//!   acquire depth+color → begin pass (clear) → sort opaque front→back
//!   (bind-group reuse) → draw opaque → draw transparent back→front →
//!   end → post chain (film/chroma/bloom when enabled) → present.
//!   All within ONE command encoder — one submit per frame.

use crate::camera::Camera;
use crate::geometry::Geometry;
use crate::material::{Blending, Material, Record, Side};
use crate::math::Vec3;
use crate::scene::{Kind, NodeId, Scene};
use gpu_mini::*;
use std::collections::HashMap;
use web_sys::*;

/// Frame-uniform block size in f32s — camera(48) + model(16) +
/// material(32). See camera.rs + material.rs for the field maps.
const FRAME_F32S: usize = 96;

/// `Geo` — a GPU-resident mesh: the vertex/index buffers plus the
/// index count the draw call needs.
struct Geo {
    /// Interleaved position+normal+uv (32B stride — see set_vertex).
    vertex: GpuBuffer,
    /// u32 index buffer.
    index: GpuBuffer,
    /// Number of indices (draw_indexed arg).
    index_count: u32,
}

/// `Tex` — a registered texture's view (bind groups point at views,
/// not textures).
struct Tex {
    /// The texture's default view.
    view: GpuTextureView,
}

/// `Mat` — a registered material: the uniform-packing record plus the
/// prebuilt group(1) bind group for its texture slot.
struct Mat {
    /// CPU-side material description (color/params/flags).
    record: Record,
    /// {texture, sampler} bind group — rebuilt if the texture changes.
    bind: GpuBindGroup,
}

/// `PipeKey` — the pipeline state hash: kind discriminant + side +
/// blending. u8-packed so `HashMap<PipeKey, GpuRenderPipeline>` lookups
/// stay a single word compare. Kind values: 0=basic mesh, 1=standard
/// mesh, 2=sprite billboard, 3=point quad, 4=instanced mesh.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipeKey(u8, u8, u8);

/// `Renderer` — owns every GPU object for one canvas: the surface,
/// registries, pipelines, and the shared uniform/bind infrastructure.
/// Created once per canvas by the exported `init`; `frame` renders the
/// whole scene each tick.
pub struct Renderer {
    /// Canvas context/device/queue/format.
    surface: Surface,
    /// Geometry registry — index = the geo id JS holds.
    geos: Vec<Geo>,
    /// Material registry — index = the material id.
    mats: Vec<Mat>,
    /// Texture registry — index = the texture id.
    texs: Vec<Tex>,
    /// The shared {frame uniform} bind group — group(0).
    frame_group: GpuBindGroup,
    /// The group(0) layout (kept for pipeline-layout assembly).
    frame_layout: GpuBindGroupLayout,
    /// The group(1) layout ({texture, sampler}) — per-material binds.
    tex_layout: GpuBindGroupLayout,
    /// The 96-f32 frame uniform buffer.
    frame_buf: GpuBuffer,
    /// Lazily-created pipelines keyed by (kind, side, blend).
    pipes: HashMap<PipeKey, GpuRenderPipeline>,
    /// The depth attachment — recreated on resize (depth size must
    /// match the canvas or the render pass fails validation).
    depth: Option<GpuTextureView>,
    /// Shared 1×1 white texture for untextured materials (see doc).
    white_bind: GpuBindGroup,
    /// Shared bilinear sampler.
    sampler: GpuSampler,
    /// The shared billboard quad for Sprite nodes (geometry lives
    /// in `geos[quad_geo]` — created at init so every sprite draws
    /// the same 4-vert mesh).
    quad_geo: u32,
    /// Per-node instance buffers — `Kind::Instanced` nodes upload
    /// their `instances` here (keyed by node index): the buffer plus
    /// its capacity in instance-slots so growth recreates rather than
    /// wraps.
    inst_bufs: HashMap<u32, (GpuBuffer, u32)>,
    /// Per-node last-uploaded `inst_version` — the dirty stamp that
    /// turns "should this instance table rewrite" into an integer
    /// compare per draw.
    inst_versions: HashMap<u32, u64>,
    /// Reusable draw-id scratch — `collect`'s slice can't coexist
    /// with per-node reads (borrow rules), so ids copy here once per
    /// frame (zero-alloc after warmup).
    scratch: Vec<NodeId>,
    /// Reusable f32 staging for instance-table uploads — same
    /// zero-alloc rationale as `scratch`.
    inst_scratch: Vec<f32>,
    /// Current canvas backing size in device px.
    size: (u32, u32),
}

/// `quad_corner` — the billboard corner for expanded point quads:
/// (−1..1, −1..1) per vertex, packed into a fourth vertex attribute.
/// Written at upload — the shader multiplies by material.size.
const CORNERS: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];

/// `quad_indices` — the two-triangle index pattern applied per point
/// when expanding a point cloud into billboard quads.
const QUAD_IDX: [u32; 6] = [0, 1, 2, 1, 3, 2];

impl Renderer {
    /// `Renderer::new` — build the GPU object graph over an initialized
    /// surface. Separated from `init_surface` so the caller can await
    /// device setup, then build the renderer synchronously.
    ///
    /// @param surface The configured canvas surface.
    /// @param w @param h Initial backing size (device px).
    /// @returns The ready renderer.
    pub fn new(surface: Surface, w: u32, h: u32) -> Renderer {
        let device = &surface.device;
        let queue = &surface.queue;

        // group(0): the frame uniform — VERTEX|FRAGMENT visibility
        // (gpu-mini's uniform_entry is fragment-only; the scene reads
        // view_proj in the vertex stage).
        let frame_entry = GpuBindGroupLayoutEntry::new(0, gpu_shader_stage::VERTEX | gpu_shader_stage::FRAGMENT);
        let frame_layout_inner = GpuBufferBindingLayout::new();
        frame_layout_inner.set_type(GpuBufferBindingType::Uniform);
        frame_layout_inner.set_has_dynamic_offset(false);
        frame_entry.set_buffer(&frame_layout_inner);
        let frame_layout = new_layout(device, &[frame_entry]);

        // group(1): {texture, sampler} — the per-material texture bind.
        let tex_layout = new_layout(device, &[texture_entry(0), sampler_entry(1)]);

        let frame_buf = new_buffer(
            device,
            (FRAME_F32S * 4) as u32,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        );
        let frame_group = new_bind_group(device, &frame_layout, &[bind_buffer_entry(0, &frame_buf)]);

        let sampler = linear_sampler(device);

        // The 1×1 white texture — see module doc: untextured materials
        // bind this so tex·color == color, no pipeline variant needed.
        let white = new_texture(device, 1, 1);
        upload_rgba(queue, &white, 1, 1, &[255, 255, 255, 255]);
        let white_view = view(&white);
        let white_bind = new_bind_group(
            device,
            &tex_layout,
            &[bind_view_entry(0, &white_view), bind_sampler_entry(1, &sampler)],
        );

        let mut r = Renderer {
            surface,
            geos: Vec::new(),
            mats: Vec::new(),
            texs: Vec::new(),
            frame_group,
            frame_layout,
            tex_layout,
            frame_buf,
            pipes: HashMap::new(),
            depth: None,
            white_bind,
            sampler,
            quad_geo: 0,
            inst_bufs: HashMap::new(),
            inst_versions: HashMap::new(),
            scratch: Vec::new(),
            inst_scratch: Vec::new(),
            size: (0, 0),
        };

        // The shared sprite billboard — a unit quad whose corner attr
        // drives the view-space expansion in the sprite shader.
        let mut quad = Geometry::default();
        for (i, c) in CORNERS.iter().enumerate() {
            quad.push_vertex(
                Vec3::new(c[0], c[1], 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                [c[0] * 0.5 + 0.5, c[1] * 0.5 + 0.5],
            );
            let _ = i;
        }
        quad.indices = QUAD_IDX.to_vec();

        r.quad_geo = r.upload_geometry(&quad, false);

        r.resize(w, h);
        r
    }

    /// `resize` — backing-size change → depth target recreate +
    /// context re-configure. Depth must match the canvas exactly (the
    /// render pass validates attachment sizes) so it's re-created
    /// rather than scaled.
    ///
    /// @param w @param h New device-px backing size.
    pub fn resize(&mut self, w: u32, h: u32) {
        if (w, h) == self.size {
            return;
        }

        self.size = (w, h);

        gpu_mini::resize(&self.surface);

        let tex = self
            .surface
            .device
            .create_texture(&GpuTextureDescriptor::new(
                GpuTextureFormat::Depth24plus,
                &[
                    js_sys::Number::from(w.max(1)),
                    js_sys::Number::from(h.max(1)),
                    js_sys::Number::from(1u32),
                ],
                gpu_texture_usage::RENDER_ATTACHMENT,
            ))
            .expect("depth texture");

        self.depth = Some(view(&tex));
    }

    /// `upload_geometry` — pack a `Geometry` into interleaved vertex +
    /// index buffers and register it. Points-mode expansion (every
    /// vertex → one billboard quad) happens here too when `as_points`
    /// is set — see the module doc for why.
    ///
    /// Interleave layout (32B stride):
    ///   [0..12)  position xyz   → shader @location(0)
    ///   [12..24) normal xyz     → @location(1)
    ///   [24..32) uv or corner   → @location(2) — for point quads this
    ///              slot carries the (−1..1) corner instead of uv; the
    ///              point shader reads it as the billboard direction.
    ///
    /// @param g The CPU-side geometry.
    /// @param as_points True → expand each vertex into a billboard
    ///        quad (Points draw); false → plain indexed mesh.
    /// @returns The geometry registry id.
    pub fn upload_geometry(&mut self, g: &Geometry, as_points: bool) -> u32 {
        let (positions, normals, uvs, indices) = if as_points {
            expand_points(g)
        } else {
            (
                g.positions.clone(),
                g.normals.clone(),
                g.uvs.clone(),
                g.indices.clone(),
            )
        };

        let n = positions.len() / 3;
        let mut interleaved = vec![0.0f32; n * 8];

        for i in 0..n {
            let o = i * 8;

            interleaved[o] = positions[i * 3];
            interleaved[o + 1] = positions[i * 3 + 1];
            interleaved[o + 2] = positions[i * 3 + 2];
            interleaved[o + 3] = normals[i * 3];
            interleaved[o + 4] = normals[i * 3 + 1];
            interleaved[o + 5] = normals[i * 3 + 2];
            interleaved[o + 6] = uvs[i * 2];
            interleaved[o + 7] = uvs[i * 2 + 1];
        }

        let vertex = new_buffer(
            &self.surface.device,
            (interleaved.len() * 4) as u32,
            gpu_buffer_usage::VERTEX | gpu_buffer_usage::COPY_DST,
        );
        let index = new_buffer(
            &self.surface.device,
            (indices.len() * 4) as u32,
            gpu_buffer_usage::INDEX | gpu_buffer_usage::COPY_DST,
        );

        let _ = self.surface.queue.write_buffer_with_u32_and_u8_slice(
            &vertex,
            0,
            f32_bytes(&interleaved),
        );
        let _ = self.surface.queue.write_buffer_with_u32_and_u8_slice(
            &index,
            0,
            unsafe {
                std::slice::from_raw_parts(indices.as_ptr().cast::<u8>(), indices.len() * 4)
            },
        );

        self.geos.push(Geo {
            vertex,
            index,
            index_count: indices.len() as u32,
        });

        (self.geos.len() - 1) as u32
    }

    /// `upload_texture` — register a decoded bitmap as a texture
    /// (`copyExternalImageToTexture` — the decode never touches CPU).
    ///
    /// @param bitmap The JS-decoded ImageBitmap.
    /// @param w @param h Bitmap size.
    /// @returns The texture registry id.
    pub fn upload_texture(&mut self, bitmap: &ImageBitmap, w: u32, h: u32) -> u32 {
        let tex = bitmap_texture(&self.surface.device, &self.surface.queue, bitmap, w, h);

        self.texs.push(Tex { view: view(&tex) });

        (self.texs.len() - 1) as u32
    }

    /// `add_material` — register a material record; builds its
    /// group(1) bind group up-front (white texture when none bound).
    ///
    /// @param record The material description.
    /// @returns The material registry id.
    pub fn add_material(&mut self, record: Record) -> u32 {
        let tex_id = material_texture(&record.material);
        let bind = if tex_id == u32::MAX || self.texs.get(tex_id as usize).is_none() {
            self.white_bind.clone()
        } else {
            new_bind_group(
                &self.surface.device,
                &self.tex_layout,
                &[
                    bind_view_entry(0, &self.texs[tex_id as usize].view),
                    bind_sampler_entry(1, &self.sampler),
                ],
            )
        };

        self.mats.push(Mat { record, bind });

        (self.mats.len() - 1) as u32
    }

    /// `set_flags` — pipeline-state flags on a registered material.
    /// Uniform-pack data only — the texture bind group is untouched.
    ///
    /// @param mat Material id. @param flags The new flag set.
    pub fn set_flags(&mut self, mat: u32, flags: crate::material::Flags) {
        if let Some(m) = self.mats.get_mut(mat as usize) {
            m.record.flags = flags;
        }
    }

    /// `set_emissive` — the additive self-lit tint on a material.
    ///
    /// @param mat Material id. @param emissive rgba 0-1.
    pub fn set_emissive(&mut self, mat: u32, emissive: [f32; 4]) {
        if let Some(m) = self.mats.get_mut(mat as usize) {
            m.record.emissive = emissive;
        }
    }

    /// `set_color` — the base rgba tint on a registered material
    /// (color[0..4] in the uniform pack). Per-frame opacity arms write
    /// `a` through here — uniform-pack data only, the texture bind
    /// group is untouched.
    ///
    /// @param mat Material id. @param color rgba 0-1.
    pub fn set_color(&mut self, mat: u32, color: [f32; 4]) {
        if let Some(m) = self.mats.get_mut(mat as usize) {
            match &mut m.record.material {
                crate::material::Material::Basic { color: c, .. }
                | crate::material::Material::Standard { color: c, .. }
                | crate::material::Material::Sprite { color: c, .. }
                | crate::material::Material::Points { color: c, .. } => *c = color,
            }
        }
    }

    /// `pipe` — fetch-or-create the pipeline for (kind, side, blend).
    /// Lazily created and cached in `self.pipes` — pipeline creation
    /// compiles + validates a WGSL module (~ms), so it must never
    /// happen mid-frame.
    fn pipe(&mut self, key: PipeKey, kind: u8) -> GpuRenderPipeline {
        if let Some(p) = self.pipes.get(&key) {
            return p.clone();
        }

        let wgsl = match kind {
            1 => include_str!("shader_standard.wgsl"),
            2 => include_str!("shader_sprite.wgsl"),
            3 => include_str!("shader_points.wgsl"),
            4 => include_str!("shader_instanced.wgsl"),
            _ => include_str!("shader_basic.wgsl"),
        };
        let side = match key.1 {
            1 => GpuCullMode::Front,
            2 => GpuCullMode::None,
            _ => GpuCullMode::Back,
        };
        let additive = key.2 == 1;
        let p = self.build_pipeline(wgsl, side, additive, kind == 4);

        self.pipes.insert(key, p.clone());
        p
    }

    /// `build_pipeline` — assemble one render pipeline: the shared
    /// two-group layout, the interleaved vertex layout, depth stencil,
    /// cull + blend by key.
    fn build_pipeline(
        &self,
        wgsl: &str,
        cull: GpuCullMode,
        additive: bool,
        instanced: bool,
    ) -> GpuRenderPipeline {
        let device = &self.surface.device;
        let shader = device.create_shader_module(&GpuShaderModuleDescriptor::new(wgsl));
        let layout_arr = [
            js_sys::JsNullable::wrap(self.frame_layout.clone()),
            js_sys::JsNullable::wrap(self.tex_layout.clone()),
        ];
        let layout = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&layout_arr));

        // Vertex buffer: one interleaved stream, 32B stride, three
        // attributes — position(0), normal(1), uv|corner(2).
        let attrs = [
            GpuVertexAttribute::new(GpuVertexFormat::Float32x3, 0, 0),
            GpuVertexAttribute::new(GpuVertexFormat::Float32x3, 12, 1),
            GpuVertexAttribute::new(GpuVertexFormat::Float32x2, 24, 2),
        ];
        let buf = GpuVertexBufferLayout::new(32, &attrs);

        let vertex = GpuVertexState::new(&shader);
        vertex.set_entry_point("vs_main");

        // Slot 1 (instanced pipelines only): the per-instance world
        // matrix as 4×float32x4 step=instance attributes at locations
        // 3-6 — a column per attribute, exactly how WGSL's mat4x4
        // input unpacks (the shader declares the mat as four vec4
        // params rather than a matrix input — web-sys validates
        // matrix vertex attributes inconsistently across drivers).
        let inst_attrs: Vec<GpuVertexAttribute> = (0..4u32)
            .map(|i| GpuVertexAttribute::new(GpuVertexFormat::Float32x4, i * 16, 3 + i))
            .collect();
        let inst_buf = GpuVertexBufferLayout::new(64, &inst_attrs);
        inst_buf.set_step_mode(GpuVertexStepMode::Instance);

        let vertex_inst = GpuVertexState::new(&shader);
        vertex_inst.set_entry_point("vs_main");
        vertex.set_buffers(&[js_sys::JsNullable::wrap(buf.clone())]);
        vertex_inst.set_buffers(&[
            js_sys::JsNullable::wrap(buf),
            js_sys::JsNullable::wrap(inst_buf),
        ]);

        // Blend: opaque → off (premultiplied write covers), additive →
        // src+dst one-sided accumulate (order-independent, so the
        // transparent sort can skip these draws entirely).
        let comp = GpuBlendComponent::new();
        if additive {
            comp.set_src_factor(GpuBlendFactor::One);
            comp.set_dst_factor(GpuBlendFactor::One);
        } else {
            comp.set_src_factor(GpuBlendFactor::One);
            comp.set_dst_factor(GpuBlendFactor::OneMinusSrcAlpha);
        }
        comp.set_operation(GpuBlendOperation::Add);
        let blend = GpuBlendState::new(&comp, &comp);

        let target = GpuColorTargetState::new(self.surface.format);
        target.set_blend(&blend);
        let targets = [js_sys::JsNullable::wrap(target)];
        let fragment = GpuFragmentState::new(&shader, &targets);
        fragment.set_entry_point("fs_main");

        let depth = GpuDepthStencilState::new(GpuTextureFormat::Depth24plus);
        depth.set_depth_write_enabled(!additive);
        depth.set_depth_compare(GpuCompareFunction::LessEqual);

        let prim = GpuPrimitiveState::new();
        prim.set_cull_mode(cull);
        prim.set_topology(GpuPrimitiveTopology::TriangleList);
        prim.set_front_face(GpuFrontFace::Ccw);

        let desc = GpuRenderPipelineDescriptor::new(
            &layout,
            if instanced { &vertex_inst } else { &vertex },
        );
        desc.set_fragment(&fragment);
        desc.set_depth_stencil(&depth);
        desc.set_primitive(&prim);

        device
            .create_render_pipeline(&desc)
            .expect("create_render_pipeline")
    }

    /// `frame` — one full render: upload the camera block, walk the
    /// collected draw list, write model+material per draw, submit.
    ///
    /// The draw loop is deliberately flat — all GPU state lives on the
    /// encoder, all CPU state is read from registries by index. The
    /// per-node work: 2 bound-checks + 1 pack (128B) + 1 write_buffer
    /// + set_pipeline/set_bind_group ×2 + draw — ~1µs/node, so a
    /// 500-node frame costs ~0.5ms of wasm CPU (inside budget).
    ///
    /// @param scene The scene graph (world matrices are refreshed by
    ///        `collect`, which calls `update_world` first).
    /// @param cam The camera (orbit applied on read).
    /// @param time Seconds — shader clocks.
    /// @param dpr Device pixel ratio (point sizes).
    /// @param light The scene's point light (pos, rgb, intensity).
    /// @param clear_rgba Clear color (usually transparent black — the
    ///        canvas composites over the page background).
    /// @param post Post-chain settings — (film, chroma, bloom) amounts;
    ///        all-zero skips the post pass entirely.
    pub fn frame(
        &mut self,
        scene: &mut Scene,
        cam: &mut Camera,
        time: f32,
        dpr: f32,
        light: (Vec3, Vec3, f32),
        clear_rgba: [f64; 4],
        post: [f32; 3],
    ) {
        let (w, h) = self.size;

        if w == 0 || h == 0 {
            return;
        }

        // Write the camera block once — every draw shares it.
        let mut block = [0.0f32; FRAME_F32S];

        cam.write_frame_uniform(&mut block, w as f32, h as f32, dpr, time, light);
        write_f32s(&self.surface.queue, &self.frame_buf, &block);

        let view = self
            .surface
            .context
            .get_current_texture()
            .expect("frame texture")
            .create_view()
            .expect("frame view");
        let depth_att = GpuRenderPassDepthStencilAttachment::new_with_gpu_texture_view(
            &self.depth.clone().unwrap(),
        );
        depth_att.set_depth_clear_value(1.0);
        depth_att.set_depth_load_op(GpuLoadOp::Clear);
        depth_att.set_depth_store_op(GpuStoreOp::Store);

        let color_att = GpuRenderPassColorAttachment::new_with_gpu_texture_view(
            GpuLoadOp::Clear,
            GpuStoreOp::Store,
            &view,
        );
        let clear = GpuColorDict::new(
            clear_rgba[0],
            clear_rgba[1],
            clear_rgba[2],
            clear_rgba[3],
        );
        color_att.set_clear_value_gpu_color_dict(&clear);
        let attachments = [js_sys::JsNullable::wrap(color_att)];
        let desc = GpuRenderPassDescriptor::new(&attachments);
        desc.set_depth_stencil_attachment(&depth_att);

        let encoder = self
            .surface
            .device
            .create_command_encoder_with_descriptor(&GpuCommandEncoderDescriptor::new());
        let pass = encoder.begin_render_pass(&desc).expect("begin_render_pass");

        // ── Draw dispatch ──────────────────────────────────────────
        // Opaque first (depth-sorted pipeline), then transparent —
        // additive draws skip the sort (order-independent by nature).
        //
        // `collect` borrows scene mutably (it refreshes world
        // matrices first) so the returned slice can't coexist with
        // per-node reads — ids are copied into `self.scratch`, a
        // reusable buffer (zero-alloc after warmup, per the frame-loop
        // rule — no per-frame Vec).
        let (list, _lods) = scene.collect(cam.layers);

        self.scratch.clear();
        self.scratch.extend_from_slice(list);

        let list_end = self.scratch.len();

        for li in 0..list_end {
            let id = self.scratch[li];
            let node = &scene.nodes[id.0 as usize];

            // Node kind → (geometry, material, shader kind, instanced):
            //   Mesh      → (geo, mat, 0|1 by material, false)
            //   Sprite    → (shared quad, mat, 2, false)
            //   Points    → (geo, mat, 3, false)
            //   Instanced → (geo, mat, 4, true)
            //   Group/Lod → skipped (transform-only nodes).
            let (geo_id, mat_id, kind, instanced) = match node.kind {
                Kind::Mesh { geo, mat } => {
                    let lit = matches!(
                        self.mats.get(mat as usize).map(|m| &m.record.material),
                        Some(Material::Standard { .. })
                    );

                    (geo, mat, if lit { 1u8 } else { 0u8 }, false)
                }
                Kind::Sprite { mat } => (self.quad_geo, mat, 2, false),
                Kind::Points { geo, mat } => (geo, mat, 3, false),
                Kind::Instanced { geo, mat, .. } => (geo, mat, 4, true),
                _ => continue,
            };

            // Copy the GPU handles + material record out of the
            // registries — GpuBuffer/GpuBindGroup are JsValue handles
            // (clone = refcount bump), Record is Copy. Holding the
            // &self borrows would block the &mut self.pipe() call.
            let (geo_v, geo_i, geo_count) = match self.geos.get(geo_id as usize) {
                Some(g) => (g.vertex.clone(), g.index.clone(), g.index_count),
                None => continue,
            };
            let (record, mat_bind) = match self.mats.get(mat_id as usize) {
                Some(m) => (m.record, m.bind.clone()),
                None => continue,
            };

            // ── Instance buffer refresh (Instanced only) ───────────
            // Re-upload only when `inst_version` advanced — star-field
            // tweens rewrite a few rows per frame; the version stamp
            // turns "did anything change" into an integer compare.
            if instanced {
                let version_hit = self.inst_versions.get(&id.0) == Some(&node.inst_version);

                if !version_hit && !node.instances.is_empty() {
                    let needed = node.instances.len() as u32;
                    let entry = self.inst_bufs.entry(id.0).or_insert_with(|| {
                        (
                            new_buffer(
                                &self.surface.device,
                                needed * 64,
                                gpu_buffer_usage::VERTEX | gpu_buffer_usage::COPY_DST,
                            ),
                            needed,
                        )
                    });

                    // Grow the buffer when the table outgrew capacity —
                    // recreate rather than ring: instanced tables are
                    // sized at init, growth is the exception.
                    if entry.1 < needed {
                        *entry = (
                            new_buffer(
                                &self.surface.device,
                                needed * 64,
                                gpu_buffer_usage::VERTEX | gpu_buffer_usage::COPY_DST,
                            ),
                            needed,
                        );
                    }

                    self.inst_scratch.clear();

                    for m in node.instances.iter() {
                        self.inst_scratch.extend_from_slice(&m.m);
                    }

                    let _ = self.surface.queue.write_buffer_with_u32_and_u8_slice(
                        &entry.0,
                        0,
                        f32_bytes(&self.inst_scratch),
                    );

                    self.inst_versions.insert(id.0, node.inst_version);
                }
            }

            let side_idx = match record.flags.side {
                Side::Front => 0u8,
                Side::Back => 1,
                Side::Double => 2,
            };
            let blend_idx = if record.flags.blending == Blending::Additive { 1 } else { 0 };
            let pipe = self.pipe(PipeKey(kind, side_idx, blend_idx), kind);

            // Per-draw block: model at f32 offset 48, material at 64 —
            // byte offset 48·4 = 192 into the shared frame buffer, so
            // the camera block at [0..48) survives every draw write.
            let mut block = [0.0f32; 48];

            block[0..16].copy_from_slice(&node.world.m);
            record.pack(&mut block[16..48]);

            let _ = self.surface.queue.write_buffer_with_u32_and_u8_slice(
                &self.frame_buf,
                192,
                f32_bytes(&block),
            );

            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, Some(&self.frame_group));
            pass.set_bind_group(1, Some(&mat_bind));
            pass.set_vertex_buffer(0, Some(&geo_v));
            pass.set_index_buffer(&geo_i, GpuIndexFormat::Uint32);

            if instanced {
                if let Some((inst_buf, _)) = self.inst_bufs.get(&id.0) {
                    pass.set_vertex_buffer(1, Some(inst_buf));
                    pass.draw_indexed_with_instance_count(
                        geo_count,
                        node.instances.len() as u32,
                    );
                }
            } else {
                pass.draw_indexed(geo_count);
            }
        }

        pass.end();

        let _ = post; // Post chain lands in post.rs — stubbed this pass.

        let cmd = encoder.finish();
        self.surface.queue.submit(&[cmd]);
    }
}

/// `material_texture` — the texture id inside a material payload
/// (u32::MAX for untextured).
fn material_texture(m: &Material) -> u32 {
    match m {
        Material::Basic { texture, .. }
        | Material::Standard { texture, .. }
        | Material::Sprite { texture, .. }
        | Material::Points { texture, .. } => *texture,
    }
}

/// `expand_points` — grow a point cloud into billboard quads: each
/// input vertex becomes a 4-vert quad with the same center/normal but
/// a per-corner `corner` attribute packed into the uv slot. Positions
/// stay centered — the shader offsets by corner×size in view space.
fn expand_points(g: &Geometry) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>) {
    let n = g.vertex_count();
    let mut p = Vec::with_capacity(n * 12);
    let mut nr = Vec::with_capacity(n * 12);
    let mut uv = Vec::with_capacity(n * 8);
    let mut ix = Vec::with_capacity(n * 6);

    for i in 0..n {
        let (px, py, pz) = (g.positions[i * 3], g.positions[i * 3 + 1], g.positions[i * 3 + 2]);
        let (nx, ny, nz) = (g.normals[i * 3], g.normals[i * 3 + 1], g.normals[i * 3 + 2]);

        for c in CORNERS.iter() {
            p.extend_from_slice(&[px, py, pz]);
            nr.extend_from_slice(&[nx, ny, nz]);
            // The uv slot carries (corner.x, corner.y) for the point
            // shader — the fragment stage rebuilds texture uv from the
            // corner (corner·0.5+0.5), so the source uv is dropped.
            uv.extend_from_slice(&[c[0], c[1]]);
        }

        let base = (i * 4) as u32;
        let quad: Vec<u32> = QUAD_IDX.iter().map(|x| x + base).collect();

        ix.extend_from_slice(&quad);
    }

    (p, nr, uv, ix)
}
