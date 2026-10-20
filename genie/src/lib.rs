//! genie — the expand-modal media mesh-warp transition.
//!
//! Standalone per-widget artifact — see `rust/README.md`. Replaces the
//! CSS modal zoom with the macOS genie field: the composited modal
//! sheet texture is drawn as a 192-row synthesized mesh whose vertices
//! deform through the shared per-row staggered eased lerp (the
//! ui-layouts mac-genie reference field — `core/tokens/motion/genie.ts`
//! holds the identical math the 2D backend runs), plus the reference's
//! dock-flash accent in the fragment stage. JS owns the clock and
//! pushes funnel progress + rects per frame; the crate owns the draw.
//!
//! # Geometry without buffers
//!
//! The pipeline synthesizes vertices from `vertex_index` — no vertex
//! or index buffers: the mesh is ROWS-1 quads × 6 verts (570 draws).
//!
//! # Uniform layout (f32s = 24 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[GENIE]` and the WGSL struct)
//!
//!   [0,1]   res       canvas DEVICE px
//!   [2]     time      seconds
//!   [3]     prog      funnel progress 0..1 (0 = modal sheet,
//!                     1 = figure funnel — JS-owned linear clock)
//!   [4..8]  modal     modal rect (x,y,w,h) CSS px
//!   [8..12] fig       figure rect
//!   [12,13] reserved  (kept for the 24-f32 ABI)
//!   [14]    dpr       device/css ratio (rects arrive in CSS px)
//!   [15]    flags     bit0 opening · bit1 nearBottom · bit2 hasTex
//!   [16..20] tint     accent rgba (untex'd ghost fill)
//!   [20..24] uv       texture uv scale.xy + offset.xy
//!
//! # Bindings
//!
//!   Mesh pipeline:      0 uniform · 1 texture · 2 sampler
//!
//! # Reduced motion
//!
//! JS skips the widget entirely under reduced-motion (the transition
//! simply doesn't run) — the crate has no motion gate.

#[cfg(target_arch = "wasm32")]
use std::cell::{Cell, RefCell};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::{GpuBindGroupLayout, GpuDevice, HtmlCanvasElement, ImageBitmap};

/// WGSL — the mesh deform + dock-flash fragment.
#[cfg(target_arch = "wasm32")]
const WGSL: &str = include_str!("genie.wgsl");

/// Uniform f32 count — see module doc.
#[cfg(target_arch = "wasm32")]
const UNIFORM_F32S: usize = 24;

/// Mesh vertex count — (ROWS-1) quads × 6 verts, synthesized in WGSL
/// (ROWS = 192 → 191 quads; ~4.7px/row on a 900px sheet keeps the
/// funnel's sloped walls raster-smooth — 96 rows stair-stepped them).
#[cfg(target_arch = "wasm32")]
const MESH_VERTS: u32 = 1146;

/// GPU state — surface, pipeline, buffers, bind group, texture.
#[cfg(target_arch = "wasm32")]
struct GpuState {
    /// The live canvas surface.
    surface: gpu_mini::Surface,
    /// The mesh-deform pipeline (texture+sampler+uniform).
    mesh_pipe: web_sys::GpuRenderPipeline,
    /// The shared 24-f32 uniform block.
    ubuf: web_sys::GpuBuffer,
    /// Mesh bind-group layout — kept so `upload_bitmap` can rebuild.
    mesh_layout: web_sys::GpuBindGroupLayout,
    /// Mesh bind group — rebuilt on texture upload.
    mesh_bind: web_sys::GpuBindGroup,
    /// The media texture (placeholder 1×1 until `upload_bitmap`).
    tex: web_sys::GpuTexture,
    /// The shared linear sampler.
    sampler: web_sys::GpuSampler,
}

/// Scalar state repacked into the uniform block per frame.
#[cfg(target_arch = "wasm32")]
struct State {
    /// Canvas device px — `res`.
    dev: [f32; 2],
    /// Canvas CSS px — derives `dpr` against `dev`.
    css: [f32; 2],
    /// Modal rect, CSS px — the field's source (S).
    from: [f32; 4],
    /// Figure rect, CSS px — the funnel's target (D).
    to: [f32; 4],
    /// bit0 opening · bit1 nearBottom · bit2 hasTex.
    flags: u32,
    /// Accent rgba — tints the untextured ghost fill.
    tint: [f32; 4],
    /// Texture uv transform (1,1,0,0 — the modal fits the whole frame).
    uv: [f32; 4],
    /// Eased funnel progress — pushed by the JS driver each frame
    /// (0 = modal sheet, 1 = figure funnel).
    prog: f32,
}

/// Public wasm surface — one genie transition per expand modal.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct Genie {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<GpuState>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Elapsed seconds — the `time` uniform's clock.
    clock: Cell<f32>,
}

/// `pipeline` — one shader module, one entry-point pair, premult-over
/// blend. Mirrors `gpu_mini::fullscreen_pipeline` minus the vertex-
/// buffer-free fullscreen assumption (same shape — no buffers).
///
/// @param device The shared GPUDevice.
/// @param format The surface's swap format.
/// @param vs Vertex entry point name.
/// @param fs Fragment entry point name.
/// @param layouts Bind-group layouts for the pipeline.
/// @returns The render pipeline, or Err on a device fault.
#[cfg(target_arch = "wasm32")]
fn pipeline(
    device: &GpuDevice,
    format: web_sys::GpuTextureFormat,
    vs: &str,
    fs: &str,
    layouts: &[GpuBindGroupLayout],
) -> Result<web_sys::GpuRenderPipeline, JsValue> {
    use web_sys::*;

    let shader = device.create_shader_module(&GpuShaderModuleDescriptor::new(WGSL));

    let layout_arr: Vec<js_sys::JsNullable<GpuBindGroupLayout>> = layouts
        .iter()
        .map(|l| js_sys::JsNullable::wrap(l.clone()))
        .collect();
    let layout = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&layout_arr));

    let vertex = GpuVertexState::new(&shader);
    vertex.set_entry_point(vs);

    let comp = GpuBlendComponent::new();
    comp.set_src_factor(GpuBlendFactor::One);
    comp.set_dst_factor(GpuBlendFactor::OneMinusSrcAlpha);
    comp.set_operation(GpuBlendOperation::Add);
    let blend = GpuBlendState::new(&comp, &comp);

    let target = GpuColorTargetState::new(format);
    target.set_blend(&blend);
    let targets = [js_sys::JsNullable::wrap(target)];

    let fragment = GpuFragmentState::new(&shader, &targets);
    fragment.set_entry_point(fs);

    let desc = GpuRenderPipelineDescriptor::new(&layout, &vertex);
    desc.set_fragment(&fragment);
    device
        .create_render_pipeline(&desc)
        .map_err(|e| JsValue::from_str(&format!("create_render_pipeline: {e:?}")))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Genie {
    /// `Genie::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Genie {
        Genie {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                dev: [1.0, 1.0],
                css: [1.0, 1.0],
                from: [0.0, 0.0, 1.0, 1.0],
                to: [0.0, 0.0, 1.0, 1.0],
                flags: 0,
                tint: [1.0, 1.0, 1.0, 1.0],
                uv: [1.0, 1.0, 0.0, 0.0],
                prog: 0.0,
            }),
            clock: Cell::new(0.0),
        }
    }

    /// `init` — device + the mesh pipeline + the placeholder texture.
    ///
    /// @param canvas The overlay canvas.
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        use web_sys::*;

        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // Mesh: 0 uniform · 1 texture · 2 sampler — genie.wgsl.
        let mesh_layout = gpu_mini::new_layout(
            &surface.device,
            &[
                gpu_mini::uniform_entry(0),
                gpu_mini::texture_entry(1),
                gpu_mini::sampler_entry(2),
            ],
        )?;

        let mesh_pipe = pipeline(
            &surface.device,
            surface.format,
            "vs_mesh",
            "fs_mesh",
            &[mesh_layout.clone()],
        )?;

        let ubuf = gpu_mini::new_buffer(
            &surface.device,
            (UNIFORM_F32S * 4) as u32,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        )?;

        let sampler = gpu_mini::linear_sampler(&surface.device);

        // 1×1 placeholder — the mesh pipeline always has a texture to
        // bind; `has_tex` stays clear until a real bitmap uploads.
        let tex = gpu_mini::new_texture(&surface.device, 1, 1)?;

        gpu_mini::upload_rgba(&surface.queue, &tex, 1, 1, &[0u8, 0, 0, 0]);

        let tex_view = gpu_mini::view(&tex)?;

        let mesh_bind = gpu_mini::new_bind_group(
            &surface.device,
            &mesh_layout,
            &[
                gpu_mini::bind_buffer_entry(0, &ubuf),
                gpu_mini::bind_view_entry(1, &tex_view),
                gpu_mini::bind_sampler_entry(2, &sampler),
            ],
        );

        // try_borrow everywhere: a JS exception crossing a wasm frame
        // mid-call leaks the RefCell guard — a later borrow() panics
        // `unreachable` (the observed widget-crash cascade).
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };
        *slot = Some(GpuState {
            surface,
            mesh_pipe,
            ubuf,
            mesh_layout,
            mesh_bind,
            tex,
            sampler,
        });

        Ok(())
    }

    /// `set_rects` — the warp's origin + destination rects, CSS px.
    ///
    /// @param fx @param fy @param fw @param fh Origin rect.
    /// @param tx @param ty @param tw @param th Destination rect.
    pub fn set_rects(
        &self,
        fx: f32,
        fy: f32,
        fw: f32,
        fh: f32,
        tx: f32,
        ty: f32,
        tw: f32,
        th: f32,
    ) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.from = [fx, fy, fw, fh];
            s.to = [tx, ty, tw, th];
        }
    }

    /// `set_flags` — direction + facing edge.
    ///
    /// @param opening 1.0 while opening, 0.0 on the collapse.
    /// @param near_bottom 1.0 when the origin sits below the destination.
    pub fn set_flags(&self, opening: f32, near_bottom: f32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.flags = (s.flags & !3u32)
                | (opening > 0.5) as u32
                | (((near_bottom > 0.5) as u32) << 1);
        }
    }

    /// `set_tint` — the accent rgba (0-1, theme-resolved in JS) the
    /// untextured ghost fill reads.
    ///
    /// @param r @param g @param b @param a Rgba channels.
    pub fn set_tint(&self, r: f32, g: f32, b: f32, a: f32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.tint = [r, g, b, a];
        }
    }

    /// `set_progress` — the JS-owned spring progress 0..1.
    ///
    /// @param p Progress.
    pub fn set_progress(&self, p: f32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.prog = p;
        }
    }

    /// `set_size` — canvas CSS px (drives the dpr uniform against `dev`).
    ///
    /// @param w @param h CSS size.
    pub fn set_size(&self, w: f32, h: f32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.css = [w.max(1.0), h.max(1.0)];
        }
    }

    /// `resize` — device px — the `res` uniform + surface re-configure.
    ///
    /// @param w @param h Backing size in device pixels.
    pub fn resize(&self, w: f32, h: f32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.dev = [w.max(1.0), h.max(1.0)];
        }

        if let Ok(gpu) = self.gpu.try_borrow() {
            if let Some(g) = gpu.as_ref() {
                gpu_mini::resize(&g.surface);
            }
        }
    }

    /// `upload_bitmap` — decode happened off-thread (createImageBitmap);
    /// GPU→GPU copy into a fresh texture, then the mesh bind group is
    /// rebuilt against it and `has_tex` goes live.
    ///
    /// @param bitmap The decoded frame.
    /// @param w @param h Bitmap pixel size.
    pub fn upload_bitmap(&self, bitmap: ImageBitmap, w: u32, h: u32) {
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return;
        };

        let Some(g) = slot.as_mut() else {
            return;
        };

        let Ok(tex) = gpu_mini::bitmap_texture(&g.surface.device, &g.surface.queue, &bitmap, w, h)
        else {
            return;
        };

        let Ok(view) = gpu_mini::view(&tex) else {
            return;
        };

        g.mesh_bind = gpu_mini::new_bind_group(
            &g.surface.device,
            &g.mesh_layout,
            &[
                gpu_mini::bind_buffer_entry(0, &g.ubuf),
                gpu_mini::bind_view_entry(1, &view),
                gpu_mini::bind_sampler_entry(2, &g.sampler),
            ],
        );

        g.tex = tex;

        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.flags |= 4;
        }
    }

    /// `frame` — repack uniforms + one render pass (the mesh draw).
    ///
    /// @param dt Frame delta in seconds.
    /// @returns false once the surface is stale/lost.
    pub fn frame(&self, dt: f32) -> bool {
        let Ok(gpu) = self.gpu.try_borrow() else {
            return false;
        };

        let Some(g) = gpu.as_ref() else {
            return false;
        };

        let t = self.clock.get() + dt;

        self.clock.set(t);

        {
            let Ok(s) = self.state.try_borrow() else {
                return false;
            };

            // Uniform repack — the 24-f32 block documented above.
            let dpr = if s.css[0] > 0.0 { s.dev[0] / s.css[0] } else { 1.0 };

            let mut u = [0.0f32; UNIFORM_F32S];

            u[0] = s.dev[0];
            u[1] = s.dev[1];
            u[2] = t;
            u[3] = s.prog;
            u[4..8].copy_from_slice(&s.from);
            u[8..12].copy_from_slice(&s.to);
            // u[12], u[13] reserved — the funnel field has no tunables.
            u[14] = dpr;
            u[15] = s.flags as f32;
            u[16..20].copy_from_slice(&s.tint);
            u[20..24].copy_from_slice(&s.uv);

            gpu_mini::write_f32s(&g.surface.queue, &g.ubuf, &u);
        }

        use web_sys::*;

        let tex = match g.surface.context.get_current_texture() {
            Ok(t) => t,
            Err(_) => return false,
        };

        let view = match tex.create_view() {
            Ok(v) => v,
            Err(_) => return false,
        };

        let clear = GpuColorDict::new(0.0, 0.0, 0.0, 0.0);
        let attach = GpuRenderPassColorAttachment::new_with_gpu_texture_view(
            GpuLoadOp::Clear,
            GpuStoreOp::Store,
            &view,
        );
        attach.set_clear_value_gpu_color_dict(&clear);

        let attachments = [js_sys::JsNullable::wrap(attach)];
        let pass_desc = GpuRenderPassDescriptor::new(&attachments);

        let encoder = g
            .surface
            .device
            .create_command_encoder_with_descriptor(&GpuCommandEncoderDescriptor::new());

        match encoder.begin_render_pass(&pass_desc) {
            Ok(pass) => {
                pass.set_pipeline(&g.mesh_pipe);
                pass.set_bind_group(0, Some(&g.mesh_bind));
                pass.draw(MESH_VERTS);

                pass.end();
            }
            Err(_) => return false,
        }

        g.surface.queue.submit(&[encoder.finish()]);

        true
    }
}

/// `destroy` parity — forget the GPU state instead of destructing it:
/// the FinalizationRegistry's `__wbg_*_free` walks every stored
/// JsValue's externref dealloc — on a torn-down table that traps
/// `memory access out of bounds` during GC. GPU objects die with the
/// shared device; leaking the handles is safe and un-panickable.
#[cfg(target_arch = "wasm32")]
impl Drop for Genie {
    fn drop(&mut self) {
        if let Some(g) = self.gpu.get_mut().take() {
            std::mem::forget(g);
        }
    }
}
