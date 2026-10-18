//! media — the textured-quad widget for decoded images + video frames.
//!
//! Standalone per-widget artifact — see `rust/README.md`.
//!
//! # Decode strategy (deliberately no `image` crate)
//!
//! The browser decodes off-main-thread for free: JS hands us an
//! `ImageBitmap` (still images) or a `VideoFrame` (WebCodecs), and
//! `copyExternalImageToTexture` moves it GPU→GPU with no CPU pixel
//! round-trip. That skips ~800 KB of `image`-crate decode code — this
//! artifact stays tens of KB — AND outperforms Rust decode for the
//! common case (hardware JPEG/H.264 paths).
//!
//! # Uniform layout (f32s = 12 — 3×vec4; the WGSL `Uniforms` struct)
//!
//!   [0..3]  rect     x,y,w,h — widget rect, canvas CSS px (0,0,w,h
//!                    on the dedicated canvas)
//!   [4..7]  uv_rect  u,v,du,dv — cover-fit crop window, JS computes
//!   [8,9]   viewport canvas CSS px — the NDC projection divisor
//!   [10,11] pad      16B alignment pad
//!
//! # Bindings
//!
//!   0 uniform · 1 texture (RGBA8) · 2 linear sampler
//!
//! # Frame cost
//!
//! One ~48 B uniform write + one 6-vertex draw; texture swaps happen
//! only on `upload_bitmap`/`push_frame`, not per frame.

use std::cell::RefCell;
use wasm_bindgen::prelude::*;
use web_sys::{gpu_buffer_usage, HtmlCanvasElement, ImageBitmap, VideoFrame};

/// WGSL — the textured-quad shader (6-vertex unit quad, cover-fit UV).
const WGSL: &str = include_str!("media.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 12;

/// GPU state — surface + texture-dependent bind group.
struct Gpu {
    /// The live canvas surface.
    surface: gpu_mini::Surface,
    /// The textured-quad pipeline (6 verts, no vertex buffer).
    pipeline: web_sys::GpuRenderPipeline,
    /// Uniform block — the 12-f32 layout above.
    ubuf: web_sys::GpuBuffer,
    /// Shared linear sampler.
    sampler: web_sys::GpuSampler,
    /// Current media texture — swapped by uploads.
    tex: RefCell<Option<web_sys::GpuTexture>>,
    /// Bind-group layout — kept so uploads can rebuild `bind`.
    layout: web_sys::GpuBindGroupLayout,
    /// Bound resources — rebuilt on every texture swap.
    bind: RefCell<Option<web_sys::GpuBindGroup>>,
}

/// Media state repacked into the uniform block per frame.
struct State {
    /// Canvas CSS px — the `rect` and `viewport` fields. The vertex
    /// shader projects `rect` px → ndc via `/ viewport` — a RATIO, so
    /// any self-consistent unit works; CSS px is what `set_size`
    /// receives and what the DOM geometry API speaks, so the whole
    /// pipeline stays in CSS units with zero DPR math. (Only frag-
    /// driven shaders need device px — this one never reads `frag`.)
    css: [f32; 2],
    /// Cover-fit crop window [u, v, du, dv] — JS computes aspect math
    /// so the shader stays ignorant of source/dest aspect ratios.
    uv_rect: [f32; 4],
}

/// Public wasm surface — one media canvas per instance.
#[wasm_bindgen]
pub struct Media {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
}

#[wasm_bindgen]
impl Media {
    /// `Media::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Media {
        Media {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                css: [1.0, 1.0],
                uv_rect: [0.0, 0.0, 1.0, 1.0],
            }),
        }
    }

    /// `init` — device + pipeline + sampler; the bind group builds on
    /// the first upload.
    ///
    /// @param canvas The widget's dedicated canvas.
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub async fn init(&self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
        let surface = gpu_mini::init_surface(&canvas).await?;

        // binding 0 uniform · 1 texture · 2 sampler — the WGSL contract.
        let layout = gpu_mini::new_layout(
            &surface.device,
            &[
                gpu_mini::uniform_entry(0),
                gpu_mini::texture_entry(1),
                gpu_mini::sampler_entry(2),
            ],
        );

        let pipeline = gpu_mini::fullscreen_pipeline(
            &surface.device,
            surface.format,
            WGSL,
            &[layout.clone()],
        );

        let ubuf = gpu_mini::new_buffer(
            &surface.device,
            256,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        );

        let sampler = gpu_mini::linear_sampler(&surface.device);

        *self.gpu.borrow_mut() = Some(Gpu {
            surface,
            pipeline,
            ubuf,
            sampler,
            tex: RefCell::new(None),
            layout,
            bind: RefCell::new(None),
        });

        Ok(())
    }

    /// `rebind` — rebuild the bind group around the current texture.
    /// Bind groups are immutable, so every texture swap rebinds.
    fn rebind(&self) {
        let gpu = self.gpu.borrow();

        let Some(g) = gpu.as_ref() else {
            return;
        };

        let tex = g.tex.borrow();

        let Some(tex) = tex.as_ref() else {
            return;
        };

        let bind = gpu_mini::new_bind_group(
            &g.surface.device,
            &g.layout,
            &[
                gpu_mini::bind_buffer_entry(0, &g.ubuf),
                gpu_mini::bind_view_entry(1, &gpu_mini::view(tex)),
                gpu_mini::bind_sampler_entry(2, &g.sampler),
            ],
        );

        *g.bind.borrow_mut() = Some(bind);
    }

    /// `upload_bitmap` — paint a decoded `ImageBitmap` (JS did the
    /// decode off-main-thread via createImageBitmap). GPU→GPU copy —
    /// the fastest possible path and zero wasm-side decode code.
    ///
    /// @param bitmap The decoded image.
    /// @param w Bitmap width px.
    /// @param h Bitmap height px.
    pub fn upload_bitmap(&self, bitmap: ImageBitmap, w: u32, h: u32) {
        let gpu = self.gpu.borrow();

        let Some(g) = gpu.as_ref() else {
            return;
        };

        let tex = gpu_mini::bitmap_texture(&g.surface.device, &g.surface.queue, &bitmap, w, h);

        *g.tex.borrow_mut() = Some(tex);
        drop(gpu);
        self.rebind();
    }

    /// `push_frame` — stream a WebCodecs `VideoFrame` into the texture.
    /// Same GPU→GPU path as bitmaps — video frames upload per
    /// requestVideoFrameCallback, NOT per rAF.
    ///
    /// @param frame The decoded frame.
    /// @param w Frame width px.
    /// @param h Frame height px.
    pub fn push_frame(&self, frame: VideoFrame, w: u32, h: u32) {
        let gpu = self.gpu.borrow();

        let Some(g) = gpu.as_ref() else {
            return;
        };

        // VideoFrame → texture: same copyExternalImageToTexture path
        // (the source-info constructor takes VideoFrame directly).
        let src = web_sys::GpuCopyExternalImageSourceInfo::new_with_video_frame(&frame);
        let tex = gpu_mini::new_texture(&g.surface.device, w.max(1), h.max(1));
        let dest = web_sys::GpuCopyExternalImageDestInfo::new(&tex);
        let size = web_sys::GpuExtent3dDict::new(w.max(1));
        size.set_height(h.max(1));

        let _ = g
            .surface
            .queue
            .copy_external_image_to_texture_with_gpu_extent_3d_dict(&src, &dest, &size);

        *g.tex.borrow_mut() = Some(tex);
        drop(gpu);
        self.rebind();
    }

    /// `resize` — canvas-bitmap change → surface re-configure. The
    /// projection is pure CSS px (`rect`/`viewport` — see State doc),
    /// so no size state updates here.
    ///
    /// @param _w Device-pixel width (element-side; unused — see doc).
    /// @param _h Device-pixel height.
    pub fn resize(&self, _w: u32, _h: u32) {
        let gpu = self.gpu.borrow();

        if let Some(g) = gpu.as_ref() {
            gpu_mini::resize(&g.surface);
        }
    }

    /// `set_size` — the canvas's CSS-pixel size (rect + viewport).
    ///
    /// @param w Canvas width in CSS pixels.
    /// @param h Canvas height in CSS pixels.
    pub fn set_size(&self, w: f32, h: f32) {
        self.state.borrow_mut().css = [w.max(1.0), h.max(1.0)];
    }

    /// `set_uv_rect` — the cover-fit crop window: which sub-rect of
    /// the texture maps onto the widget. JS computes the aspect math
    /// (single responsibility — the shader just samples it).
    ///
    /// @param u Crop window left edge, 0..1.
    /// @param v Crop window top edge, 0..1.
    /// @param du Crop window width, 0..1.
    /// @param dv Crop window height, 0..1.
    pub fn set_uv_rect(&self, u: f32, v: f32, du: f32, dv: f32) {
        self.state.borrow_mut().uv_rect = [u, v, du, dv];
    }

    /// `frame` — draw the current texture. No clock needed (static
    /// until an upload lands); the bridge calls this on resize/scroll
    /// or per rAF — drawing costs one 6-vertex call.
    ///
    /// @param _dt Seconds since last call (unused — media is static
    ///            between uploads; kept for API symmetry).
    pub fn frame(&self, _dt: f32) {
        let gpu = self.gpu.borrow();

        let Some(g) = gpu.as_ref() else {
            return;
        };

        let bind = g.bind.borrow();

        let Some(bind) = bind.as_ref() else {
            return;
        };

        let s = self.state.borrow();

        // [0..3] rect=(0,0,css) CSS px [4..7] uv_rect
        // [8,9] viewport=css CSS px [10,11] pad — the px→ndc ratio is
        // unit-agnostic; CSS px keeps the whole pipeline in DOM units.
        let uniforms: [f32; UNIFORM_F32S] = [
            0.0,
            0.0,
            s.css[0],
            s.css[1],
            s.uv_rect[0],
            s.uv_rect[1],
            s.uv_rect[2],
            s.uv_rect[3],
            s.css[0],
            s.css[1],
            0.0,
            0.0,
        ];

        drop(s);

        gpu_mini::write_f32s(&g.surface.queue, &g.ubuf, &uniforms);

        // draw 6 verts — the quad is two triangles (the shared `draw`
        // hardcodes 3; media needs its own pass call, so it goes
        // through `draw_vertices` below).
        if !draw_media(&g.surface, &g.pipeline, bind) {
            gpu_mini::resize(&g.surface);
        }
    }
}

/// `draw_media` — like `gpu_mini::draw` but 6 vertices (the shader's
/// two-triangle unit quad). Kept local because only the media shader
/// uses a 6-vertex topology.
///
/// @param surface The live surface.
/// @param pipeline The media pipeline.
/// @param bind Its bind group.
/// @returns true when a frame was submitted.
fn draw_media(
    surface: &gpu_mini::Surface,
    pipeline: &web_sys::GpuRenderPipeline,
    bind: &web_sys::GpuBindGroup,
) -> bool {
    let tex = match surface.context.get_current_texture() {
        Ok(t) => t,
        Err(_) => return false,
    };

    let view = match tex.create_view() {
        Ok(v) => v,
        Err(_) => return false,
    };

    let clear = web_sys::GpuColorDict::new(0.0, 0.0, 0.0, 0.0);
    let attach = web_sys::GpuRenderPassColorAttachment::new_with_gpu_texture_view(
        web_sys::GpuLoadOp::Clear,
        web_sys::GpuStoreOp::Store,
        &view,
    );
    attach.set_clear_value_gpu_color_dict(&clear);

    let attachments = [js_sys::JsNullable::wrap(attach)];
    let pass_desc = web_sys::GpuRenderPassDescriptor::new(&attachments);

    let encoder = surface
        .device
        .create_command_encoder_with_descriptor(&web_sys::GpuCommandEncoderDescriptor::new());

    match encoder.begin_render_pass(&pass_desc) {
        Ok(pass) => {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(bind));
            pass.draw(6);
            pass.end();
        }
        Err(_) => return false,
    }

    surface.queue.submit(&[encoder.finish()]);
    true
}
