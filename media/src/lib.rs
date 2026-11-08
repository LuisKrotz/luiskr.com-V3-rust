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
use web_sys::{gpu_buffer_usage, GpuDevice, HtmlCanvasElement, ImageBitmap, VideoFrame};

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
    uniform_buffer: web_sys::GpuBuffer,
    /// Shared linear sampler.
    sampler: web_sys::GpuSampler,
    /// Current media texture — swapped by uploads.
    media_tex: RefCell<Option<web_sys::GpuTexture>>,
    /// Bind-group layout — kept so uploads can rebuild `bind`.
    layout: web_sys::GpuBindGroupLayout,
    /// Bound resources — rebuilt on every texture swap.
    bind_group: RefCell<Option<web_sys::GpuBindGroup>>,
}

/// Media state repacked into the uniform block per frame.
struct State {
    /// Canvas CSS px — the `rect` and `viewport` fields. The vertex
    /// shader projects `rect` px → ndc via `/ viewport` — a RATIO, so
    /// any self-consistent unit works; CSS px is what `set_size`
    /// receives and what the DOM geometry API speaks, so the whole
    /// pipeline stays in CSS units with zero DPR math. (Only frag-
    /// driven shaders need device px — this one never reads `frag`.)
    css_size: [f32; 2],
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
        // Pre-init state: no GPU yet; 1.0px canvas + identity crop
        // window (whole texture) keep the projection degenerate-safe.
        Media {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                css_size: [1.0, 1.0],
                uv_rect: [0.0, 0.0, 1.0, 1.0],
            }),
        }
    }

    /// `init` — device + pipeline + sampler; the bind group builds on
    /// the first upload.
    ///
    /// @param canvas The widget's dedicated canvas.
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        // Step 1 — wrap the canvas in a configured surface.
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // Step 2 — bind-group layout: binding 0 uniform · 1 texture ·
        // 2 sampler — the WGSL contract.
        let layout = gpu_mini::new_layout(
            &surface.device,
            &[
                gpu_mini::uniform_entry(0),
                gpu_mini::texture_entry(1),
                gpu_mini::sampler_entry(2),
            ],
        )?;

        // Step 3 — compile the pipeline (the media shader draws a
        // 6-vertex quad — see draw_media for the counted pass). The
        // shader keeps its own vs_main, so it takes the helpers
        // prelude (quad_corner), not the fullscreen triangle source.
        let source = gpu_mini::wgsl::helpers_source(WGSL);
        let pipeline = gpu_mini::fullscreen_pipeline(
            &surface.device,
            surface.format,
            &source,
            &[layout.clone()],
        )?;

        // Step 4 — uniform buffer; 256 B over the 48 B block keeps
        // headroom for layout growth without reallocating.
        let uniform_buffer = gpu_mini::new_buffer(
            &surface.device,
            256,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        )?;

        // Step 5 — the linear sampler the media texture reads.
        let sampler = gpu_mini::linear_sampler(&surface.device);

        // try_borrow everywhere: a JS exception crossing a wasm frame
        // mid-call leaks the RefCell guard — a later borrow() panics
        // `unreachable` (the observed widget-crash cascade).
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };
        // Publish the scaffold — tex/bind stay `None` until the first
        // upload lands; `frame` skips draws until then.
        *slot = Some(Gpu {
            surface,
            pipeline,
            uniform_buffer,
            sampler,
            media_tex: RefCell::new(None),
            layout,
            bind_group: RefCell::new(None),
        });

        Ok(())
    }

    /// `rebind` — rebuild the bind group around the current texture.
    /// Bind groups are immutable, so every texture swap rebinds.
    fn rebind(&self) {
        // Bail while init is in flight.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Still `None` → init hasn't resolved.
        let Some(gpu_state) = gpu.as_ref() else {
            return;
        };

        // Borrow the current texture slot…
        let Ok(tex) = gpu_state.media_tex.try_borrow() else {
            return;
        };

        // …`None` → no upload has landed yet; nothing to bind.
        let Some(tex) = tex.as_ref() else {
            return;
        };

        // A device fault skips the rebind — the previous bind group
        // keeps drawing rather than a wasm panic poisoning the frame
        // callbacks.
        let Ok(view) = gpu_mini::view(tex) else {
            return;
        };

        // Rebuild the group: same uniform_buffer + sampler, fresh
        // texture view.
        let bind_group = gpu_mini::new_bind_group(
            &gpu_state.surface.device,
            &gpu_state.layout,
            &[
                gpu_mini::bind_buffer_entry(0, &gpu_state.uniform_buffer),
                gpu_mini::bind_view_entry(1, &view),
                gpu_mini::bind_sampler_entry(2, &gpu_state.sampler),
            ],
        );

        // Publish — `frame` picks up the new texture next tick.
        let Ok(mut bind_slot) = gpu_state.bind_group.try_borrow_mut() else {
            return;
        };

        *bind_slot = Some(bind_group);
    }

    /// `upload_bitmap` — paint a decoded `ImageBitmap` (JS did the
    /// decode off-main-thread via createImageBitmap). GPU→GPU copy —
    /// the fastest possible path and zero wasm-side decode code.
    ///
    /// @param bitmap The decoded image.
    /// @param width Bitmap width px.
    /// @param height Bitmap height px.
    pub fn upload_bitmap(&self, bitmap: ImageBitmap, width: u32, height: u32) {
        // Bail while init is in flight.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Still `None` → init hasn't resolved.
        let Some(gpu_state) = gpu.as_ref() else {
            return;
        };

        // GPU→GPU copy — a device fault keeps the previous texture
        // rather than panicking inside the upload call.
        let Ok(tex) = gpu_mini::bitmap_texture(&gpu_state.surface.device, &gpu_state.surface.queue, &bitmap, width, height)
        else {
            return;
        };

        // Swap the texture slot — a contended borrow skips the swap
        // and the rebind below just rebinds the stale texture (still
        // drawable).
        if let Ok(mut tex_slot) = gpu_state.media_tex.try_borrow_mut() {
            *tex_slot = Some(tex);
        }

        // Release the gpu guard BEFORE rebind — it borrows the same
        // slot and would deadlock on a reentrant borrow otherwise.
        drop(gpu);

        self.rebind();
    }

    /// `push_frame` — stream a WebCodecs `VideoFrame` into the texture.
    /// Same GPU→GPU path as bitmaps — video frames upload per
    /// requestVideoFrameCallback, NOT per rAF.
    ///
    /// @param frame The decoded frame.
    /// @param width Frame width px.
    /// @param height Frame height px.
    pub fn push_frame(&self, frame: VideoFrame, width: u32, height: u32) {
        // Bail while init is in flight.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Still `None` → init hasn't resolved.
        let Some(gpu_state) = gpu.as_ref() else {
            return;
        };

        // VideoFrame → texture: same copyExternalImageToTexture path
        // (the source-info constructor takes VideoFrame directly).
        let source_info = web_sys::GpuCopyExternalImageSourceInfo::new_with_video_frame(&frame);

        // A fresh destination texture per frame — ≥1 clamps guard a
        // degenerate 0-size frame from producing an invalid object.
        let Ok(tex) = gpu_mini::new_texture(&gpu_state.surface.device, width.max(1), height.max(1)) else {
            return;
        };

        let dest_info = web_sys::GpuCopyExternalImageDestInfo::new(&tex);

        let extent = web_sys::GpuExtent3dDict::new(width.max(1));
        extent.set_height(height.max(1));

        // The GPU→GPU copy — synchronous queue call; the new frame
        // replaces the whole texture.
        let _ = gpu_state
            .surface
            .queue
            .copy_external_image_to_texture_with_gpu_extent_3d_dict(&source_info, &dest_info, &extent);

        // Swap the texture slot (contended → skip, same as upload).
        if let Ok(mut tex_slot) = gpu_state.media_tex.try_borrow_mut() {
            *tex_slot = Some(tex);
        }

        // Release the gpu guard BEFORE rebind — same reentrancy rule
        // as upload_bitmap above.
        drop(gpu);

        self.rebind();
    }

    /// `resize` — canvas-bitmap change → surface re-configure. The
    /// projection is pure CSS px (`rect`/`viewport` — see State doc),
    /// so no size state updates here.
    ///
    /// @param _width Device-pixel width (element-side; unused — see doc).
    /// @param _height Device-pixel height.
    pub fn resize(&self, _width: u32, _height: u32) {
        // Bail while init is in flight — the surface configures
        // itself when it arrives.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Re-configure the live surface to match the new bitmap.
        if let Some(gpu_state) = gpu.as_ref() {
            gpu_mini::resize(&gpu_state.surface);
        }
    }

    /// `set_size` — the canvas's CSS-pixel size (rect + viewport).
    ///
    /// @param width Canvas width in CSS pixels.
    /// @param height Canvas height in CSS pixels.
    pub fn set_size(&self, width: f32, height: f32) {
        // CSS-px pair — ≥1 clamp so the px→ndc divisors stay non-zero.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.css_size = [width.max(1.0), height.max(1.0)];
        }
    }

    /// `set_uv_rect` — the cover-fit crop window: which sub-rect of
    /// the texture maps onto the widget. JS computes the aspect math
    /// (single responsibility — the shader just samples it).
    ///
    /// @param uv_left Crop window left edge, 0..1.
    /// @param uv_top Crop window top edge, 0..1.
    /// @param uv_width Crop window width, 0..1.
    /// @param uv_height Crop window height, 0..1.
    pub fn set_uv_rect(&self, uv_left: f32, uv_top: f32, uv_width: f32, uv_height: f32) {
        // Store the crop window verbatim — JS already did the
        // cover-fit aspect math.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.uv_rect = [uv_left, uv_top, uv_width, uv_height];
        }
    }

    /// `frame` — draw the current texture. No clock needed (static
    /// until an upload lands); the bridge calls this on resize/scroll
    /// or per rAF — drawing costs one 6-vertex call.
    ///
    /// @param _delta_seconds Seconds since last call (unused — media
    ///            is static between uploads; kept for API symmetry).
    pub fn frame(&self, _delta_seconds: f32) {
        // Bail while init is in flight.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Still `None` → init hasn't resolved.
        let Some(gpu_state) = gpu.as_ref() else {
            return;
        };

        // Borrow the bind group — contended → skip the tick.
        let Ok(bind_group) = gpu_state.bind_group.try_borrow() else {
            return;
        };

        // `None` → no upload yet; an unbound draw would be a
        // validation error, so skip silently.
        let Some(bound) = bind_group.as_ref() else {
            return;
        };

        // Snapshot the scalar state under the guard…
        let Ok(state) = self.state.try_borrow() else {
            return;
        };

        // [0..3] rect_css=(0,0,css) CSS px [4..7] uv_rect
        // [8,9] viewport_css CSS px [10,11] reserved — the px→ndc
        // ratio is unit-agnostic; CSS px keeps the whole pipeline in
        // DOM units.
        let uniforms: [f32; UNIFORM_F32S] = [
            0.0,
            0.0,
            state.css_size[0],
            state.css_size[1],
            state.uv_rect[0],
            state.uv_rect[1],
            state.uv_rect[2],
            state.uv_rect[3],
            state.css_size[0],
            state.css_size[1],
            0.0,
            0.0,
        ];

        // Release the state borrow BEFORE the GPU calls — no RefCell
        // guard may be held across the JS-boundary work.
        drop(state);

        // Push the packed block to the GPU…
        gpu_mini::write_f32s(&gpu_state.surface.queue, &gpu_state.uniform_buffer, &uniforms);

        // draw 6 verts — the quad is two triangles (the shared `draw`
        // hardcodes 3; `draw_n` is the counted variant: 6 verts × 1
        // instance, same acquire→pass→submit shape).
        if !gpu_mini::draw_n(&gpu_state.surface, &gpu_state.pipeline, bound, None, 6, 1) {
            gpu_mini::resize(&gpu_state.surface);
        }
    }
}
