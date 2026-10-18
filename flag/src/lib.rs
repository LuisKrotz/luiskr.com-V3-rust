//! flag — the locale flag widget.
//!
//! Standalone per-widget artifact — see `rust/README.md`. The shader
//! warps one or two flag textures: wind sine envelope (pole-fixed →
//! free-fly edge), horizontal foreshortening, specular sheen, seam
//! darkening on split flags, per-locale `flutter`/`specular`
//! personalities (`GPU_FLAG_ANIMS` on the JS side, 12 variants).
//!
//! # Uniform layout (f32s = 12 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[FLAG]` and the WGSL struct)
//!
//!   [0,1]  res       canvas CSS px
//!   [2]    time      seconds — wind phase advance
//!   [3]    hover     0..1 hover amount (amplitude boost)
//!   [4]    anim_type locale personality 0..11 (`GPU_FLAG_ANIMS`)
//!   [5]    is_split  0 single flag / 1 split flag (two textures)
//!   [6]    split_x   split seam position 0..1 (0.5 = halves)
//!   [7]    pad       16B alignment pad
//!   [8..11] rect     (0,0,w,h) on the dedicated canvas
//!
//! # Bindings (must match the WGSL `@binding` indices)
//!
//!   0 uniform · 1 texture1 (primary) · 2 texture2 (secondary —
//!   single-flag widgets bind tex1 twice) · 3 linear sampler
//!
//! # Frame cost
//!
//! One ~48 B uniform write + one 3-vertex draw; textures upload once
//! via `upload_textures`, not per frame.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{gpu_buffer_usage, HtmlCanvasElement};

/// WGSL — wind-warp + specular flag shader.
const WGSL: &str = include_str!("flag.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 12;

/// GPU state — surface + the texture-dependent bind group (rebuilt
/// when `upload_textures` lands new pixels).
struct Gpu {
    /// The live canvas surface.
    surface: gpu_mini::Surface,
    /// The flag pipeline (fullscreen triangle, textured).
    pipeline: web_sys::GpuRenderPipeline,
    /// Uniform block — the 12-f32 layout above.
    ubuf: web_sys::GpuBuffer,
    /// Shared linear sampler for both flag textures.
    sampler: web_sys::GpuSampler,
    /// Primary flag texture (current flag).
    tex1: RefCell<Option<web_sys::GpuTexture>>,
    /// Secondary texture (split-flag second half; tex1 when absent).
    tex2: RefCell<Option<web_sys::GpuTexture>>,
    /// The bind-group layout (kept so `rebind` can rebuild `bind`).
    layout: web_sys::GpuBindGroupLayout,
    /// Bound resources — rebuilt by `rebind` after texture uploads.
    bind: RefCell<Option<web_sys::GpuBindGroup>>,
}

/// Flag state repacked into the uniform block per frame.
struct State {
    /// Canvas CSS px — the `res` field (design-unit domain).
    css: [f32; 2],
    /// Canvas DEVICE px — the `rect.zw` divisor mapping `frag`
    /// (device px) into widget uv 0..1. Written by `resize`.
    dev: [f32; 2],
    /// 0..1 hover amount — drives the wind-amplitude boost.
    hover: f32,
    /// Locale personality discriminant (0..11).
    anim: f32,
    /// 0 single flag / 1 split flag.
    is_split: f32,
    /// Split seam position 0..1 (0.5 = halves).
    split_x: f32,
}

/// Public wasm surface — one flag canvas per instance.
#[wasm_bindgen]
pub struct Flag {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Accumulated seconds — the wind phase clock.
    clock: Cell<f32>,
}

#[wasm_bindgen]
impl Flag {
    /// `Flag::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Flag {
        Flag {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                css: [1.0, 1.0],
                dev: [1.0, 1.0],
                hover: 0.0,
                anim: 0.0,
                is_split: 0.0,
                split_x: 0.5,
            }),
            clock: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline + sampler; the bind group builds
    /// once textures land (`upload_textures` → `rebind`).
    ///
    /// @param canvas The widget's dedicated canvas.
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub async fn init(&self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
        let surface = gpu_mini::init_surface(&canvas).await?;

        // binding 0 uniform · 1 tex1 · 2 tex2 · 3 sampler — the WGSL's
        // declared contract; indices must not drift.
        let layout = gpu_mini::new_layout(
            &surface.device,
            &[
                gpu_mini::uniform_entry(0),
                gpu_mini::texture_entry(1),
                gpu_mini::texture_entry(2),
                gpu_mini::sampler_entry(3),
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
            tex1: RefCell::new(None),
            tex2: RefCell::new(None),
            layout,
            bind: RefCell::new(None),
        });

        Ok(())
    }

    /// `upload_textures` — stage one or two RGBA8 flag images. Single-
    /// flag widgets pass `rgba_b` identical to `rgba_a` (the shader
    /// reads tex2 only when is_split=1 — a duplicate binding is the
    /// cheapest no-op). Rebuilds the bind group so the new textures go
    /// live next frame.
    ///
    /// `ImageBitmap` inputs — the browser decodes the SVG off-main-
    /// thread and `copyExternalImageToTexture` moves pixels GPU→GPU,
    /// so no CPU pixel round-trip ever touches the flag path.
    ///
    /// @param a The primary flag's decoded bitmap.
    /// @param b The secondary flag's decoded bitmap (same as `a` for
    ///          unsplit locales — a duplicate binding is the cheapest
    ///          no-op).
    pub fn upload_textures(&self, a: web_sys::ImageBitmap, b: web_sys::ImageBitmap) {
        let gpu = self.gpu.borrow();

        let Some(g) = gpu.as_ref() else {
            return;
        };

        let t1 = gpu_mini::bitmap_texture(
            &g.surface.device,
            &g.surface.queue,
            &a,
            a.width(),
            a.height(),
        );

        let t2 = gpu_mini::bitmap_texture(
            &g.surface.device,
            &g.surface.queue,
            &b,
            b.width(),
            b.height(),
        );

        *g.tex1.borrow_mut() = Some(t1);
        *g.tex2.borrow_mut() = Some(t2);

        // Rebuild the bind group with the fresh views — bind groups are
        // immutable by spec, so a texture swap means a new group.
        let bind = gpu_mini::new_bind_group(
            &g.surface.device,
            &g.layout,
            &[
                gpu_mini::bind_buffer_entry(0, &g.ubuf),
                gpu_mini::bind_view_entry(
                    1,
                    &gpu_mini::view(g.tex1.borrow().as_ref().unwrap()),
                ),
                gpu_mini::bind_view_entry(
                    2,
                    &gpu_mini::view(g.tex2.borrow().as_ref().unwrap()),
                ),
                gpu_mini::bind_sampler_entry(3, &g.sampler),
            ],
        );

        *g.bind.borrow_mut() = Some(bind);
    }

    /// `resize` — canvas-bitmap change → surface re-configure.
    ///
    /// @param _w Device-pixel width (element-side).
    /// @param _h Device-pixel height.
    pub fn resize(&self, w: u32, h: u32) {
        self.state.borrow_mut().dev = [(w.max(1)) as f32, (h.max(1)) as f32];

        let gpu = self.gpu.borrow();

        if let Some(g) = gpu.as_ref() {
            gpu_mini::resize(&g.surface);
        }
    }

    /// `set_size` — the canvas's CSS-pixel size (res + rect).
    ///
    /// @param w Canvas width in CSS pixels.
    /// @param h Canvas height in CSS pixels.
    pub fn set_size(&self, w: f32, h: f32) {
        self.state.borrow_mut().css = [w.max(1.0), h.max(1.0)];
    }

    /// `set_state` — bundled scalar update: hover amount, the locale
    /// personality, and the split-flag geometry.
    ///
    /// @param hover 0..1 hover amount.
    /// @param anim Locale personality 0..11 (`GPU_FLAG_ANIMS`).
    /// @param is_split 0 single / 1 split flag.
    /// @param split_x Seam position 0..1.
    pub fn set_state(&self, hover: f32, anim: f32, is_split: f32, split_x: f32) {
        let mut s = self.state.borrow_mut();

        s.hover = hover.clamp(0.0, 1.0);
        s.anim = anim;
        s.is_split = if is_split > 0.5 { 1.0 } else { 0.0 };
        s.split_x = split_x.clamp(0.0, 1.0);
    }

    /// `frame` — one animation tick: pack, write, draw. Skips silently
    /// until textures land (no bind group yet — an unbound draw would
    /// be a validation error).
    ///
    /// @param dt Seconds since the previous call.
    pub fn frame(&self, dt: f32) {
        let gpu = self.gpu.borrow();

        let Some(g) = gpu.as_ref() else {
            return;
        };

        let bind = g.bind.borrow();

        let Some(bind) = bind.as_ref() else {
            return;
        };

        self.clock.set(self.clock.get() + dt);

        let s = self.state.borrow();

        // [0,1] res [2] time [3] hover [4] anim [5] is_split
        // [6] split_x [7] pad [8..11] rect = (0,0,w,h)
        let uniforms: [f32; UNIFORM_F32S] = [
            s.css[0],
            s.css[1],
            self.clock.get(),
            s.hover,
            s.anim,
            s.is_split,
            s.split_x,
            0.0,
            0.0,
            0.0,
            s.dev[0],
            s.dev[1],
        ];

        drop(s);

        gpu_mini::write_f32s(&g.surface.queue, &g.ubuf, &uniforms);

        if !gpu_mini::draw(&g.surface, &g.pipeline, bind, None) {
            gpu_mini::resize(&g.surface);
        }
    }
}
