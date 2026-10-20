//! text-draw — the per-character writing animation (draw-text).
//!
//! Standalone per-widget artifact — see `rust/README.md`. Replaces
//! the CSS per-char `<span>` animation (`char-draw` keyframes:
//! opacity 0→1 + translateY 0.12em→0, cubic-bezier(0.22,1,0.36,1),
//! delay = offset + i·char-delay + word cascade) with ONE instanced
//! draw over a rasterized atlas:
//!
//! - JS rasterizes the rendered text into an atlas canvas ONCE (the
//!   browser's font engine does the raster — fonts don't exist in
//!   wasm) → `ImageBitmap` → `upload_atlas`.
//! - JS pushes per-char records through `set_chars`: dest rect
//!   (device px), atlas uv rect, per-char delay (the exact CSS value:
//!   offset + i·char-delay + word cascade folded in), duration.
//! - The vertex shader synthesizes each char's quad from
//!   vertex_index+instance_index against the storage sheet — no
//!   vertex buffer — applies the eased rise + a settling hand tremor,
//!   and the fragment stage reveals the atlas pixels through a
//!   wobbled stroke-direction mask so every letter reads as drawn by
//!   a pen, not faded in (see text_draw.wgsl's header for the mask
//!   math; delays/durations are the unchanged CSS values).
//!
//! The DOM text stays mounted (aria-hidden char spans + the real
//! text sibling) — indexability, screen readers, and reader mode are
//! unaffected; the canvas is the visual layer only, like the glitch
//! overlay over the portrait <img>.
//!
//! # Char-sheet layout (12 f32s per char — storage `array<CharRec>`)
//!
//!   [0..4)  dest    x, y, w, h — device px on the overlay canvas
//!   [4..8)  uv      u0, v0, u1, v1 — the char's atlas sub-rect
//!   [8]     delay   ms until this char starts (CSS delay folded:
//!                   offset + i·char-delay + word cascade)
//!   [9]     dur     ms — the char's animation duration
//!   [10]    em      the char's font-size px — `translateY(0.12em)` unit
//!   [11]    pad
//!
//! # Uniform layout (f32s = 16 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[TEXT_DRAW]` and the WGSL struct)
//!
//!   [0,1]   res     canvas DEVICE px
//!   [2]     time    ms since the draw was triggered (ms — delays are
//!                   ms-denominated to match the CSS values)
//!   [3]     count   live char count (instance count mirror — the
//!                   shader early-outs past it)
//!   [4..8]  ink     text color rgba (computed style pushed once)
//!   [8..16] pad
//!
//! # Bindings
//!
//!   0 uniform · 1 char sheet (read-only storage) ·
//!   2 atlas texture · 3 linear sampler
//!
//! # Frame cost
//!
//! One 16-f32 uniform write + one `draw(6, count)` — the whole
//! stagger, any char count, is a single draw call. When `time` passes
//! the last char's delay+dur the JS side stops driving frames (the
//! last presented frame persists) and marks the block done.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{gpu_buffer_usage, GpuDevice, HtmlCanvasElement};

/// WGSL — instanced char-quad shader.
const WGSL: &str = include_str!("text_draw.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 16;

/// Char-sheet stride in f32s — 12 per char (see module doc).
pub const CHAR_F32S: usize = 12;

/// GPU state — surface + the storage/texture-dependent bind group
/// (rebuilt when the atlas or sheet lands).
struct Gpu {
    /// The live canvas surface.
    surface: gpu_mini::Surface,
    /// The instanced char pipeline.
    pipeline: web_sys::GpuRenderPipeline,
    /// Uniform block — the 16-f32 layout above.
    ubuf: web_sys::GpuBuffer,
    /// Per-char sheet — read-only storage; RefCell'd because
    /// `set_chars` recreates the buffer on growth (the same interior-
    /// mutability pattern `tex`/`bind` use).
    sheet: RefCell<web_sys::GpuBuffer>,
    /// Sheet capacity in chars — recreated on growth (sheets are
    /// written once per trigger, so recreate-on-grow is free).
    sheet_cap: Cell<u32>,
    /// Shared linear sampler for the atlas.
    sampler: web_sys::GpuSampler,
    /// The atlas texture (uploaded once per raster).
    tex: RefCell<Option<web_sys::GpuTexture>>,
    /// The bind-group layout (kept so `rebind` can rebuild `bind`).
    layout: web_sys::GpuBindGroupLayout,
    /// Bound resources — rebuilt by `rebind` after uploads.
    bind: RefCell<Option<web_sys::GpuBindGroup>>,
}

/// Scalar state repacked into the uniform block per frame.
struct State {
    /// Canvas DEVICE px — `res`.
    dev: [f32; 2],
    /// Ink color rgba — the text's computed color.
    ink: [f32; 4],
    /// Live char count — mirrors the sheet's populated rows.
    count: u32,
    /// ms timestamp the draw triggered at — `frame` reports elapsed
    /// ms as (clock − t0)·1000 so delay/dur stay in the CSS's units.
    t0_ms: f32,
}

/// Public wasm surface — one text-draw canvas per animated block.
#[wasm_bindgen]
pub struct TextDraw {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Elapsed seconds — the stagger clock.
    clock: Cell<f32>,
}

#[wasm_bindgen]
impl TextDraw {
    /// `TextDraw::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> TextDraw {
        TextDraw {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                dev: [1.0, 1.0],
                ink: [1.0, 1.0, 1.0, 1.0],
                count: 0,
                t0_ms: 0.0,
            }),
            clock: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline + sampler; the bind group builds
    /// once the atlas + first sheet land (`upload_atlas`/`set_chars`
    /// → `rebind`).
    ///
    /// @param canvas The overlay canvas.
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // binding 0 uniform · 1 sheet · 2 atlas · 3 sampler — the
        // vertex stage reads the uniform (res/time) and the char
        // sheet (gpu-mini's entries are VERTEX+FRAGMENT for exactly
        // this reason).
        let layout = gpu_mini::new_layout(
            &surface.device,
            &[
                gpu_mini::uniform_entry(0),
                gpu_mini::storage_entry(1),
                gpu_mini::texture_entry(2),
                gpu_mini::sampler_entry(3),
            ],
        )?;

        let pipeline = gpu_mini::fullscreen_pipeline(
            &surface.device,
            surface.format,
            WGSL,
            &[layout.clone()],
        )?;

        let ubuf = gpu_mini::new_buffer(
            &surface.device,
            (UNIFORM_F32S * 4) as u32,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        )?;

        // Initial sheet: capacity for one char — `set_chars` recreates
        // on growth (sheets are written once per trigger, so the
        // recreate path is cold).
        let sheet = gpu_mini::new_buffer(
            &surface.device,
            (CHAR_F32S * 4) as u32,
            gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
        )?;

        let sampler = gpu_mini::linear_sampler(&surface.device);

        // try_borrow everywhere below: a JS exception that crossed a
        // wasm frame mid-call leaks the RefCell guard (wasm has no
        // unwinding) — a later borrow() then panics `unreachable`.
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };
        *slot = Some(Gpu {
            surface,
            pipeline,
            ubuf,
            sheet: RefCell::new(sheet),
            sheet_cap: Cell::new(1),
            sampler,
            tex: RefCell::new(None),
            layout,
            bind: RefCell::new(None),
        });

        Ok(())
    }

    /// `upload_atlas` — stage the rasterized char atlas. `ImageBitmap`
    /// input — `copyExternalImageToTexture` moves pixels GPU→GPU.
    /// Rebinds so the new atlas goes live next frame.
    ///
    /// @param bitmap The decoded atlas bitmap.
    /// @returns true when the texture landed, false before init.
    pub fn upload_atlas(&self, bitmap: &web_sys::ImageBitmap) -> bool {
        let Ok(gpu) = self.gpu.try_borrow() else {
            return false;
        };

        let Some(g) = gpu.as_ref() else {
            return false;
        };

        // A device fault reports false — the JS arm keeps the DOM
        // text visible instead of a wasm panic inside the callback.
        let Ok(tex) = gpu_mini::bitmap_texture(
            &g.surface.device,
            &g.surface.queue,
            bitmap,
            bitmap.width(),
            bitmap.height(),
        ) else {
            return false;
        };

        let Ok(mut t) = g.tex.try_borrow_mut() else {
            return false;
        };
        *t = Some(tex);
        drop(t);

        self.rebind(g);

        true
    }

    /// `set_chars` — upload the per-char sheet (12 f32s per char —
    /// see module doc). Recreates the storage buffer when the sheet
    /// outgrows capacity, then rebinds.
    ///
    /// @param values The flat f32 sheet (len = count·12).
    /// @returns The char count adopted (0 on malformed input).
    pub fn set_chars(&self, values: &[f32]) -> u32 {
        let Ok(gpu) = self.gpu.try_borrow() else {
            return 0;
        };

        let Some(g) = gpu.as_ref() else {
            return 0;
        };

        let count = (values.len() / CHAR_F32S) as u32;

        let Ok(mut s) = self.state.try_borrow_mut() else {
            return 0;
        };
        s.count = count;
        drop(s);

        if count == 0 {
            return 0;
        }

        if count > g.sheet_cap.get() {
            // A device fault mid-regrow keeps the old sheet — 0 tells
            // the JS arm the sheet wasn't adopted rather than a wasm
            // panic inside the set_chars callback.
            let Ok(sheet) = gpu_mini::new_buffer(
                &g.surface.device,
                (values.len() * 4) as u32,
                gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
            ) else {
                return 0;
            };

            let Ok(mut sh) = g.sheet.try_borrow_mut() else {
                return 0;
            };
            *sh = sheet;
            drop(sh);
            g.sheet_cap.set(count);
        }

        let Ok(sheet_ref) = g.sheet.try_borrow() else {
            return 0;
        };
        let _ = g.surface.queue.write_buffer_with_u32_and_u8_slice(
            &sheet_ref,
            0,
            gpu_mini::f32_bytes(values),
        );
        drop(sheet_ref);

        if g.tex.try_borrow().map_or(false, |t| t.is_some()) {
            self.rebind(g);
        }

        count
    }

    /// `set_ink` — push the text color (computed style read once).
    ///
    /// @param r @param g @param b @param a rgba 0-1.
    pub fn set_ink(&self, r: f32, g: f32, b: f32, a: f32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.ink = [r, g, b, a];
        }
    }

    /// `trigger` — start the stagger: the clock's current value
    /// becomes t0 so `frame` reports elapsed ms from HERE.
    pub fn trigger(&self) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.t0_ms = self.clock.get() * 1000.0;
        }
    }

    /// `set_size` — CSS px (contract parity; unused — device px only).
    ///
    /// @param _w @param _h CSS size — unused.
    pub fn set_size(&self, _w: f32, _h: f32) {}

    /// `resize` — device px — the `res` uniform.
    ///
    /// @param w @param h Backing size in device pixels.
    pub fn resize(&self, w: f32, h: f32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.dev = [w, h];
        }

        // Canvas-bitmap change → the context must re-configure or the
        // next get_current_texture lands on the stale-size (or zero-
        // size, when init ran before the overlay was laid out) surface
        // — every frame after resize drew into an invalid texture.
        if let Ok(gpu) = self.gpu.try_borrow() {
            if let Some(g) = gpu.as_ref() {
                gpu_mini::resize(&g.surface);
            }
        }
    }

    /// `rebind` — rebuild the bind group after a sheet/atlas swap.
    /// Bind groups are immutable by spec — resource swaps mean a new
    /// group, cheap at trigger-time (never mid-frame).
    fn rebind(&self, g: &Gpu) {
        let Ok(tex) = g.tex.try_borrow() else {
            return;
        };

        let Some(t) = tex.as_ref() else {
            return;
        };

        // A device fault skips the rebind — the previous group keeps
        // drawing rather than a wasm panic inside the callback.
        let Ok(v) = gpu_mini::view(t) else {
            return;
        };

        let Ok(sheet_ref) = g.sheet.try_borrow() else {
            return;
        };

        let bind = gpu_mini::new_bind_group(
            &g.surface.device,
            &g.layout,
            &[
                gpu_mini::bind_buffer_entry(0, &g.ubuf),
                gpu_mini::bind_buffer_entry(1, &sheet_ref),
                gpu_mini::bind_view_entry(2, &v),
                gpu_mini::bind_sampler_entry(3, &g.sampler),
            ],
        );
        drop(sheet_ref);
        drop(tex);

        if let Ok(mut b) = g.bind.try_borrow_mut() {
            *b = Some(bind);
        }
    }

    /// `frame` — advance the clock, repack, draw all live chars.
    ///
    /// @param dt Frame delta in seconds.
    /// @returns false once the surface is stale/lost (JS disarms).
    pub fn frame(&self, dt: f32) -> bool {
        let Ok(gpu) = self.gpu.try_borrow() else {
            return false;
        };

        let Some(g) = gpu.as_ref() else {
            return false;
        };

        let Ok(bind) = g.bind.try_borrow() else {
            return false;
        };

        let Some(b) = bind.as_ref() else {
            return false;
        };

        let t = self.clock.get() + dt;

        self.clock.set(t);

        let Ok(s) = self.state.try_borrow() else {
            return false;
        };
        let mut u = [0.0f32; UNIFORM_F32S];

        u[0] = s.dev[0];
        u[1] = s.dev[1];
        u[2] = t * 1000.0 - s.t0_ms;
        u[3] = s.count as f32;
        u[4..8].copy_from_slice(&s.ink);

        gpu_mini::write_f32s(&g.surface.queue, &g.ubuf, &u);

        if s.count == 0 {
            return true;
        }

        gpu_mini::draw_n(&g.surface, &g.pipeline, b, None, 6, s.count)
    }
}

/// `destroy` parity — forget the GPU state instead of destructing
/// it: the FinalizationRegistry's `__wbg_textdraw_free` would walk
/// every stored JsValue's externref dealloc — on a torn-down table
/// that traps `memory access out of bounds` during GC (the observed
/// `TextDrawFinalization` crash). GPU objects die with the shared
/// device; leaking the handles is safe and un-panickable.
impl Drop for TextDraw {
    fn drop(&mut self) {
        if let Some(g) = self.gpu.get_mut().take() {
            std::mem::forget(g);
        }
    }
}
