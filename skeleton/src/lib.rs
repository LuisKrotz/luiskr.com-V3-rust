//! skeleton — the standalone, first-loading WebGPU surface.
//!
//! # Why a separate crate (and why it must stay tiny)
//!
//! The loading skeleton is the ONLY animated surface that must exist
//! before the application boots — it is the shimmer users see while
//! bundles, fonts, and content stream in. It therefore gets its own
//! artifact built on `gpu-mini` (raw web-sys WebGPU bindings, not the
//! multi-MB `wgpu` framework): the whole crate compiles to tens of KB,
//! small enough to fetch beside the HTML and paint on the first
//! frames. Every OTHER animation lazy-loads its own artifact when its
//! DOM surface mounts.
//!
//! # How it differs from a traditional structure
//!
//! A traditional app draws skeletons with CSS animations (cheap but
//! static) or one shared renderer (which forces the whole engine onto
//! the critical path). This is the middle path: the same "decoding
//! glyphs" WGSL the WebGL fallback evolved into, running in a self-
//! contained module with its own GPUDevice on its own canvas — zero
//! shared state, safe to init and tear down independently.
//!
//! # SYNC CONTRACT — two copies of the shader exist
//!
//! `src/skeleton.wgsl` is byte-identical to
//! `rust/overlay/src/widgets/skeleton.wgsl` (the overlay keeps its
//! copy to document the port lineage). Edit both together. The 12-f32
//! uniform layout and the 24×4-vec4 sheet are mirrored by
//! `core/tokens/motion/gpu-widgets.ts` and the JS bridge
//! `core/utils/gpu/webgpu-skeleton.ts` — bump all together.
//!
//! # Frame loop
//!
//! JS owns the clock: the bridge calls `frame(dt)` per rAF. Per frame
//! cost: one ~64 B uniform write, one render pass, one 3-vertex draw —
//! zero allocation in the hot path.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{gpu_buffer_usage, GpuDevice, HtmlCanvasElement};

/// WGSL — the "data decoding" glyph field. Byte-identical to
/// `rust/overlay/src/widgets/skeleton.wgsl` — see SYNC CONTRACT above.
const WGSL: &str = include_str!("skeleton.wgsl");

/// MAX_RECTS — mirrors `SKELETON_RENDER.MAX_RECTS` (24) in
/// `core/tokens/motion/skeleton.ts`. The shader's `MAX_RECTS`, this
/// buffer's vec4 count, and the JS sheet-write size must all agree.
const MAX_RECTS: usize = 24;

/// vec4 fields per rect row — [rect, meta, sbase, sink] × MAX_RECTS.
const SHEET_VEC4S: usize = MAX_RECTS * 4;

/// Uniform f32 count — res(2) time resolve ink_alpha count pad(2)
/// rect(4) = 12; mirrored by `GPU_WIDGET_UNIFORM_F32S[SKELETON]`.
const UNIFORM_F32S: usize = 12;

/// GPU state — created once in `init`, never re-allocated (the frame-
/// budget rule: steady frames must not allocate GPU objects).
struct Gpu {
    /// The live canvas surface from gpu-mini.
    surface: gpu_mini::Surface,
    /// The skeleton pipeline — fullscreen triangle + storage sheet.
    pipeline: web_sys::GpuRenderPipeline,
    /// Uniform block — res/time/resolve/alpha/count/rect, 256 B.
    ubuf: web_sys::GpuBuffer,
    /// Per-rect sheet — read-only storage, MAX_RECTS×4 vec4s.
    sheet: web_sys::GpuBuffer,
    /// Pre-built bind group — valid for the pipeline's lifetime.
    bind: web_sys::GpuBindGroup,
}

/// Scalar state repacked into the uniform block each `frame`.
struct State {
    /// Resolve progress 0..1 — drives the glyph→fill collapse + fade.
    resolve: f32,
    /// Global glyph opacity multiplier (theme-driven).
    ink_alpha: f32,
    /// Live rect count (≤ MAX_RECTS).
    count: f32,
    /// Canvas CSS px — the `res` field (design-unit domain; the
    /// shader currently uses `rect` for clipping and `res` only in
    /// documentation, but the field is packed for contract parity).
    css: [f32; 2],
    /// Canvas DEVICE px — the `rect` field: `frag` arrives in device
    /// px and the shader compares sheet rows (DEVICE px too — the JS
    /// side uploads CSS geometry × DPR) against it, so both operands
    /// must share the unit.
    dev: [f32; 2],
}

/// Public wasm surface — one skeleton canvas per instance. A page may
/// mount several (hero + content areas shimmer independently).
#[wasm_bindgen]
pub struct Skeleton {
    /// GPU state — `None` until the async `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// Scalar state repacked into the uniform block per frame.
    state: RefCell<State>,
    /// Frame clock — accumulated seconds; absorbing dt jitter keeps
    /// the glyph cadence smooth (decorative field, wall-time needless).
    clock: Cell<f32>,
}

#[wasm_bindgen]
impl Skeleton {
    /// `Skeleton::new` — un-initialized surface; `init` must follow.
    /// Split because the device request is an async browser promise.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Skeleton {
        Skeleton {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                resolve: 0.0,
                ink_alpha: 1.0,
                count: 0.0,
                css: [1.0, 1.0],
                dev: [1.0, 1.0],
            }),
            clock: Cell::new(0.0),
        }
    }

    /// `init` — acquire adapter/device, configure the canvas, compile
    /// the pipeline, allocate the uniform + sheet buffers. Rejection
    /// propagates to JS, which falls back to the WebGL/CSS skeleton.
    ///
    /// @param canvas The dedicated canvas this surface paints.
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a configured surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // Layout: binding 0 uniform, binding 1 read-only storage sheet
        // — identical to the WGSL's @binding declarations.
        let layout = gpu_mini::new_layout(
            &surface.device,
            &[gpu_mini::uniform_entry(0), gpu_mini::storage_entry(1)],
        )?;

        let pipeline = gpu_mini::fullscreen_pipeline(
            &surface.device,
            surface.format,
            WGSL,
            &[layout.clone()],
        )?;

        // 256 B covers the 48 B block with alignment headroom.
        let ubuf = gpu_mini::new_buffer(
            &surface.device,
            256,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        )?;

        // The sheet: 24 rects × 4 vec4s × 16 B = 1536 B.
        let sheet = gpu_mini::new_buffer(
            &surface.device,
            (SHEET_VEC4S * 16) as u32,
            gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
        )?;

        let bind = gpu_mini::new_bind_group(
            &surface.device,
            &layout,
            &[
                gpu_mini::bind_buffer_entry(0, &ubuf),
                gpu_mini::bind_buffer_entry(1, &sheet),
            ],
        );

        // try_borrow everywhere: a JS exception that crossed a wasm
        // frame mid-call leaks the RefCell guard (wasm has no
        // unwinding) — a later borrow() then panics `unreachable`.
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };
        *slot = Some(Gpu {
            surface,
            pipeline,
            ubuf,
            sheet,
            bind,
        });

        Ok(())
    }

    /// `resize` — re-configure the surface after a canvas-bitmap
    /// change + store the DEVICE-pixel size as the shader's `rect`
    /// (the whole-canvas clip, in `frag`'s device units).
    ///
    /// @param w Device-pixel width (element CSS px × DPR).
    /// @param h Device-pixel height.
    pub fn resize(&self, w: u32, h: u32) {
        let Ok(mut s) = self.state.try_borrow_mut() else {
            return;
        };

        s.dev = [(w.max(1)) as f32, (h.max(1)) as f32];
        drop(s);

        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        if let Some(g) = gpu.as_ref() {
            gpu_mini::resize(&g.surface);
        }
    }

    /// `set_size` — the canvas's CSS-pixel size. Feeds the shader's
    /// `res` field (the widget rect IS the whole canvas — device-px
    /// `rect` is written by `resize` instead).
    ///
    /// @param w Canvas width in CSS pixels.
    /// @param h Canvas height in CSS pixels.
    pub fn set_size(&self, w: f32, h: f32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.css = [w.max(1.0), h.max(1.0)];
        }
    }

    /// `write_sheet` — upload the per-rect data block: 16 f32s per
    /// rect — [x,y,w,h][radius,cell,row_h,unused][sbase rgb,unused]
    /// [sink rgb,unused]. Truncated at MAX_RECTS rows; `count` (set
    /// via `set_state`) decides how many rows the GPU reads.
    ///
    /// @param values Flat f32 sheet — 16 values per rect row.
    pub fn write_sheet(&self, values: &[f32]) {
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        if let Some(g) = gpu.as_ref() {
            gpu_mini::write_f32s(&g.surface.queue, &g.sheet, values);
        }
    }

    /// `set_state` — scalar updates bundled to keep the JS call count
    /// low: resolve progress, ink alpha, live rect count.
    ///
    /// @param resolve 0..1 resolve progress (collapse+fade driver).
    /// @param ink_alpha Global glyph opacity multiplier.
    /// @param count Live rect count, clamped to MAX_RECTS.
    pub fn set_state(&self, resolve: f32, ink_alpha: f32, count: f32) {
        let Ok(mut s) = self.state.try_borrow_mut() else {
            return;
        };

        s.resolve = resolve.clamp(0.0, 1.0);
        s.ink_alpha = ink_alpha.clamp(0.0, 1.0);
        s.count = count.clamp(0.0, MAX_RECTS as f32);
    }

    /// `frame` — one animation tick. Per-frame cost: one ~64 B uniform
    /// write + one pass + one 3-vertex draw. The fragment loop early-
    /// outs per rect — bounded and cheap.
    ///
    /// @param dt Seconds since the previous `frame` call.
    pub fn frame(&self, dt: f32) {
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        let Some(g) = gpu.as_ref() else {
            return;
        };

        self.clock.set(self.clock.get() + dt);

        let Ok(s) = self.state.try_borrow() else {
            return;
        };

        // Uniform block — field-for-field the WGSL `Uniforms` struct:
        //   [0,1] res (CSS px)   [2] time   [3] resolve   [4] ink_alpha
        //   [5] count   [6,7] pad  [8..11] rect=(0,0,dev) DEVICE px —
        //   the sheet rows JS uploads are device px too (CSS × DPR).
        let uniforms: [f32; UNIFORM_F32S] = [
            s.css[0],
            s.css[1],
            self.clock.get(),
            s.resolve,
            s.ink_alpha,
            s.count,
            0.0,
            0.0,
            0.0,
            0.0,
            s.dev[0],
            s.dev[1],
        ];

        drop(s);

        gpu_mini::write_f32s(&g.surface.queue, &g.ubuf, &uniforms);

        if !gpu_mini::draw(&g.surface, &g.pipeline, &g.bind, None) {
            // Lost/stale surface (resize mid-frame, occluded tab) —
            // re-configure so the next rAF recovers.
            gpu_mini::resize(&g.surface);
        }
    }
}

/// `Drop` — forget the GPU state instead of destructing it: the JS
/// FinalizationRegistry's `__wbg_skeleton_free` would otherwise walk
/// every stored JsValue's externref dealloc — on a torn-down table
/// that walk traps `memory access out of bounds` during GC. GPU
/// objects die with the shared device; leaking the handles is both
/// safe and un-panickable.
impl Drop for Skeleton {
    fn drop(&mut self) {
        if let Some(g) = self.gpu.get_mut().take() {
            std::mem::forget(g);
        }
    }
}
