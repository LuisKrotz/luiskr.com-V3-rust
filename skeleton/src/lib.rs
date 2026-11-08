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
//! # SYNC CONTRACT — the layout is mirrored JS-side
//!
//! The 12-f32 uniform layout and the 24×4-vec4 sheet are mirrored by
//! `core/tokens/motion/gpu-widgets.ts` and the JS bridge
//! `core/utils/gpu/webgpu-skeleton.ts` — bump all together. (The old
//! `rust/overlay` twin copy no longer exists — this file is canonical.)
//!
//! # Frame loop
//!
//! JS owns the clock: the bridge calls `frame(dt)` per rAF. Per frame
//! cost: one ~64 B uniform write, one render pass, one 3-vertex draw —
//! zero allocation in the hot path.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{gpu_buffer_usage, GpuDevice, HtmlCanvasElement};

/// WGSL — the "data decoding" glyph field (fragment chunk; the shared
/// sd_round_box + fullscreen vs_main arrive via the gpu-mini library).
const WGSL: &str = include_str!("skeleton.wgsl");

/// MAX_RECTS — mirrors `SKELETON_RENDER.MAX_RECTS` (24) in
/// `core/tokens/motion/skeleton.ts`. The shader's `MAX_RECTS`, this
/// buffer's vec4 count, and the JS sheet-write size must all agree.
const MAX_RECTS: usize = 24;

/// vec4 fields per rect row — [rect_px, row_meta, base_rgba, ink_rgba]
/// × MAX_RECTS.
const SHEET_VEC4S: usize = MAX_RECTS * 4;

/// Uniform f32 count — resolution_css(2) time_seconds resolve_amount
/// ink_alpha rect_count reserved(2) clip_rect(4) = 12; mirrored by
/// `GPU_WIDGET_UNIFORM_F32S[SKELETON]`.
const UNIFORM_F32S: usize = 12;

/// GPU state — created once in `init`, never re-allocated (the frame-
/// budget rule: steady frames must not allocate GPU objects).
struct Gpu {
    /// The live canvas surface from gpu-mini.
    surface: gpu_mini::Surface,
    /// The skeleton pipeline — fullscreen triangle + storage sheet.
    pipeline: web_sys::GpuRenderPipeline,
    /// Uniform block — resolution/time/resolve/alpha/count/rect, 256 B.
    uniform_buffer: web_sys::GpuBuffer,
    /// Per-rect sheet — read-only storage, MAX_RECTS×4 vec4s.
    tile_sheet: web_sys::GpuBuffer,
    /// Pre-built bind group — valid for the pipeline's lifetime.
    bind_group: web_sys::GpuBindGroup,
}

/// Scalar state repacked into the uniform block each `frame`.
struct State {
    /// Resolve progress 0..1 — drives the glyph→fill collapse + fade.
    resolve_amount: f32,
    /// Global glyph opacity multiplier (theme-driven).
    ink_alpha: f32,
    /// Live rect count (≤ MAX_RECTS).
    rect_count: f32,
    /// Canvas CSS px — the `resolution_css` field (design-unit
    /// domain; the shader currently uses `clip_rect` for clipping and
    /// `resolution_css` only in documentation, but the field is packed
    /// for contract parity).
    css_size: [f32; 2],
    /// Canvas DEVICE px — the `clip_rect` field: `pixel_coord` arrives
    /// in device px and the shader compares sheet rows (DEVICE px too —
    /// the JS side uploads CSS geometry × DPR) against it, so both
    /// operands must share the unit.
    device_size: [f32; 2],
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
    clock_seconds: Cell<f32>,
}

#[wasm_bindgen]
impl Skeleton {
    /// `Skeleton::new` — un-initialized surface; `init` must follow.
    /// Split because the device request is an async browser promise.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Skeleton {
        // Pre-init state: no GPU yet, unresolved (0) progress, no live
        // rects; 1.0px sizes keep the shader divisors non-zero until
        // JS pushes real metrics.
        Skeleton {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                resolve_amount: 0.0,
                ink_alpha: 1.0,
                rect_count: 0.0,
                css_size: [1.0, 1.0],
                device_size: [1.0, 1.0],
            }),
            clock_seconds: Cell::new(0.0),
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
        // Step 1 — wrap the canvas in a configured surface.
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // Step 2 — bind-group layout: binding 0 uniform, binding 1
        // read-only storage sheet — identical to the WGSL's @binding
        // declarations.
        let layout = gpu_mini::new_layout(
            &surface.device,
            &[gpu_mini::uniform_entry(0), gpu_mini::storage_entry(1)],
        )?;

        // Step 3 — compile the glyph-field pipeline. The shader is a
        // fragment chunk, so it takes the shared fullscreen source
        // (helpers + triangle vs_main).
        let source = gpu_mini::wgsl::fullscreen_source(WGSL);
        let pipeline = gpu_mini::fullscreen_pipeline(
            &surface.device,
            surface.format,
            &source,
            &[layout.clone()],
        )?;

        // 256 B covers the 48 B block with alignment headroom.
        let uniform_buffer = gpu_mini::new_buffer(
            &surface.device,
            256,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        )?;

        // The sheet: 24 rects × 4 vec4s × 16 B = 1536 B.
        let tile_sheet = gpu_mini::new_buffer(
            &surface.device,
            (SHEET_VEC4S * 16) as u32,
            gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
        )?;

        // Step 5 — bind both buffers once; the sheet is fixed-size so
        // the group stays valid for the surface's whole lifetime.
        let bind_group = gpu_mini::new_bind_group(
            &surface.device,
            &layout,
            &[
                gpu_mini::bind_buffer_entry(0, &uniform_buffer),
                gpu_mini::bind_buffer_entry(1, &tile_sheet),
            ],
        );

        // try_borrow everywhere: a JS exception that crossed a wasm
        // frame mid-call leaks the RefCell guard (wasm has no
        // unwinding) — a later borrow() then panics `unreachable`.
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };
        // Publish the assembled GPU bundle — `frame` may now borrow it.
        *slot = Some(Gpu {
            surface,
            pipeline,
            uniform_buffer,
            tile_sheet,
            bind_group,
        });

        Ok(())
    }

    /// `resize` — re-configure the surface after a canvas-bitmap
    /// change + store the DEVICE-pixel size as the shader's
    /// `clip_rect` (the whole-canvas clip, in `pixel_coord`'s device
    /// units).
    ///
    /// @param width Device-pixel width (element CSS px × DPR).
    /// @param height Device-pixel height.
    pub fn resize(&self, width: u32, height: u32) {
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };

        // Store the device-px size (clip_rect.zw); clamped ≥1 so the
        // frag→sheet compare never sees a zero-size canvas.
        state.device_size = [(width.max(1)) as f32, (height.max(1)) as f32];

        // Release the state guard BEFORE the gpu borrow below — one
        // RefCell borrow at a time per the module-wide try_borrow rule.
        drop(state);

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

    /// `set_size` — the canvas's CSS-pixel size. Feeds the shader's
    /// `resolution_css` field (the widget rect IS the whole canvas —
    /// device-px `clip_rect` is written by `resize` instead).
    ///
    /// @param width Canvas width in CSS pixels.
    /// @param height Canvas height in CSS pixels.
    pub fn set_size(&self, width: f32, height: f32) {
        // CSS-px pair — same ≥1 clamp; a contended slot just drops
        // the update (the next resize carries the newer size).
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.css_size = [width.max(1.0), height.max(1.0)];
        }
    }

    /// `write_sheet` — upload the per-rect data block: 16 f32s per
    /// rect — [x,y,w,h][radius,cell,row_h,unused][base rgb,unused]
    /// [ink rgb,unused]. Truncated at MAX_RECTS rows; `rect_count`
    /// (set via `set_state`) decides how many rows the GPU reads.
    ///
    /// @param values Flat f32 sheet — 16 values per rect row.
    pub fn write_sheet(&self, values: &[f32]) {
        // Bail while init is in flight.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Upload the flat sheet into the storage buffer — JS truncates
        // at MAX_RECTS rows before calling (write_buffer would reject
        // an over-capacity write anyway).
        if let Some(gpu_state) = gpu.as_ref() {
            gpu_mini::write_f32s(&gpu_state.surface.queue, &gpu_state.tile_sheet, values);
        }
    }

    /// `set_state` — scalar updates bundled to keep the JS call count
    /// low: resolve progress, ink alpha, live rect count.
    ///
    /// @param resolve_amount 0..1 resolve progress (collapse+fade driver).
    /// @param ink_alpha Global glyph opacity multiplier.
    /// @param rect_count Live rect count, clamped to MAX_RECTS.
    pub fn set_state(&self, resolve_amount: f32, ink_alpha: f32, rect_count: f32) {
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };

        // Clamp all three scalars into their domains: 0..1 progress,
        // 0..1 alpha, 0..MAX_RECTS live rows.
        state.resolve_amount = resolve_amount.clamp(0.0, 1.0);

        state.ink_alpha = ink_alpha.clamp(0.0, 1.0);

        state.rect_count = rect_count.clamp(0.0, MAX_RECTS as f32);
    }

    /// `frame` — one animation tick. Per-frame cost: one ~64 B uniform
    /// write + one pass + one 3-vertex draw. The fragment loop early-
    /// outs per rect — bounded and cheap.
    ///
    /// @param delta_seconds Seconds since the previous `frame` call.
    pub fn frame(&self, delta_seconds: f32) {
        // Bail while init is in flight — a contended slot skips a tick.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Still `None` → init hasn't resolved; nothing to draw yet.
        let Some(gpu_state) = gpu.as_ref() else {
            return;
        };

        // Advance the glyph cadence clock — summing dt absorbs rAF
        // jitter.
        self.clock_seconds
            .set(self.clock_seconds.get() + delta_seconds);

        // Snapshot the scalar state under the guard…
        let Ok(state) = self.state.try_borrow() else {
            return;
        };

        // Uniform block — field-for-field the WGSL `Uniforms` struct:
        //   [0,1] resolution_css (CSS px) [2] time_seconds
        //   [3] resolve_amount [4] ink_alpha [5] rect_count
        //   [6,7] reserved [8..11] clip_rect=(0,0,device) DEVICE px —
        //   the sheet rows JS uploads are device px too (CSS × DPR).
        let uniforms: [f32; UNIFORM_F32S] = [
            state.css_size[0],
            state.css_size[1],
            self.clock_seconds.get(),
            state.resolve_amount,
            state.ink_alpha,
            state.rect_count,
            0.0,
            0.0,
            0.0,
            0.0,
            state.device_size[0],
            state.device_size[1],
        ];

        // Release the state borrow BEFORE the GPU calls — no RefCell
        // guard may be held across the JS-boundary work.
        drop(state);

        // Push the packed block to the GPU…
        gpu_mini::write_f32s(&gpu_state.surface.queue, &gpu_state.uniform_buffer, &uniforms);

        // …then the shared acquire→pass→submit body.
        if !gpu_mini::draw(&gpu_state.surface, &gpu_state.pipeline, &gpu_state.bind_group, None) {
            // Lost/stale surface (resize mid-frame, occluded tab) —
            // re-configure so the next rAF recovers.
            gpu_mini::resize(&gpu_state.surface);
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
        if let Some(gpu_state) = self.gpu.get_mut().take() {
            std::mem::forget(gpu_state);
        }
    }
}
