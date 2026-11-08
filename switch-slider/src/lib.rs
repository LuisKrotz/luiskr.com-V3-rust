//! switch-slider — the preference toggle (pill track + knob).
//!
//! Standalone per-widget artifact — see `rust/README.md`. The shader
//! works in RAW PIXELS (the original was a fixed 28px-tall canvas):
//! `resolution_css` must be the canvas's actual CSS size, and the
//! canvas should match the design size — pixel math does NOT scale.
//!
//! # Uniform layout (f32s = 12 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[SWITCH_SLIDER]` and the WGSL struct)
//!
//!   [0,1]  resolution_css canvas CSS px (track: 14px tall, radius 14)
//!   [2]    time_seconds   seconds — wave/streak animation
//!   [3]    switch_progress 0..1 off→on (JS eases; shader lerps)
//!   [4]    knob_center_x  knob center X in px (JS owns the spring)
//!   [5]    context_id     0=stats 1=grid 2=motion — picks color+pattern
//!   [6,7]  reserved       16B alignment pad
//!   [8..11] clip_rect     (0,0,w,h) on the dedicated canvas
//!
//! # Frame cost
//!
//! One ~48 B uniform write + one 3-vertex draw.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{GpuDevice, HtmlCanvasElement};

/// WGSL — pill track + knob + per-context waveform fragment chunk
/// (shared dist_to_segment + vs_main arrive via the gpu-mini library).
const WGSL: &str = include_str!("switch_slider.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 12;

/// Toggle state repacked into the uniform block per frame.
struct State {
    /// Canvas CSS px — the `resolution_css` field (pixel domain).
    css_size: [f32; 2],
    /// Canvas DEVICE px — the `clip_rect.zw` divisor mapping
    /// `pixel_coord` (device px) into widget uv 0..1. Written by `resize`.
    device_size: [f32; 2],
    /// 0..1 off→on (JS eases before passing).
    switch_progress: f32,
    /// Knob center X in px — JS owns the spring curve.
    knob_center_x: f32,
    /// Track personality: 0 stats / 1 grid / 2 motion.
    context_id: f32,
}

/// Public wasm surface — one toggle canvas per instance.
#[wasm_bindgen]
pub struct SwitchSlider {
    /// GPU scaffold — `None` until `init` resolves.
    gpu: RefCell<Option<gpu_mini::Widget>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Accumulated seconds — absorbs rAF jitter.
    clock_seconds: Cell<f32>,
}

#[wasm_bindgen]
impl SwitchSlider {
    /// `SwitchSlider::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> SwitchSlider {
        // Pre-init state: no GPU yet; off position (progress 0, knob
        // at the 14px rest spot), stats context — `set_state` pushes
        // the real values before the first frame.
        SwitchSlider {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                css_size: [1.0, 1.0],
                device_size: [1.0, 1.0],
                switch_progress: 0.0,
                knob_center_x: 14.0,
                context_id: 0.0,
            }),
            clock_seconds: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline. Rejection → JS keeps the WebGL
    /// toggle; mounting can never regress the page.
    ///
    /// @param canvas The widget's dedicated canvas.
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        // try_borrow everywhere: a JS exception crossing a wasm frame
        // mid-call leaks the RefCell guard — a later borrow() panics
        // `unreachable` (the observed widget-crash cascade). Build the
        // GPU state BEFORE taking the guard so the fallible JS-boundary
        // calls can never run while the borrow is held.
        let built = gpu_mini::init_widget(&canvas, &device, WGSL)?;

        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };

        // Publish the live scaffold — `frame` may now borrow it.
        *slot = Some(built);

        Ok(())
    }

    /// `resize` — canvas-bitmap change → surface re-configure + the
    /// DEVICE-pixel size that becomes `clip_rect.zw` (the frag→uv
    /// divisor; `pixel_coord` arrives in device px, so the divisor
    /// must too).
    ///
    /// @param width Device-pixel width (element CSS px × DPR).
    /// @param height Device-pixel height.
    pub fn resize(&self, width: u32, height: u32) {
        // Store the device-px size (clip_rect.zw); clamped ≥1 so the
        // frag→uv divisor never sees a zero-sized canvas.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.device_size = [(width.max(1)) as f32, (height.max(1)) as f32];
        }

        // Bail while init is in flight — the surface configures
        // itself when it arrives.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Re-configure the live surface to match the new bitmap.
        if let Some(widget) = gpu.as_ref() {
            gpu_mini::resize(&widget.surface);
        }
    }

    /// `set_size` — the canvas's CSS-pixel size (resolution_css +
    /// clip_rect). NOTE: the shader's pixel-space math assumes the
    /// design height (~28px) — mount the canvas at the design size.
    ///
    /// @param width Canvas width in CSS pixels.
    /// @param height Canvas height in CSS pixels.
    pub fn set_size(&self, width: f32, height: f32) {
        // CSS-px pair — ≥1 clamp; the shader's pixel math treats it
        // as the design-unit domain (mount at design size, see doc).
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.css_size = [width.max(1.0), height.max(1.0)];
        }
    }

    /// `set_state` — bundled scalar update: eased progress, knob X,
    /// and the context personality (0 stats / 1 grid / 2 motion —
    /// mirrored by `GPU_SWITCH_CONTEXTS` on the JS side).
    ///
    /// @param switch_progress 0..1 eased on-ness.
    /// @param knob_center_x Knob center X in px.
    /// @param context_id Track personality discriminant.
    pub fn set_state(&self, switch_progress: f32, knob_center_x: f32, context_id: f32) {
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };

        // Clamp the eased on-ness into 0..1.
        state.switch_progress = switch_progress.clamp(0.0, 1.0);

        // Knob position — verbatim; JS owns the spring math.
        state.knob_center_x = knob_center_x;

        // Personality discriminant — verbatim (0/1/2 by contract).
        state.context_id = context_id;
    }

    /// `frame` — one animation tick: pack, write, draw.
    ///
    /// @param delta_seconds Seconds since the previous call.
    pub fn frame(&self, delta_seconds: f32) {
        // Bail while init is in flight — a contended slot skips a tick.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Still `None` → init hasn't resolved; nothing to draw yet.
        let Some(widget) = gpu.as_ref() else {
            return;
        };

        // Advance the wave/streak clock — summing dt absorbs rAF jitter.
        self.clock_seconds
            .set(self.clock_seconds.get() + delta_seconds);

        // Snapshot the scalar state under the guard…
        let Ok(state) = self.state.try_borrow() else {
            return;
        };

        // [0,1] resolution_css [2] time_seconds [3] switch_progress
        // [4] knob_center_x [5] context_id [6,7] reserved
        // [8..11] clip_rect = (0,0,device_size) — DEVICE px divisor
        let uniforms: [f32; UNIFORM_F32S] = [
            state.css_size[0],
            state.css_size[1],
            self.clock_seconds.get(),
            state.switch_progress,
            state.knob_center_x,
            state.context_id,
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

        // Write + draw; a stale/lost surface re-configures itself so
        // the next frame recovers cleanly.
        if !gpu_mini::widget_frame(widget, &uniforms) {
            gpu_mini::resize(&widget.surface);
        }
    }
}

/// `destroy` parity — forget the GPU state instead of destructing
/// it: the FinalizationRegistry's `__wbg_*_free` walks every stored
/// JsValue's externref dealloc — on a torn-down table that traps
/// `memory access out of bounds` during GC. GPU objects die with the
/// shared device; leaking the handles is safe and un-panickable.
impl Drop for SwitchSlider {
    fn drop(&mut self) {
        if let Some(gpu_state) = self.gpu.get_mut().take() {
            std::mem::forget(gpu_state);
        }
    }
}
