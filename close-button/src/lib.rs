//! close-button — the modal close control.
//!
//! Standalone per-widget artifact — see `rust/README.md`. The shader
//! draws a circular button: liquid fill with foam crest rising on
//! hover, a self-drawing + slowly rotating X, bubbles, and a click
//! shockwave ring.
//!
//! # Uniform layout (f32s = 12 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[CLOSE_BUTTON]` and the WGSL struct)
//!
//!   [0,1]  resolution_css canvas CSS px (button diameter)
//!   [2]    time_seconds   seconds — wave phase, bubbles, shockwave clock
//!   [3]    liquid_fill    0..1 fill height (JS eases on hover)
//!   [4]    x_draw         0..1 stroke progress for the self-drawing X
//!   [5]    x_rotation     X rotation angle, rad (JS advances slowly)
//!   [6]    click_seconds  seconds-stamp of last click (−1 = none)
//!   [7]    reserved       16B alignment pad
//!   [8..11] clip_rect     (0,0,w,h) on the dedicated canvas
//!
//! # Frame cost
//!
//! One ~48 B uniform write + one 3-vertex draw.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{GpuDevice, HtmlCanvasElement};

/// WGSL — liquid-fill circular close button fragment chunk (shared
/// helpers + palette + vs_main arrive via the gpu-mini library).
const WGSL: &str = include_str!("close_button.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 12;

/// Button state repacked into the uniform block per frame.
struct State {
    /// Canvas CSS px — the `resolution_css` field (design-unit domain).
    css_size: [f32; 2],
    /// Canvas DEVICE px — the `clip_rect.zw` divisor mapping
    /// `pixel_coord` (device px) into widget uv 0..1. Written by `resize`.
    device_size: [f32; 2],
    /// 0..1 liquid fill height (JS eases on hover).
    liquid_fill: f32,
    /// 0..1 X stroke progress (JS eases on mount/hover).
    x_draw: f32,
    /// X rotation angle in radians (JS advances slowly).
    x_rotation: f32,
    /// Seconds-stamp of the last click (−1 = none — shockwave inert).
    click_seconds: f32,
}

/// Public wasm surface — one button canvas per instance.
#[wasm_bindgen]
pub struct CloseButton {
    /// GPU scaffold — `None` until `init` resolves.
    gpu: RefCell<Option<gpu_mini::Widget>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Accumulated seconds — absorbs rAF jitter.
    clock_seconds: Cell<f32>,
}

#[wasm_bindgen]
impl CloseButton {
    /// `CloseButton::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> CloseButton {
        // Pre-init state: no GPU surface yet; inert defaults — 1.0px
        // divisors, 0 progress scalars, click_seconds = −1 (no
        // shockwave recorded, so the ring stays dormant).
        CloseButton {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                css_size: [1.0, 1.0],
                device_size: [1.0, 1.0],
                liquid_fill: 0.0,
                x_draw: 0.0,
                x_rotation: 0.0,
                click_seconds: -1.0,
            }),
            clock_seconds: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline. Rejection → JS keeps the WebGL
    /// button; mounting can never regress the page.
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

        // Publish the live surface — `frame` may now borrow it.
        *slot = Some(built);

        Ok(())
    }

    /// `resize` — canvas-bitmap change → surface re-configure.
    ///
    /// @param width Device-pixel width (element-side).
    /// @param height Device-pixel height.
    pub fn resize(&self, width: u32, height: u32) {
        // Store the device-px size (clip_rect.zw); clamped ≥1 so the
        // frag→uv divisor is never zero — a contended state slot
        // just skips the write.
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
    /// clip_rect).
    ///
    /// @param width Canvas width in CSS pixels.
    /// @param height Canvas height in CSS pixels.
    pub fn set_size(&self, width: f32, height: f32) {
        // try-borrow — drop the update if the slot is contended; the
        // CSS-px pair is clamped ≥1 like the device pair above.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.css_size = [width.max(1.0), height.max(1.0)];
        }
    }

    /// `set_state` — bundled scalar update: liquid fill, X stroke
    /// progress, X rotation. All eased JS-side; the shader just draws.
    ///
    /// @param liquid_fill 0..1 fill height.
    /// @param x_draw 0..1 X stroke progress.
    /// @param x_rotation X rotation in radians.
    pub fn set_state(&self, liquid_fill: f32, x_draw: f32, x_rotation: f32) {
        // Bail on a contended slot — JS re-pushes eased values every
        // animation step anyway.
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };

        // Fill height + stroke progress are unit-interval eased — clamp
        // both; rotation is an unbounded angle, stored as-is.
        state.liquid_fill = liquid_fill.clamp(0.0, 1.0);

        state.x_draw = x_draw.clamp(0.0, 1.0);

        state.x_rotation = x_rotation;
    }

    /// `click` — registers a click for the shockwave ring: the bridge
    /// passes the frame clock's current seconds (the shader computes
    /// `time_seconds − click_seconds` for ring expansion/fade).
    ///
    /// @param time_seconds The frame clock's current seconds.
    pub fn click(&self, time_seconds: f32) {
        // Stamp the click on the frame clock — the shader expands the
        // shockwave ring from `time_seconds − click_seconds`.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.click_seconds = time_seconds;
        }
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

        // Advance the shader clock — summing dt absorbs rAF jitter.
        self.clock_seconds
            .set(self.clock_seconds.get() + delta_seconds);

        // Snapshot the scalar state under the guard…
        let Ok(state) = self.state.try_borrow() else {
            return;
        };

        // [0,1] resolution_css [2] time_seconds [3] liquid_fill
        // [4] x_draw [5] x_rotation [6] click_seconds [7] reserved
        // [8..11] clip_rect = (0,0,w,h)
        let uniforms: [f32; UNIFORM_F32S] = [
            state.css_size[0],
            state.css_size[1],
            self.clock_seconds.get(),
            state.liquid_fill,
            state.x_draw,
            state.x_rotation,
            state.click_seconds,
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
impl Drop for CloseButton {
    fn drop(&mut self) {
        if let Some(gpu_state) = self.gpu.get_mut().take() {
            std::mem::forget(gpu_state);
        }
    }
}
