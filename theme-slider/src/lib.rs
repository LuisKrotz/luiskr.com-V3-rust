//! theme-slider — the day/system/night preference slider.
//!
//! Standalone per-widget artifact — see `rust/README.md`. The WGSL is
//! the largest widget shader (a full procedural scene: sky gradient,
//! starfield, sun+corona, aurora, crescent moon, two parallax mesas,
//! 3D knob); `theme_progress` 0..2 morphs night → system → day.
//!
//! # Uniform layout (f32s = 12 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[THEME_SLIDER]` and the WGSL struct)
//!
//!   [0,1]  resolution_css canvas CSS px (aspect-corrected scene domain)
//!   [2]    time_seconds   seconds — star twinkle, aurora, corona, shimmer
//!   [3]    theme_progress 0..2 — 0..1 night→system, 1..2 system→day
//!   [4]    knob_center_x  knob center X in px (JS spring)
//!   [5]    ripple_time    seconds-stamp of last tap (−1 = none)
//!   [6]    ripple_pos     tap X in px → expanding ring origin
//!   [7]    reserved       16B alignment pad
//!   [8..11] clip_rect     (0,0,w,h) on the dedicated canvas
//!
//! # Frame cost
//!
//! One ~48 B uniform write + one 3-vertex draw; the shader is the
//! heaviest of the widgets (~80 hash/noise evals/px worst case) but
//! the canvas is small.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{GpuDevice, HtmlCanvasElement};

/// WGSL — the procedural day/night sky scene fragment chunk (shared
/// dist_to_segment + vs_main arrive via the gpu-mini library; the
/// star hash stays local — a different seed would re-roll the sky).
const WGSL: &str = include_str!("theme_slider.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 12;

/// Slider state repacked into the uniform block per frame.
struct State {
    /// Canvas CSS px — the `resolution_css` field (design-unit domain).
    css_size: [f32; 2],
    /// Canvas DEVICE px — the `clip_rect.zw` divisor mapping
    /// `pixel_coord` (device px) into widget uv 0..1. Written by `resize`.
    device_size: [f32; 2],
    /// 0..2 theme morph (JS eases; clamp keeps the shader's two
    /// mix() arms well-defined).
    theme_progress: f32,
    /// Knob center X in px — JS owns the spring.
    knob_center_x: f32,
    /// Seconds-stamp of the last tap (−1 = none — the shader's
    /// `>= 0` guard keeps it inert).
    ripple_time: f32,
    /// Tap X in px — ripple ring origin.
    ripple_pos: f32,
}

/// Public wasm surface — one slider canvas per instance.
#[wasm_bindgen]
pub struct ThemeSlider {
    /// GPU scaffold — `None` until `init` resolves.
    gpu: RefCell<Option<gpu_mini::Widget>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Accumulated seconds — absorbs rAF jitter.
    clock_seconds: Cell<f32>,
}

#[wasm_bindgen]
impl ThemeSlider {
    /// `ThemeSlider::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> ThemeSlider {
        // Pre-init state: no GPU yet; night scene (progress 0), knob at
        // the left edge, ripple_time = −1 (no tap recorded, so the
        // ring stays dormant).
        ThemeSlider {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                css_size: [1.0, 1.0],
                device_size: [1.0, 1.0],
                theme_progress: 0.0,
                knob_center_x: 0.0,
                ripple_time: -1.0,
                ripple_pos: 0.0,
            }),
            clock_seconds: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline. Rejection → JS keeps the WebGL
    /// slider; mounting can never regress the page.
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

    /// `resize` — canvas-bitmap change → surface re-configure.
    ///
    /// @param width Device-pixel width (element-side).
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
    /// clip_rect).
    ///
    /// @param width Canvas width in CSS pixels.
    /// @param height Canvas height in CSS pixels.
    pub fn set_size(&self, width: f32, height: f32) {
        // CSS-px pair — ≥1 clamp; `resolution_css` is the aspect-
        // corrected scene domain the shader projects from.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.css_size = [width.max(1.0), height.max(1.0)];
        }
    }

    /// `set_state` — bundled scalar update: eased progress + knob X.
    ///
    /// @param theme_progress 0..2 theme morph (0..1 night→system,
    ///   1..2 system→day; clamped to the shader's domain).
    /// @param knob_center_x Knob center X in px.
    pub fn set_state(&self, theme_progress: f32, knob_center_x: f32) {
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };

        // Clamp the theme morph into 0..2 — the shader's two mix()
        // arms are only defined inside it.
        state.theme_progress = theme_progress.clamp(0.0, 2.0);

        // Knob position — verbatim; JS owns the spring math.
        state.knob_center_x = knob_center_x;
    }

    /// `tap` — registers a tap for the expanding-ring ripple: the JS
    /// bridge passes its clock time + the tap's X position.
    ///
    /// @param time_seconds The frame clock's current seconds (matches
    ///   what `frame` accumulates — the shader compares
    ///   `time_seconds - ripple_time`).
    /// @param tap_x Tap X in canvas CSS px.
    pub fn tap(&self, time_seconds: f32, tap_x: f32) {
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };

        // Stamp the tap on the frame clock + its X position — the
        // shader expands the ring from `time − ripple_time` outward
        // from `ripple_pos`.
        state.ripple_time = time_seconds;

        state.ripple_pos = tap_x;
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

        // Advance the scene clock — summing dt absorbs rAF jitter.
        self.clock_seconds
            .set(self.clock_seconds.get() + delta_seconds);

        // Snapshot the scalar state under the guard…
        let Ok(state) = self.state.try_borrow() else {
            return;
        };

        // [0,1] resolution_css [2] time_seconds [3] theme_progress
        // [4] knob_center_x [5] ripple_time [6] ripple_pos
        // [7] reserved [8..11] clip_rect
        let uniforms: [f32; UNIFORM_F32S] = [
            state.css_size[0],
            state.css_size[1],
            self.clock_seconds.get(),
            state.theme_progress,
            state.knob_center_x,
            state.ripple_time,
            state.ripple_pos,
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
impl Drop for ThemeSlider {
    fn drop(&mut self) {
        if let Some(gpu_state) = self.gpu.get_mut().take() {
            std::mem::forget(gpu_state);
        }
    }
}
