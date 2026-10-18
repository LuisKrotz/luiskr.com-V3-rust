//! switch-slider — the preference toggle (pill track + knob).
//!
//! Standalone per-widget artifact — see `rust/README.md`. The shader
//! works in RAW PIXELS (the original was a fixed 28px-tall canvas):
//! `res` must be the canvas's actual CSS size, and the canvas should
//! match the design size — pixel math does NOT scale.
//!
//! # Uniform layout (f32s = 12 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[SWITCH_SLIDER]` and the WGSL struct)
//!
//!   [0,1]  res      canvas CSS px (track: 14px tall, radius 14)
//!   [2]    time     seconds — wave/streak animation
//!   [3]    progress 0..1 off→on (JS eases; shader lerps)
//!   [4]    knob_x   knob center X in px (JS owns the spring)
//!   [5]    context  0=stats 1=grid 2=motion — picks color+pattern
//!   [6,7]  pad      16B alignment pad
//!   [8..11] rect    (0,0,w,h) on the dedicated canvas
//!
//! # Frame cost
//!
//! One ~48 B uniform write + one 3-vertex draw.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

/// WGSL — pill track + knob + per-context waveform.
const WGSL: &str = include_str!("switch_slider.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 12;

/// Toggle state repacked into the uniform block per frame.
struct State {
    /// Canvas CSS px — the `res` field (design-unit pixel domain).
    css: [f32; 2],
    /// Canvas DEVICE px — the `rect.zw` divisor mapping `frag`
    /// (device px) into widget uv 0..1. Written by `resize`.
    dev: [f32; 2],
    /// 0..1 off→on (JS eases before passing).
    progress: f32,
    /// Knob center X in px — JS owns the spring curve.
    knob_x: f32,
    /// Track personality: 0 stats / 1 grid / 2 motion.
    context: f32,
}

/// Public wasm surface — one toggle canvas per instance.
#[wasm_bindgen]
pub struct SwitchSlider {
    /// GPU scaffold — `None` until `init` resolves.
    gpu: RefCell<Option<gpu_mini::Widget>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Accumulated seconds — absorbs rAF jitter.
    clock: Cell<f32>,
}

#[wasm_bindgen]
impl SwitchSlider {
    /// `SwitchSlider::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> SwitchSlider {
        SwitchSlider {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                css: [1.0, 1.0],
                dev: [1.0, 1.0],
                progress: 0.0,
                knob_x: 14.0,
                context: 0.0,
            }),
            clock: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline. Rejection → JS keeps the WebGL
    /// toggle; mounting can never regress the page.
    ///
    /// @param canvas The widget's dedicated canvas.
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub async fn init(&self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
        *self.gpu.borrow_mut() = Some(gpu_mini::init_widget(&canvas, WGSL).await?);
        Ok(())
    }

    /// `resize` — canvas-bitmap change → surface re-configure + the
    /// DEVICE-pixel size that becomes `rect.zw` (the frag→uv divisor;
    /// `frag` arrives in device px, so the divisor must too).
    ///
    /// @param w Device-pixel width (element CSS px × DPR).
    /// @param h Device-pixel height.
    pub fn resize(&self, w: u32, h: u32) {
        self.state.borrow_mut().dev = [(w.max(1)) as f32, (h.max(1)) as f32];

        let gpu = self.gpu.borrow();

        if let Some(w) = gpu.as_ref() {
            gpu_mini::resize(&w.surface);
        }
    }

    /// `set_size` — the canvas's CSS-pixel size (res + rect). NOTE: the
    /// shader's pixel-space math assumes the design height (~28px) —
    /// mount the canvas at the design size.
    ///
    /// @param w Canvas width in CSS pixels.
    /// @param h Canvas height in CSS pixels.
    pub fn set_size(&self, w: f32, h: f32) {
        self.state.borrow_mut().css = [w.max(1.0), h.max(1.0)];
    }

    /// `set_state` — bundled scalar update: eased progress, knob X,
    /// and the context personality (0 stats / 1 grid / 2 motion —
    /// mirrored by `GPU_SWITCH_CONTEXTS` on the JS side).
    ///
    /// @param progress 0..1 eased on-ness.
    /// @param knob_x Knob center X in px.
    /// @param context Track personality discriminant.
    pub fn set_state(&self, progress: f32, knob_x: f32, context: f32) {
        let mut s = self.state.borrow_mut();

        s.progress = progress.clamp(0.0, 1.0);
        s.knob_x = knob_x;
        s.context = context;
    }

    /// `frame` — one animation tick: pack, write, draw.
    ///
    /// @param dt Seconds since the previous call.
    pub fn frame(&self, dt: f32) {
        let gpu = self.gpu.borrow();

        let Some(w) = gpu.as_ref() else {
            return;
        };

        self.clock.set(self.clock.get() + dt);

        let s = self.state.borrow();

        // [0,1] res (CSS px) [2] time [3] progress [4] knob_x [5] context
        // [6,7] pad [8..11] rect = (0,0,dev) — DEVICE px, frag→uv divisor
        let uniforms: [f32; UNIFORM_F32S] = [
            s.css[0],
            s.css[1],
            self.clock.get(),
            s.progress,
            s.knob_x,
            s.context,
            0.0,
            0.0,
            0.0,
            0.0,
            s.dev[0],
            s.dev[1],
        ];

        drop(s);

        if !gpu_mini::widget_frame(w, &uniforms) {
            gpu_mini::resize(&w.surface);
        }
    }
}
