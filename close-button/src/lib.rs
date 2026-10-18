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
//!   [0,1]  res        canvas CSS px (button diameter)
//!   [2]    time       seconds — wave phase, bubbles, shockwave clock
//!   [3]    liquid     0..1 fill height (JS eases on hover)
//!   [4]    draw       0..1 stroke progress for the self-drawing X
//!   [5]    rot        X rotation angle, rad (JS advances slowly)
//!   [6]    click_time seconds-stamp of last click (−1 = none)
//!   [7]    pad        16B alignment pad
//!   [8..11] rect      (0,0,w,h) on the dedicated canvas
//!
//! # Frame cost
//!
//! One ~48 B uniform write + one 3-vertex draw.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

/// WGSL — liquid-fill circular close button.
const WGSL: &str = include_str!("close_button.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 12;

/// Button state repacked into the uniform block per frame.
struct State {
    /// Canvas CSS px — the `res` field (design-unit domain).
    css: [f32; 2],
    /// Canvas DEVICE px — the `rect.zw` divisor mapping `frag`
    /// (device px) into widget uv 0..1. Written by `resize`.
    dev: [f32; 2],
    /// 0..1 liquid fill height (JS eases on hover).
    liquid: f32,
    /// 0..1 X stroke progress (JS eases on mount/hover).
    draw: f32,
    /// X rotation angle in radians (JS advances slowly).
    rot: f32,
    /// Seconds-stamp of the last click (−1 = none — shockwave inert).
    click_time: f32,
}

/// Public wasm surface — one button canvas per instance.
#[wasm_bindgen]
pub struct CloseButton {
    /// GPU scaffold — `None` until `init` resolves.
    gpu: RefCell<Option<gpu_mini::Widget>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Accumulated seconds — absorbs rAF jitter.
    clock: Cell<f32>,
}

#[wasm_bindgen]
impl CloseButton {
    /// `CloseButton::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> CloseButton {
        CloseButton {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                css: [1.0, 1.0],
                dev: [1.0, 1.0],
                liquid: 0.0,
                draw: 0.0,
                rot: 0.0,
                click_time: -1.0,
            }),
            clock: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline. Rejection → JS keeps the WebGL
    /// button; mounting can never regress the page.
    ///
    /// @param canvas The widget's dedicated canvas.
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub async fn init(&self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
        *self.gpu.borrow_mut() = Some(gpu_mini::init_widget(&canvas, WGSL).await?);
        Ok(())
    }

    /// `resize` — canvas-bitmap change → surface re-configure.
    ///
    /// @param _w Device-pixel width (element-side).
    /// @param _h Device-pixel height.
    pub fn resize(&self, w: u32, h: u32) {
        self.state.borrow_mut().dev = [(w.max(1)) as f32, (h.max(1)) as f32];

        let gpu = self.gpu.borrow();

        if let Some(w) = gpu.as_ref() {
            gpu_mini::resize(&w.surface);
        }
    }

    /// `set_size` — the canvas's CSS-pixel size (res + rect).
    ///
    /// @param w Canvas width in CSS pixels.
    /// @param h Canvas height in CSS pixels.
    pub fn set_size(&self, w: f32, h: f32) {
        self.state.borrow_mut().css = [w.max(1.0), h.max(1.0)];
    }

    /// `set_state` — bundled scalar update: liquid fill, X stroke
    /// progress, X rotation. All eased JS-side; the shader just draws.
    ///
    /// @param liquid 0..1 fill height.
    /// @param draw 0..1 X stroke progress.
    /// @param rot X rotation in radians.
    pub fn set_state(&self, liquid: f32, draw: f32, rot: f32) {
        let mut s = self.state.borrow_mut();

        s.liquid = liquid.clamp(0.0, 1.0);
        s.draw = draw.clamp(0.0, 1.0);
        s.rot = rot;
    }

    /// `click` — registers a click for the shockwave ring: the bridge
    /// passes the frame clock's current seconds (the shader computes
    /// `time - click_time` for ring expansion/fade).
    ///
    /// @param time The frame clock's current seconds.
    pub fn click(&self, time: f32) {
        self.state.borrow_mut().click_time = time;
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

        // [0,1] res [2] time [3] liquid [4] draw [5] rot
        // [6] click_time [7] pad [8..11] rect = (0,0,w,h)
        let uniforms: [f32; UNIFORM_F32S] = [
            s.css[0],
            s.css[1],
            self.clock.get(),
            s.liquid,
            s.draw,
            s.rot,
            s.click_time,
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
