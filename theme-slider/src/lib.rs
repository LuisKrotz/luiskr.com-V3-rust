//! theme-slider — the day/system/night preference slider.
//!
//! Standalone per-widget artifact — see `rust/README.md`. The WGSL is
//! the largest widget shader (a full procedural scene: sky gradient,
//! starfield, sun+corona, aurora, crescent moon, two parallax mesas,
//! 3D knob); `progress` 0..2 morphs night → system → day.
//!
//! # Uniform layout (f32s = 12 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[THEME_SLIDER]` and the WGSL struct)
//!
//!   [0,1]  res         canvas CSS px (aspect-corrected scene domain)
//!   [2]    time        seconds — star twinkle, aurora, corona, shimmer
//!   [3]    progress    0..2 — 0..1 night→system, 1..2 system→day
//!   [4]    knob_x      knob center X in px (JS spring)
//!   [5]    ripple_time seconds-stamp of last tap (−1 = none)
//!   [6]    ripple_pos  tap X in px → expanding ring origin
//!   [7]    pad         16B alignment pad
//!   [8..11] rect       (0,0,w,h) on the dedicated canvas
//!
//! # Frame cost
//!
//! One ~48 B uniform write + one 3-vertex draw; the shader is the
//! heaviest of the widgets (~80 hash/noise evals/px worst case) but
//! the canvas is small.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

/// WGSL — the procedural day/night sky scene.
const WGSL: &str = include_str!("theme_slider.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 12;

/// Slider state repacked into the uniform block per frame.
struct State {
    /// Canvas CSS px — the `res` field (design-unit domain).
    css: [f32; 2],
    /// Canvas DEVICE px — the `rect.zw` divisor mapping `frag`
    /// (device px) into widget uv 0..1. Written by `resize`.
    dev: [f32; 2],
    /// 0..2 theme morph (JS eases; clamp keeps the shader's two
    /// mix() arms well-defined).
    progress: f32,
    /// Knob center X in px — JS owns the spring.
    knob_x: f32,
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
    clock: Cell<f32>,
}

#[wasm_bindgen]
impl ThemeSlider {
    /// `ThemeSlider::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> ThemeSlider {
        ThemeSlider {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                css: [1.0, 1.0],
                dev: [1.0, 1.0],
                progress: 0.0,
                knob_x: 0.0,
                ripple_time: -1.0,
                ripple_pos: 0.0,
            }),
            clock: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline. Rejection → JS keeps the WebGL
    /// slider; mounting can never regress the page.
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

    /// `set_state` — bundled scalar update: eased progress + knob X.
    ///
    /// @param progress 0..2 theme morph (0..1 night→system,
    ///                 1..2 system→day; clamped to the shader's domain).
    /// @param knob_x Knob center X in px.
    pub fn set_state(&self, progress: f32, knob_x: f32) {
        let mut s = self.state.borrow_mut();

        s.progress = progress.clamp(0.0, 2.0);
        s.knob_x = knob_x;
    }

    /// `tap` — registers a tap for the expanding-ring ripple: the JS
    /// bridge passes its clock time + the tap's X position.
    ///
    /// @param time The frame clock's current seconds (matches what
    ///              `frame` accumulates — the shader compares
    ///              `u.time - ripple_time`).
    /// @param x Tap X in canvas CSS px.
    pub fn tap(&self, time: f32, x: f32) {
        let mut s = self.state.borrow_mut();

        s.ripple_time = time;
        s.ripple_pos = x;
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

        // [0,1] res [2] time [3] progress [4] knob_x
        // [5] ripple_time [6] ripple_pos [7] pad [8..11] rect
        let uniforms: [f32; UNIFORM_F32S] = [
            s.css[0],
            s.css[1],
            self.clock.get(),
            s.progress,
            s.knob_x,
            s.ripple_time,
            s.ripple_pos,
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
