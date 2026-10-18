//! burger — the nav burger's animated hairline bars.
//!
//! Standalone per-widget artifact: own GPUDevice, own canvas, own
//! pipeline. Loaded lazily when the nav mounts (never on the critical
//! path — a DOM/CSS burger renders until this arrives).
//!
//! # Uniform layout (f32s — mirrors `GPU_WIDGET_UNIFORM_F32S[BURGER]`
//! = 8 and the WGSL `Uniforms` struct; bump all three together)
//!
//!   [0]    time    accumulated seconds — drives the wave/breathe sines
//!   [1,2]  res     canvas CSS px (the shader's pixel domain)
//!   [3]    dark    0.0 light → 1.0 dark theme lerp
//!   [4..7] rect    widget clip rect — (0,0,w,h) on a dedicated canvas
//!
//! # Frame cost
//!
//! One ~32 B uniform write + one 3-vertex draw; all visuals are pure
//! fragment math — the hot path is a single API call per rAF.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

/// WGSL — the hairline-bars shader (fluid wave + breathing pulse).
const WGSL: &str = include_str!("burger.wgsl");

/// Uniform f32 count — see module doc; mirrored by the JS token.
const UNIFORM_F32S: usize = 8;

/// Public wasm surface — one burger canvas per instance.
#[wasm_bindgen]
pub struct Burger {
    /// GPU scaffold — `None` until `init` resolves.
    gpu: RefCell<Option<gpu_mini::Widget>>,
    /// Canvas CSS px — the `res` field (the shader's design-unit
    /// domain; see UNITS in the module doc).
    css: RefCell<[f32; 2]>,
    /// Canvas DEVICE px — the `rect.zw` divisor that maps `frag`
    /// (device px) into widget uv 0..1. Written by `resize`.
    dev: Cell<[f32; 2]>,
    /// Theme lerp 0..1 (light → dark).
    dark: Cell<f32>,
    /// Accumulated seconds — absorbs rAF jitter so the wave never
    /// visibly jumps phase.
    clock: Cell<f32>,
}

#[wasm_bindgen]
impl Burger {
    /// `Burger::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Burger {
        Burger {
            gpu: RefCell::new(None),
            css: RefCell::new([1.0, 1.0]),
            dev: Cell::new([1.0, 1.0]),
            dark: Cell::new(0.0),
            clock: Cell::new(0.0),
        }
    }

    /// `init` — acquire device + compile the pipeline. Rejection → JS
    /// keeps the DOM/CSS burger; mounting can never regress the page.
    ///
    /// @param canvas The widget's dedicated canvas.
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub async fn init(&self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
        *self.gpu.borrow_mut() = Some(gpu_mini::init_widget(&canvas, WGSL).await?);
        Ok(())
    }

    /// `resize` — canvas-bitmap change → surface re-configure + the
    /// DEVICE-pixel size that becomes `rect.zw` (the frag→uv divisor;
    /// `frag` arrives in device px, so the divisor must too, or every
    /// widget renders squashed at DPR ≠ 1).
    ///
    /// @param w Device-pixel width (element CSS px × DPR).
    /// @param h Device-pixel height.
    pub fn resize(&self, w: u32, h: u32) {
        self.dev.set([(w.max(1)) as f32, (h.max(1)) as f32]);

        let gpu = self.gpu.borrow();

        if let Some(w) = gpu.as_ref() {
            gpu_mini::resize(&w.surface);
        }
    }

    /// `set_size` — the canvas's CSS-pixel size (the `res` field —
    /// the shader's design-unit domain; `uv * res` yields CSS px).
    ///
    /// @param w Canvas width in CSS pixels.
    /// @param h Canvas height in CSS pixels.
    pub fn set_size(&self, w: f32, h: f32) {
        *self.css.borrow_mut() = [w.max(1.0), h.max(1.0)];
    }

    /// `set_dark` — the theme lerp: 0 = light, 1 = dark. Clamped —
    /// the shader lerps ink colors on this factor.
    ///
    /// @param v Theme darkness 0..1.
    pub fn set_dark(&self, v: f32) {
        self.dark.set(v.clamp(0.0, 1.0));
    }

    /// `frame` — one animation tick: pack the 8-f32 block, write, draw.
    ///
    /// @param dt Seconds since the previous call.
    pub fn frame(&self, dt: f32) {
        let gpu = self.gpu.borrow();

        let Some(w) = gpu.as_ref() else {
            return;
        };

        self.clock.set(self.clock.get() + dt);

        let css = *self.css.borrow();
        let dev = self.dev.get();

        // [0] time  [1,2] res (CSS px)  [3] dark
        // [4..7] rect = (0,0,dev) — DEVICE px, the frag→uv divisor
        let uniforms: [f32; UNIFORM_F32S] = [
            self.clock.get(),
            css[0],
            css[1],
            self.dark.get(),
            0.0,
            0.0,
            dev[0],
            dev[1],
        ];

        if !gpu_mini::widget_frame(w, &uniforms) {
            gpu_mini::resize(&w.surface);
        }
    }
}
