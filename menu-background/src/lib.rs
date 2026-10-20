//! menu-background — the fullscreen menu's flowing contour field.
//!
//! Standalone per-widget artifact — see `rust/README.md`. The shader
//! is the most expensive of the family (5 fbm calls × 4 octaves of
//! value noise per pixel — ~80 hashes/px, ~660M ops on a 4K screen),
//! which is exactly why it lazy-loads: the artifact + GPU cost only
//! exist while the menu is open.
//!
//! # Uniform layout (f32s = 16 — the largest block; mirrors
//! `GPU_WIDGET_UNIFORM_F32S[MENU_BACKGROUND]` and the WGSL struct)
//!
//!   [0]     time    seconds (internally ×0.018 — ~55s drift)
//!   [1,2]   res     canvas CSS px
//!   [3]     reveal  0..1 menu open amount (JS eases)
//!   [4..6]  color   primary ink rgb (theme token)
//!   [7]     alpha   global opacity multiplier
//!   [8..10] color2  secondary ink rgb — field value picks between inks
//!   [11]    pad     16B alignment pad
//!   [12..15] rect   (0,0,w,h) on the dedicated canvas
//!
//! # Frame cost
//!
//! One ~64 B uniform write + one fullscreen draw; the shader cost is
//! the real budget — see the .wgsl header's COST NOTE for the safe
//! levers (octaves 4→3 = ~25% cheaper).

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{GpuDevice, HtmlCanvasElement};

/// WGSL — double-domain-warped fbm contour field.
const WGSL: &str = include_str!("menu_background.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 16;

/// Backdrop state repacked into the uniform block per frame.
struct State {
    /// Canvas CSS px — the `res` field (design-unit domain).
    css: [f32; 2],
    /// Canvas DEVICE px — the `rect.zw` divisor mapping `frag`
    /// (device px) into widget uv 0..1. Written by `resize`.
    dev: [f32; 2],
    /// 0..1 menu open amount (JS eases).
    reveal: f32,
    /// Primary ink rgb (theme token, linear 0..1).
    color: [f32; 3],
    /// Global opacity multiplier.
    alpha: f32,
    /// Secondary ink rgb — the field value mixes the two inks.
    color2: [f32; 3],
}

/// Public wasm surface — one fullscreen canvas per instance.
#[wasm_bindgen]
pub struct MenuBackground {
    /// GPU scaffold — `None` until `init` resolves.
    gpu: RefCell<Option<gpu_mini::Widget>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Accumulated seconds — internally ×0.018 in-shader for the
    /// glacial ~55s drift.
    clock: Cell<f32>,
}

#[wasm_bindgen]
impl MenuBackground {
    /// `MenuBackground::new` — un-initialized widget; `init` follows.
    #[wasm_bindgen(constructor)]
    pub fn new() -> MenuBackground {
        MenuBackground {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                css: [1.0, 1.0],
                dev: [1.0, 1.0],
                reveal: 0.0,
                color: [0.0; 3],
                alpha: 1.0,
                color2: [0.0; 3],
            }),
            clock: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline. Rejection → JS keeps the WebGL
    /// backdrop; mounting can never regress the page.
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

        *slot = Some(built);
        Ok(())
    }

    /// `resize` — canvas-bitmap change → surface re-configure.
    ///
    /// @param _w Device-pixel width (element-side).
    /// @param _h Device-pixel height.
    pub fn resize(&self, w: u32, h: u32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.dev = [(w.max(1)) as f32, (h.max(1)) as f32];
        }

        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        if let Some(w) = gpu.as_ref() {
            gpu_mini::resize(&w.surface);
        }
    }

    /// `set_size` — the canvas's CSS-pixel size (res + rect).
    ///
    /// @param w Canvas width in CSS pixels.
    /// @param h Canvas height in CSS pixels.
    pub fn set_size(&self, w: f32, h: f32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.css = [w.max(1.0), h.max(1.0)];
        }
    }

    /// `set_state` — bundled scalar update: reveal amount, the two
    /// theme inks (linear 0..1 rgb triples), and the global alpha.
    ///
    /// @param reveal 0..1 menu open amount.
    /// @param r1 Primary ink red.
    /// @param g1 Primary ink green.
    /// @param b1 Primary ink blue.
    /// @param alpha Global opacity multiplier.
    /// @param r2 Secondary ink red.
    /// @param g2 Secondary ink green.
    /// @param b2 Secondary ink blue.
    pub fn set_state(
        &self,
        reveal: f32,
        r1: f32,
        g1: f32,
        b1: f32,
        alpha: f32,
        r2: f32,
        g2: f32,
        b2: f32,
    ) {
        let Ok(mut s) = self.state.try_borrow_mut() else {
            return;
        };

        s.reveal = reveal.clamp(0.0, 1.0);
        s.color = [r1, g1, b1];
        s.alpha = alpha.clamp(0.0, 1.0);
        s.color2 = [r2, g2, b2];
    }

    /// `frame` — one animation tick: pack, write, draw.
    ///
    /// @param dt Seconds since the previous call.
    pub fn frame(&self, dt: f32) {
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        let Some(w) = gpu.as_ref() else {
            return;
        };

        self.clock.set(self.clock.get() + dt);

        let Ok(s) = self.state.try_borrow() else {
            return;
        };

        // [0] time [1,2] res [3] reveal [4..6] color [7] alpha
        // [8..10] color2 [11] pad [12..15] rect = (0,0,w,h)
        let uniforms: [f32; UNIFORM_F32S] = [
            self.clock.get(),
            s.css[0],
            s.css[1],
            s.reveal,
            s.color[0],
            s.color[1],
            s.color[2],
            s.alpha,
            s.color2[0],
            s.color2[1],
            s.color2[2],
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

/// `destroy` parity — forget the GPU state instead of destructing
/// it: the FinalizationRegistry's `__wbg_*_free` walks every stored
/// JsValue's externref dealloc — on a torn-down table that traps
/// `memory access out of bounds` during GC. GPU objects die with the
/// shared device; leaking the handles is safe and un-panickable.
impl Drop for MenuBackground {
    fn drop(&mut self) {
        if let Some(g) = self.gpu.get_mut().take() {
            std::mem::forget(g);
        }
    }
}
