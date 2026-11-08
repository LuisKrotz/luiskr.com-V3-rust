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
//!   [0]     time_seconds   seconds (internally ×DRIFT_SPEED — ~55s)
//!   [1,2]   resolution_css canvas CSS px
//!   [3]     menu_reveal    0..1 menu open amount (JS eases)
//!   [4..6]  ink_primary    primary ink rgb (theme token)
//!   [7]     global_alpha   global opacity multiplier
//!   [8..10] ink_secondary  secondary ink rgb — field value picks inks
//!   [11]    reserved       16B alignment pad
//!   [12..15] clip_rect     (0,0,w,h) on the dedicated canvas
//!
//! # Frame cost
//!
//! One ~64 B uniform write + one fullscreen draw; the shader cost is
//! the real budget — see the .wgsl header's COST NOTE for the safe
//! levers (octaves 4→3 = ~25% cheaper).

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{GpuDevice, HtmlCanvasElement};

/// WGSL — double-domain-warped fbm contour field fragment chunk (the
/// shared hash/value_noise + vs_main arrive via the gpu-mini library).
const WGSL: &str = include_str!("menu_background.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 16;

/// Backdrop state repacked into the uniform block per frame.
struct State {
    /// Canvas CSS px — the `resolution_css` field (design-unit domain).
    css_size: [f32; 2],
    /// Canvas DEVICE px — the `clip_rect.zw` divisor mapping
    /// `pixel_coord` (device px) into widget uv 0..1. Written by `resize`.
    device_size: [f32; 2],
    /// 0..1 menu open amount (JS eases).
    menu_reveal: f32,
    /// Primary ink rgb (theme token, linear 0..1).
    ink_primary: [f32; 3],
    /// Global opacity multiplier.
    global_alpha: f32,
    /// Secondary ink rgb — the field value mixes the two inks.
    ink_secondary: [f32; 3],
}

/// Public wasm surface — one fullscreen canvas per instance.
#[wasm_bindgen]
pub struct MenuBackground {
    /// GPU scaffold — `None` until `init` resolves.
    gpu: RefCell<Option<gpu_mini::Widget>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Accumulated seconds — internally ×DRIFT_SPEED in-shader for the
    /// glacial ~55s drift.
    clock_seconds: Cell<f32>,
}

#[wasm_bindgen]
impl MenuBackground {
    /// `MenuBackground::new` — un-initialized widget; `init` follows.
    #[wasm_bindgen(constructor)]
    pub fn new() -> MenuBackground {
        // Pre-init state: no GPU yet; zeroed inks + full alpha are
        // placeholders — `set_state` pushes the theme before reveal
        // rises above 0, so they never paint.
        MenuBackground {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                css_size: [1.0, 1.0],
                device_size: [1.0, 1.0],
                menu_reveal: 0.0,
                ink_primary: [0.0; 3],
                global_alpha: 1.0,
                ink_secondary: [0.0; 3],
            }),
            clock_seconds: Cell::new(0.0),
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
        // CSS-px pair — same ≥1 clamp so `resolution_css` never scales
        // by zero.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.css_size = [width.max(1.0), height.max(1.0)];
        }
    }

    /// `set_state` — bundled scalar update: reveal amount, the two
    /// theme inks (linear 0..1 rgb triples), and the global alpha.
    ///
    /// @param menu_reveal 0..1 menu open amount.
    /// @param primary_red Primary ink red.
    /// @param primary_green Primary ink green.
    /// @param primary_blue Primary ink blue.
    /// @param global_alpha Global opacity multiplier.
    /// @param secondary_red Secondary ink red.
    /// @param secondary_green Secondary ink green.
    /// @param secondary_blue Secondary ink blue.
    pub fn set_state(
        &self,
        menu_reveal: f32,
        primary_red: f32,
        primary_green: f32,
        primary_blue: f32,
        global_alpha: f32,
        secondary_red: f32,
        secondary_green: f32,
        secondary_blue: f32,
    ) {
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };

        // Clamp the eased open-amount into 0..1.
        state.menu_reveal = menu_reveal.clamp(0.0, 1.0);

        // Both ink triples store verbatim — JS already resolved the
        // theme tokens to linear rgb.
        state.ink_primary = [primary_red, primary_green, primary_blue];

        // Clamp the opacity multiplier into 0..1.
        state.global_alpha = global_alpha.clamp(0.0, 1.0);

        state.ink_secondary = [secondary_red, secondary_green, secondary_blue];
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

        // Advance the drift clock — summing dt absorbs rAF jitter.
        self.clock_seconds
            .set(self.clock_seconds.get() + delta_seconds);

        // Snapshot the scalar state under the guard…
        let Ok(state) = self.state.try_borrow() else {
            return;
        };

        // [0] time_seconds [1,2] resolution_css [3] menu_reveal
        // [4..6] ink_primary [7] global_alpha [8..10] ink_secondary
        // [11] reserved [12..15] clip_rect = (0,0,w,h)
        let uniforms: [f32; UNIFORM_F32S] = [
            self.clock_seconds.get(),
            state.css_size[0],
            state.css_size[1],
            state.menu_reveal,
            state.ink_primary[0],
            state.ink_primary[1],
            state.ink_primary[2],
            state.global_alpha,
            state.ink_secondary[0],
            state.ink_secondary[1],
            state.ink_secondary[2],
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
impl Drop for MenuBackground {
    fn drop(&mut self) {
        if let Some(gpu_state) = self.gpu.get_mut().take() {
            std::mem::forget(gpu_state);
        }
    }
}
