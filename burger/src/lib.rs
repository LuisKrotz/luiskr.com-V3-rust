//! burger — the nav burger's animated hairline bars.
//!
//! Standalone per-widget artifact: own GPUDevice, own canvas, own
//! pipeline. Loaded lazily when the nav mounts (never on the critical
//! path — a DOM/CSS burger renders until this arrives).
//!
//! # Uniform layout (f32s — mirrors `GPU_WIDGET_UNIFORM_F32S[BURGER]`
//! = 8 and the WGSL `Uniforms` struct; bump all three together)
//!
//!   [0]    time_seconds    accumulated seconds — drives the wave/breathe sines
//!   [1,2]  resolution_css  canvas CSS px (the shader's pixel domain)
//!   [3]    theme_dark      0.0 light → 1.0 dark theme lerp
//!   [4..7] clip_rect       widget clip rect — (0,0,w,h) on a dedicated canvas
//!
//! # Frame cost
//!
//! One ~32 B uniform write + one 3-vertex draw; all visuals are pure
//! fragment math — the hot path is a single API call per rAF.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{GpuDevice, HtmlCanvasElement};

/// WGSL — the hairline-bars fragment chunk (fluid wave + breathing
/// pulse); the shared helpers + vs_main arrive via the gpu-mini lib.
const WGSL: &str = include_str!("burger.wgsl");

/// Uniform f32 count — see module doc; mirrored by the JS token.
const UNIFORM_F32S: usize = 8;

/// Public wasm surface — one burger canvas per instance.
#[wasm_bindgen]
pub struct Burger {
    /// GPU scaffold — `None` until `init` resolves.
    gpu: RefCell<Option<gpu_mini::Widget>>,
    /// Canvas CSS px — the `resolution_css` field (the shader's
    /// design-unit domain; see UNITS in the module doc).
    css_size: RefCell<[f32; 2]>,
    /// Canvas DEVICE px — the `clip_rect.zw` divisor that maps
    /// `pixel_coord` (device px) into widget uv 0..1. Written by `resize`.
    device_size: Cell<[f32; 2]>,
    /// Theme lerp 0..1 (light → dark).
    theme_dark: Cell<f32>,
    /// Accumulated seconds — absorbs rAF jitter so the wave never
    /// visibly jumps phase.
    clock_seconds: Cell<f32>,
}

#[wasm_bindgen]
impl Burger {
    /// `Burger::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Burger {
        Burger {
            gpu: RefCell::new(None),
            css_size: RefCell::new([1.0, 1.0]),
            device_size: Cell::new([1.0, 1.0]),
            theme_dark: Cell::new(0.0),
            clock_seconds: Cell::new(0.0),
        }
    }

    /// `init` — acquire device + compile the pipeline. Rejection → JS
    /// keeps the DOM/CSS burger; mounting can never regress the page.
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

    /// `resize` — canvas-bitmap change → surface re-configure + the
    /// DEVICE-pixel size that becomes `clip_rect.zw` (the frag→uv
    /// divisor; `pixel_coord` arrives in device px, so the divisor must
    /// too, or every widget renders squashed at DPR ≠ 1).
    ///
    /// @param width Device-pixel width (element CSS px × DPR).
    /// @param height Device-pixel height.
    pub fn resize(&self, width: u32, height: u32) {
        self.device_size
            .set([(width.max(1)) as f32, (height.max(1)) as f32]);

        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        if let Some(widget) = gpu.as_ref() {
            gpu_mini::resize(&widget.surface);
        }
    }

    /// `set_size` — the canvas's CSS-pixel size (the `resolution_css`
    /// field — the shader's design-unit domain; `widget_uv *
    /// resolution_css` yields CSS px positions).
    ///
    /// @param width Canvas width in CSS pixels.
    /// @param height Canvas height in CSS pixels.
    pub fn set_size(&self, width: f32, height: f32) {
        if let Ok(mut css_size) = self.css_size.try_borrow_mut() {
            *css_size = [width.max(1.0), height.max(1.0)];
        }
    }

    /// `set_dark` — the theme lerp: 0 = light, 1 = dark. Clamped —
    /// the shader lerps ink colors on this factor.
    ///
    /// @param theme_dark Theme darkness 0..1.
    pub fn set_dark(&self, theme_dark: f32) {
        self.theme_dark.set(theme_dark.clamp(0.0, 1.0));
    }

    /// `frame` — one animation tick: pack the 8-f32 block, write, draw.
    ///
    /// @param delta_seconds Seconds since the previous call.
    pub fn frame(&self, delta_seconds: f32) {
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        let Some(widget) = gpu.as_ref() else {
            return;
        };

        self.clock_seconds
            .set(self.clock_seconds.get() + delta_seconds);

        let Ok(css_ref) = self.css_size.try_borrow() else {
            return;
        };
        let css_size = *css_ref;
        drop(css_ref);
        let device_size = self.device_size.get();

        // [0] time_seconds  [1,2] resolution_css  [3] theme_dark
        // [4..7] clip_rect = (0,0,device_size) — DEVICE px, the frag→uv divisor
        let uniforms: [f32; UNIFORM_F32S] = [
            self.clock_seconds.get(),
            css_size[0],
            css_size[1],
            self.theme_dark.get(),
            0.0,
            0.0,
            device_size[0],
            device_size[1],
        ];

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
impl Drop for Burger {
    fn drop(&mut self) {
        if let Some(gpu) = self.gpu.get_mut().take() {
            std::mem::forget(gpu);
        }
    }
}
