//! divider-sheen — the footer hairline + traveling accent band.
//!
//! Standalone per-widget artifact — see `rust/README.md`. Replaces
//! the CSS `::after` translateX sheen on `.footer-source-divider`
//! (awards footer) and `.internal-footer-items-divider` (legal +
//! related footers) — the shader draws BOTH the hairline gradient
//! (edge-faded line in `border` color) and the 25%-wide accent band
//! sweeping left→right on the 7s ease-in-out loop, so the divider's
//! entire visual is one canvas.
//!
//! # Uniform layout (f32s = 16 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[DIVIDER_SHEEN]` and the WGSL struct)
//!
//!   [0,1]   res       canvas DEVICE px
//!   [2]     time      seconds — the sweep phase (mod 7s inside)
//!   [3]     pad
//!   [4..8]  border    hairline rgba (var(--border-color) pushed live)
//!   [8..12] accent    band rgba (var(--color-accent) pushed live)
//!   [12..16] rect     (0,0,w,h) device px — the frag→uv divisor
//!
//! # Bindings
//!
//!   0 uniform — the widget is uniform-only (no textures, no storage).
//!
//! # Reduced motion
//!
//! JS skips the widget entirely under reduced-motion (the CSS path
//! does the same via the media query) — the crate has no motion gate
//! because it never mounts when motion is reduced.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{gpu_buffer_usage, GpuDevice, HtmlCanvasElement};

/// WGSL — hairline + traveling band shader.
const WGSL: &str = include_str!("divider_sheen.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 16;

/// GPU state — the shared uniform-widget scaffold plus surface.
struct Gpu {
    /// The live canvas surface.
    surface: gpu_mini::Surface,
    /// The sheen pipeline (fullscreen triangle, uniform-only).
    pipeline: web_sys::GpuRenderPipeline,
    /// Uniform block — the 16-f32 layout above.
    ubuf: web_sys::GpuBuffer,
    /// Bound resources — built once at init (single bind group).
    bind: web_sys::GpuBindGroup,
}

/// Scalar state repacked into the uniform block per frame.
struct State {
    /// Canvas DEVICE px — `res` + `rect.zw` (the hairline's only
    /// size reference is device px; CSS size is unused).
    dev: [f32; 2],
    /// Hairline color rgba — var(--border-color) from the theme.
    border: [f32; 4],
    /// Accent band color rgba — var(--color-accent) from the theme.
    accent: [f32; 4],
}

/// Public wasm surface — one divider canvas per footer hairline.
#[wasm_bindgen]
pub struct DividerSheen {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Elapsed seconds — the sweep phase clock.
    clock: Cell<f32>,
}

#[wasm_bindgen]
impl DividerSheen {
    /// `DividerSheen::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> DividerSheen {
        DividerSheen {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                dev: [1.0, 1.0],
                border: [0.5, 0.5, 0.5, 0.6],
                accent: [0.4, 0.7, 1.0, 1.0],
            }),
            clock: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline + the single bind group.
    ///
    /// @param canvas The divider's overlay canvas.
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // binding 0 uniform — the WGSL's declared contract.
        let layout = gpu_mini::new_layout(&surface.device, &[gpu_mini::uniform_entry(0)])?;

        let pipeline = gpu_mini::fullscreen_pipeline(
            &surface.device,
            surface.format,
            WGSL,
            &[layout.clone()],
        )?;

        let ubuf = gpu_mini::new_buffer(
            &surface.device,
            (UNIFORM_F32S * 4) as u32,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        )?;

        let bind = gpu_mini::new_bind_group(
            &surface.device,
            &layout,
            &[gpu_mini::bind_buffer_entry(0, &ubuf)],
        );

        // try_borrow everywhere: a JS exception crossing a wasm frame
        // mid-call leaks the RefCell guard — a later borrow() panics
        // `unreachable` (the observed widget-crash cascade).
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };
        *slot = Some(Gpu {
            surface,
            pipeline,
            ubuf,
            bind,
        });

        Ok(())
    }

    /// `set_colors` — push the live theme colors (read once per
    /// theme flip from the DOM's computed style — the shader never
    /// sees CSS vars).
    ///
    /// @param br @param bg @param bb @param ba Hairline rgba 0-1.
    /// @param ar @param ag @param ab @param aa Accent-band rgba 0-1.
    pub fn set_colors(
        &self,
        br: f32,
        bg: f32,
        bb: f32,
        ba: f32,
        ar: f32,
        ag: f32,
        ab: f32,
        aa: f32,
    ) {
        let Ok(mut s) = self.state.try_borrow_mut() else {
            return;
        };

        s.border = [br, bg, bb, ba];
        s.accent = [ar, ag, ab, aa];
    }

    /// `set_size` — CSS px (contract parity; unused — device px only).
    ///
    /// @param _w @param _h CSS size — unused.
    pub fn set_size(&self, _w: f32, _h: f32) {}

    /// `resize` — device px — the `res`/`rect` uniform.
    ///
    /// @param w @param h Backing size in device pixels.
    pub fn resize(&self, w: f32, h: f32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.dev = [w, h];
        }
    }

    /// `frame` — advance the sweep clock, repack, draw.
    ///
    /// @param dt Frame delta in seconds.
    /// @returns false once the surface is stale/lost.
    pub fn frame(&self, dt: f32) -> bool {
        let Ok(gpu) = self.gpu.try_borrow() else {
            return false;
        };

        let Some(g) = gpu.as_ref() else {
            return false;
        };

        let t = self.clock.get() + dt;

        self.clock.set(t);

        let Ok(s) = self.state.try_borrow() else {
            return false;
        };
        let mut u = [0.0f32; UNIFORM_F32S];

        u[0] = s.dev[0];
        u[1] = s.dev[1];
        u[2] = t;
        u[4..8].copy_from_slice(&s.border);
        u[8..12].copy_from_slice(&s.accent);
        u[12] = 0.0;
        u[13] = 0.0;
        u[14] = s.dev[0];
        u[15] = s.dev[1];

        gpu_mini::write_f32s(&g.surface.queue, &g.ubuf, &u);

        gpu_mini::draw(&g.surface, &g.pipeline, &g.bind, None)
    }
}

/// `destroy` parity — the struct drop releases the device refs.
/// `destroy` parity — forget the GPU state instead of destructing
/// it: the FinalizationRegistry's `__wbg_*_free` walks every stored
/// JsValue's externref dealloc — on a torn-down table that traps
/// `memory access out of bounds` during GC. GPU objects die with the
/// shared device; leaking the handles is safe and un-panickable.
impl Drop for DividerSheen {
    fn drop(&mut self) {
        if let Some(g) = self.gpu.get_mut().take() {
            std::mem::forget(g);
        }
    }
}
