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
//!   [0,1]   resolution_dev canvas DEVICE px
//!   [2]     time_seconds   seconds — the sweep phase (mod 7s inside)
//!   [3]     reserved
//!   [4..8]  border_rgba    hairline rgba (var(--border-color) pushed live)
//!   [8..12] accent_rgba    band rgba (var(--color-accent) pushed live)
//!   [12..16] clip_rect     (0,0,w,h) device px — the frag→uv divisor
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

/// WGSL — hairline + traveling band fragment chunk (shared
/// ease_smooth_step + vs_main arrive via the gpu-mini library).
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
    uniform_buffer: web_sys::GpuBuffer,
    /// Bound resources — built once at init (single bind group).
    bind_group: web_sys::GpuBindGroup,
}

/// Scalar state repacked into the uniform block per frame.
struct State {
    /// Canvas DEVICE px — `resolution_dev` + `clip_rect.zw` (the
    /// hairline's only size reference is device px; CSS size unused).
    device_size: [f32; 2],
    /// Hairline color rgba — var(--border-color) from the theme.
    border_rgba: [f32; 4],
    /// Accent band color rgba — var(--color-accent) from the theme.
    accent_rgba: [f32; 4],
}

/// Public wasm surface — one divider canvas per footer hairline.
#[wasm_bindgen]
pub struct DividerSheen {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Elapsed seconds — the sweep phase clock.
    clock_seconds: Cell<f32>,
}

#[wasm_bindgen]
impl DividerSheen {
    /// `DividerSheen::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> DividerSheen {
        // Pre-init state: no GPU yet; the colors below are placeholders —
        // `set_colors` pushes the theme-resolved values before mount.
        DividerSheen {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                device_size: [1.0, 1.0],
                border_rgba: [0.5, 0.5, 0.5, 0.6],
                accent_rgba: [0.4, 0.7, 1.0, 1.0],
            }),
            clock_seconds: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline + the single bind group.
    ///
    /// @param canvas The divider's overlay canvas.
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        // Step 1 — wrap the canvas in a configured surface.
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // Step 2 — bind-group layout: binding 0 uniform — the WGSL's
        // declared contract.
        let layout = gpu_mini::new_layout(&surface.device, &[gpu_mini::uniform_entry(0)])?;

        // Step 3 — compile the fullscreen-triangle pipeline against
        // the surface's negotiated format. The shader is a fragment
        // chunk, so it takes the shared fullscreen source (helpers +
        // triangle vs_main).
        let source = gpu_mini::wgsl::fullscreen_source(WGSL);
        let pipeline = gpu_mini::fullscreen_pipeline(
            &surface.device,
            surface.format,
            &source,
            &[layout.clone()],
        )?;

        // Step 4 — the 16-f32 uniform buffer (CPU-writable each frame).
        let uniform_buffer = gpu_mini::new_buffer(
            &surface.device,
            (UNIFORM_F32S * 4) as u32,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        )?;

        // Step 5 — bind the buffer once; the widget is uniform-only
        // so the group stays valid for the surface's lifetime.
        let bind_group = gpu_mini::new_bind_group(
            &surface.device,
            &layout,
            &[gpu_mini::bind_buffer_entry(0, &uniform_buffer)],
        );

        // try_borrow everywhere: a JS exception crossing a wasm frame
        // mid-call leaks the RefCell guard — a later borrow() panics
        // `unreachable` (the observed widget-crash cascade).
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };
        // Publish the assembled GPU bundle — `frame` may now borrow it.
        *slot = Some(Gpu {
            surface,
            pipeline,
            uniform_buffer,
            bind_group,
        });

        Ok(())
    }

    /// `set_colors` — push the live theme colors (read once per
    /// theme flip from the DOM's computed style — the shader never
    /// sees CSS vars).
    ///
    /// @param border_red @param border_green @param border_blue
    ///   @param border_alpha Hairline rgba 0-1.
    /// @param accent_red @param accent_green @param accent_blue
    ///   @param accent_alpha Accent-band rgba 0-1.
    #[allow(clippy::too_many_arguments)]
    pub fn set_colors(
        &self,
        border_red: f32,
        border_green: f32,
        border_blue: f32,
        border_alpha: f32,
        accent_red: f32,
        accent_green: f32,
        accent_blue: f32,
        accent_alpha: f32,
    ) {
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };

        // Store both rgba quads verbatim — JS already resolved the
        // theme tokens, so the shader never touches CSS vars.
        state.border_rgba = [border_red, border_green, border_blue, border_alpha];

        state.accent_rgba = [accent_red, accent_green, accent_blue, accent_alpha];
    }

    /// `set_size` — CSS px (contract parity; unused — device px only).
    ///
    /// @param _width @param _height CSS size — unused.
    pub fn set_size(&self, _width: f32, _height: f32) {}

    /// `resize` — device px — the `resolution_dev`/`clip_rect` uniform.
    ///
    /// @param width @param height Backing size in device pixels.
    pub fn resize(&self, width: f32, height: f32) {
        // Store the device-px backing size — feeds both
        // `resolution_dev` and `clip_rect.zw` (the whole canvas is the
        // widget rect).
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.device_size = [width, height];
        }
    }

    /// `frame` — advance the sweep clock, repack, draw.
    ///
    /// @param delta_seconds Frame delta in seconds.
    /// @returns false once the surface is stale/lost.
    pub fn frame(&self, delta_seconds: f32) -> bool {
        // Bail while init is in flight — JS treats `false` as "stop
        // pumping", correct for a not-yet-live widget too.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return false;
        };

        // Still `None` → init hasn't resolved.
        let Some(gpu_state) = gpu.as_ref() else {
            return false;
        };

        // Advance the sweep clock — summing dt absorbs rAF jitter.
        let elapsed = self.clock_seconds.get() + delta_seconds;

        self.clock_seconds.set(elapsed);

        // Snapshot the scalar state — the guard lives only through
        // the repack below, never across the queue write.
        let Ok(state) = self.state.try_borrow() else {
            return false;
        };

        // Zero-init the block ([3] reserved stays 0), then fill…
        let mut uniforms = [0.0f32; UNIFORM_F32S];

        // [0,1] resolution_dev — device px.
        uniforms[0] = state.device_size[0];

        uniforms[1] = state.device_size[1];

        // [2] time_seconds — the 7s sweep phase.
        uniforms[2] = elapsed;

        // [4..8] hairline rgba, [8..12] band rgba.
        uniforms[4..8].copy_from_slice(&state.border_rgba);

        uniforms[8..12].copy_from_slice(&state.accent_rgba);

        // [12..16] clip_rect = (0,0,w,h) — the whole canvas is the widget.
        uniforms[12] = 0.0;

        uniforms[13] = 0.0;

        uniforms[14] = state.device_size[0];

        uniforms[15] = state.device_size[1];

        // Push the block to the GPU…
        gpu_mini::write_f32s(&gpu_state.surface.queue, &gpu_state.uniform_buffer, &uniforms);

        // …and draw; the bool reports whether the surface is still live.
        gpu_mini::draw(&gpu_state.surface, &gpu_state.pipeline, &gpu_state.bind_group, None)
    }
}

/// `destroy` parity — forget the GPU state instead of destructing
/// it: the FinalizationRegistry's `__wbg_*_free` walks every stored
/// JsValue's externref dealloc — on a torn-down table that traps
/// `memory access out of bounds` during GC. GPU objects die with the
/// shared device; leaking the handles is safe and un-panickable.
impl Drop for DividerSheen {
    fn drop(&mut self) {
        if let Some(gpu_state) = self.gpu.get_mut().take() {
            std::mem::forget(gpu_state);
        }
    }
}
