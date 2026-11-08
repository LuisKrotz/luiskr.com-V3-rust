//! corner-brackets — the button corner-bracket expansion widget.
//!
//! Standalone per-widget artifact — see `rust/README.md`. Replaces
//! the `::before`/`::after` corner-bracket animation on
//! `.preferences-footer-action`, `.awards-selected-action`, and
//! `.awards-explore-action`: four L-brackets at a button's corners
//! whose arms grow from a stub to the full border length on
//! hover/focus, with a color tint to the action accent.
//!
//! Per-button widget — buttons are sparse (one canvas each), unlike
//! the ring overlay which batches a whole grid into one draw.
//!
//! # What the shader reproduces (prefs/_footer.scss parity)
//!
//!   base   — four L-brackets at the rect corners, arm length
//!            `arm0` (the stub: --space-m in the CSS).
//!   grow   — on hover/focus the arms expand to the full edge span
//!            (`width: calc(100% - 2px)`/`height: calc(100% - 2px)`
//!            in the CSS ⇒ each arm reaches the rect's edge minus the
//!            hairline inset), over --duration-normal ease-out.
//!   tint   — color interpolates edge → accent across the same
//!            progress (the CSS animates `border-color`).
//!
//! # Uniform layout (f32s = 16 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[CORNER_BRACKETS]` and the WGSL struct)
//!
//!   [0,1]   resolution_dev  canvas DEVICE px
//!   [2]     hover_amount    eased hover progress 0..1 (JS drives the
//!                     ease — one scalar, updated while the pointer
//!                     state changes; static otherwise)
//!   [3]     hairline_half   hairline half-width (device px)
//!   [4,5]   arm_rest        stub arm lengths (w, h device px)
//!   [6,7]   arm_hover       full arm lengths (w, h device px — JS
//!                     computes from the rect so the shader stays
//!                     scalar)
//!   [8..12] edge_rgba       bracket color rgba (var(--border-ink)
//!                     parity)
//!   [12..16] accent_rgba    hover rgba (var(--color-accent) parity)
//!
//! # Bindings
//!
//!   0 uniform — nothing else; the whole widget is scalar state.
//!
//! # Frame cost
//!
//! One 16-f32 uniform write + one `draw(6)` — the cheapest widget in
//! the family. JS only pumps frames while `hover` is animating; a
//! settled button presents its last frame and costs nothing.

use std::cell::RefCell;
use wasm_bindgen::prelude::*;
use web_sys::{gpu_buffer_usage, GpuDevice, HtmlCanvasElement};

/// WGSL — corner-bracket shader.
const WGSL: &str = include_str!("corner_brackets.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 16;

/// GPU state — surface + pipeline + the uniform bind group.
struct Gpu {
    /// The live canvas surface.
    surface: gpu_mini::Surface,
    /// The fullscreen-triangle pipeline.
    pipeline: web_sys::GpuRenderPipeline,
    /// Uniform block — the 16-f32 layout above.
    uniform_buffer: web_sys::GpuBuffer,
    /// Bound uniform — static for the surface's lifetime.
    bind_group: web_sys::GpuBindGroup,
}

/// Scalar state repacked into the uniform block per frame.
struct State {
    /// Canvas DEVICE px — `resolution_dev`.
    resolution_dev: [f32; 2],
    /// Eased hover progress 0..1 — arm length + tint lerp.
    hover_amount: f32,
    /// Hairline half-width (device px).
    hairline_half: f32,
    /// Stub arm lengths (w, h device px).
    arm_rest: [f32; 2],
    /// Full arm lengths (w, h device px).
    arm_hover: [f32; 2],
    /// Resting bracket color rgba.
    edge_rgba: [f32; 4],
    /// Hover accent color rgba.
    accent_rgba: [f32; 4],
}

/// Public wasm surface — one bracket canvas per button.
#[wasm_bindgen]
pub struct CornerBrackets {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
}

#[wasm_bindgen]
impl CornerBrackets {
    /// `CornerBrackets::new` — un-initialized widget; `init` must
    /// follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> CornerBrackets {
        // Pre-init state: no GPU yet; the arm/color values below are
        // placeholders — `set_arms`/`set_colors` push the real
        // rect-derived metrics before the first frame is pumped.
        CornerBrackets {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                resolution_dev: [1.0, 1.0],
                hover_amount: 0.0,
                hairline_half: 1.0,
                arm_rest: [16.0, 16.0],
                arm_hover: [200.0, 48.0],
                edge_rgba: [0.5, 0.5, 0.5, 1.0],
                accent_rgba: [0.4, 0.7, 1.0, 1.0],
            }),
        }
    }

    /// `init` — device + pipeline + uniform bind group.
    ///
    /// @param canvas The overlay canvas (the button's own box).
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        // Step 1 — wrap the canvas in a configured surface.
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // Step 2 — bind-group layout: one uniform block at binding 0.
        let layout = gpu_mini::new_layout(&surface.device, &[gpu_mini::uniform_entry(0)])?;

        // Step 3 — compile the pipeline against the surface's
        // negotiated format. The shader keeps its own vs_main (it
        // emits device-px fragment coords), so it takes the helpers
        // prelude, not the fullscreen triangle source.
        let source = gpu_mini::wgsl::helpers_source(WGSL);

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

        // Step 5 — bind the buffer once; it stays valid for the
        // surface's whole lifetime.
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

    /// `set_hover` — drive the eased hover progress (JS runs the
    /// ease so the shader stays stateless; pump frames while the
    /// value is moving, settle it to present the last frame).
    ///
    /// @param progress Eased progress 0..1 (0 = stub arms, 1 = full span).
    pub fn set_hover(&self, progress: f32) {
        // try-borrow — drop the update if the slot is contended; the
        // value is clamped into the shader's 0..1 lerp domain.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.hover_amount = progress.clamp(0.0, 1.0);
        }
    }

    /// `set_arms` — push the stub/full arm lengths and hairline (all
    /// device px — JS computes them from the rect + tokens once).
    ///
    /// @param stub_width @param stub_height Stub arm lengths.
    /// @param full_width @param full_height Full arm lengths.
    /// @param hairline Hairline half-width in device px.
    pub fn set_arms(
        &self,
        stub_width: f32,
        stub_height: f32,
        full_width: f32,
        full_height: f32,
        hairline: f32,
    ) {
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };

        // Store the rect-derived geometry verbatim — JS already
        // computed device px, no clamping needed here.
        state.arm_rest = [stub_width, stub_height];

        state.arm_hover = [full_width, full_height];

        state.hairline_half = hairline;
    }

    /// `set_colors` — push the resting/hover colors (computed style
    /// read once per theme flip — the shader never sees CSS vars).
    ///
    /// @param edge_r @param edge_g @param edge_b @param edge_a Resting rgba 0-1.
    /// @param accent_r @param accent_g @param accent_b @param accent_a Hover rgba 0-1.
    pub fn set_colors(
        &self,
        edge_r: f32,
        edge_g: f32,
        edge_b: f32,
        edge_a: f32,
        accent_r: f32,
        accent_g: f32,
        accent_b: f32,
        accent_a: f32,
    ) {
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };

        // Store both rgba quads — the shader lerps edge → accent on
        // `hover`, so it never reads CSS vars itself.
        state.edge_rgba = [edge_r, edge_g, edge_b, edge_a];

        state.accent_rgba = [accent_r, accent_g, accent_b, accent_a];
    }

    /// `set_size` — CSS px (contract parity; unused — device px only).
    ///
    /// @param _width @param _height CSS size — unused.
    pub fn set_size(&self, _width: f32, _height: f32) {}

    /// `resize` — device px — the `resolution` uniform.
    ///
    /// @param width @param height Backing size in device pixels.
    pub fn resize(&self, width: f32, height: f32) {
        // Store the device-px backing size — the shader's `resolution`
        // uniform (unlike the other widgets, no ≥1 clamp: the WGSL
        // guards its own divisions).
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.resolution_dev = [width, height];
        }
    }

    /// `frame` — repack the uniform block, draw the brackets.
    ///
    /// @param _delta_seconds Frame delta — unused (the shader is
    ///            stateless: `hover` is driven from JS so time never
    ///            appears in the uniform).
    /// @returns false once the surface is stale/lost.
    pub fn frame(&self, _delta_seconds: f32) -> bool {
        // Bail while init is in flight — JS treats `false` as "stop
        // pumping", which is correct for a not-yet-live widget too.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return false;
        };

        // Still `None` → init hasn't resolved.
        let Some(gpu_state) = gpu.as_ref() else {
            return false;
        };

        // Snapshot the scalar state under the guard — held only for
        // the repack below, never across the queue write.
        let Ok(state) = self.state.try_borrow() else {
            return false;
        };

        // Zero-init the block, then fill the documented layout…
        let mut uniforms = [0.0f32; UNIFORM_F32S];

        // [0,1] resolution — device px.
        uniforms[0] = state.resolution_dev[0];

        uniforms[1] = state.resolution_dev[1];

        // [2] eased hover progress — arm length + tint lerp.
        uniforms[2] = state.hover_amount;

        // [3] hairline half-width.
        uniforms[3] = state.hairline_half;

        // [4,5] stub arms, [6,7] full arms.
        uniforms[4] = state.arm_rest[0];

        uniforms[5] = state.arm_rest[1];

        uniforms[6] = state.arm_hover[0];

        uniforms[7] = state.arm_hover[1];

        // [8..12] resting rgba, [12..16] hover accent rgba.
        uniforms[8..12].copy_from_slice(&state.edge_rgba);

        uniforms[12..16].copy_from_slice(&state.accent_rgba);

        // Push the block to the GPU…
        gpu_mini::write_f32s(&gpu_state.surface.queue, &gpu_state.uniform_buffer, &uniforms);

        // …and draw; the bool reports whether the surface is still live.
        gpu_mini::draw(&gpu_state.surface, &gpu_state.pipeline, &gpu_state.bind_group, None)
    }
}

/// `destroy` parity — the struct drop releases the device refs.
/// `destroy` parity — forget the GPU state instead of destructing
/// it: the FinalizationRegistry's `__wbg_*_free` walks every stored
/// JsValue's externref dealloc — on a torn-down table that traps
/// `memory access out of bounds` during GC. GPU objects die with the
/// shared device; leaking the handles is safe and un-panickable.
impl Drop for CornerBrackets {
    fn drop(&mut self) {
        if let Some(gpu_state) = self.gpu.get_mut().take() {
            std::mem::forget(gpu_state);
        }
    }
}
