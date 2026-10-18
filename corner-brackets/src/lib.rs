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
//!   [0,1]   res     canvas DEVICE px
//!   [2]     hover   eased hover progress 0..1 (JS drives the ease —
//!                     one scalar, updated while the pointer state
//!                     changes; static otherwise)
//!   [3]     hair    hairline half-width (device px)
//!   [4,5]   arm0    stub arm lengths (w, h device px)
//!   [6,7]   arm1    full arm lengths (w, h device px — JS computes
//!                     from the rect so the shader stays scalar)
//!   [8..12] edge    bracket color rgba (var(--border-ink) parity)
//!   [12..16] accent hover rgba (var(--color-accent) parity)
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
use web_sys::{gpu_buffer_usage, HtmlCanvasElement};

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
    ubuf: web_sys::GpuBuffer,
    /// Bound uniform — static for the surface's lifetime.
    bind: web_sys::GpuBindGroup,
}

/// Scalar state repacked into the uniform block per frame.
struct State {
    /// Canvas DEVICE px — `res`.
    dev: [f32; 2],
    /// Eased hover progress 0..1 — arm length + tint lerp.
    hover: f32,
    /// Hairline half-width (device px).
    hair: f32,
    /// Stub arm lengths (w, h device px).
    arm0: [f32; 2],
    /// Full arm lengths (w, h device px).
    arm1: [f32; 2],
    /// Resting bracket color rgba.
    edge: [f32; 4],
    /// Hover accent color rgba.
    accent: [f32; 4],
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
        CornerBrackets {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                dev: [1.0, 1.0],
                hover: 0.0,
                hair: 1.0,
                arm0: [16.0, 16.0],
                arm1: [200.0, 48.0],
                edge: [0.5, 0.5, 0.5, 1.0],
                accent: [0.4, 0.7, 1.0, 1.0],
            }),
        }
    }

    /// `init` — device + pipeline + uniform bind group.
    ///
    /// @param canvas The overlay canvas (the button's own box).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub async fn init(&self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
        let surface = gpu_mini::init_surface(&canvas).await?;

        let layout =
            gpu_mini::new_layout(&surface.device, &[gpu_mini::uniform_entry(0)]);

        let pipeline = gpu_mini::fullscreen_pipeline(
            &surface.device,
            surface.format,
            WGSL,
            &[layout.clone()],
        );

        let ubuf = gpu_mini::new_buffer(
            &surface.device,
            (UNIFORM_F32S * 4) as u32,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        );

        let bind = gpu_mini::new_bind_group(
            &surface.device,
            &layout,
            &[gpu_mini::bind_buffer_entry(0, &ubuf)],
        );

        *self.gpu.borrow_mut() = Some(Gpu {
            surface,
            pipeline,
            ubuf,
            bind,
        });

        Ok(())
    }

    /// `set_hover` — drive the eased hover progress (JS runs the
    /// ease so the shader stays stateless; pump frames while the
    /// value is moving, settle it to present the last frame).
    ///
    /// @param v Eased progress 0..1 (0 = stub arms, 1 = full span).
    pub fn set_hover(&self, v: f32) {
        self.state.borrow_mut().hover = v.clamp(0.0, 1.0);
    }

    /// `set_arms` — push the stub/full arm lengths and hairline (all
    /// device px — JS computes them from the rect + tokens once).
    ///
    /// @param a0w @param a0h Stub arm lengths.
    /// @param a1w @param a1h Full arm lengths.
    /// @param hair Hairline half-width in device px.
    pub fn set_arms(&self, a0w: f32, a0h: f32, a1w: f32, a1h: f32, hair: f32) {
        let mut s = self.state.borrow_mut();

        s.arm0 = [a0w, a0h];
        s.arm1 = [a1w, a1h];
        s.hair = hair;
    }

    /// `set_colors` — push the resting/hover colors (computed style
    /// read once per theme flip — the shader never sees CSS vars).
    ///
    /// @param er @param eg @param eb @param ea Resting rgba 0-1.
    /// @param ar @param ag @param ab @param aa Hover rgba 0-1.
    pub fn set_colors(
        &self,
        er: f32,
        eg: f32,
        eb: f32,
        ea: f32,
        ar: f32,
        ag: f32,
        ab: f32,
        aa: f32,
    ) {
        let mut s = self.state.borrow_mut();

        s.edge = [er, eg, eb, ea];
        s.accent = [ar, ag, ab, aa];
    }

    /// `set_size` — CSS px (contract parity; unused — device px only).
    ///
    /// @param _w @param _h CSS size — unused.
    pub fn set_size(&self, _w: f32, _h: f32) {}

    /// `resize` — device px — the `res` uniform.
    ///
    /// @param w @param h Backing size in device pixels.
    pub fn resize(&self, w: f32, h: f32) {
        self.state.borrow_mut().dev = [w, h];
    }

    /// `frame` — repack the uniform block, draw the brackets.
    ///
    /// @param _dt Frame delta — unused (the shader is stateless:
    ///            `hover` is driven from JS so time never appears in
    ///            the uniform).
    /// @returns false once the surface is stale/lost.
    pub fn frame(&self, _dt: f32) -> bool {
        let gpu = self.gpu.borrow();

        let Some(g) = gpu.as_ref() else {
            return false;
        };

        let s = self.state.borrow();
        let mut u = [0.0f32; UNIFORM_F32S];

        u[0] = s.dev[0];
        u[1] = s.dev[1];
        u[2] = s.hover;
        u[3] = s.hair;
        u[4] = s.arm0[0];
        u[5] = s.arm0[1];
        u[6] = s.arm1[0];
        u[7] = s.arm1[1];
        u[8..12].copy_from_slice(&s.edge);
        u[12..16].copy_from_slice(&s.accent);

        gpu_mini::write_f32s(&g.surface.queue, &g.ubuf, &u);

        gpu_mini::draw(&g.surface, &g.pipeline, &g.bind, None)
    }
}

/// `destroy` parity — the struct drop releases the device refs.
impl Drop for CornerBrackets {
    fn drop(&mut self) {}
}
