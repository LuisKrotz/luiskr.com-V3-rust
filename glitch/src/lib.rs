//! glitch — the about-portrait VHS corruption widget.
//!
//! Standalone per-widget artifact — see `rust/README.md`. Ports the
//! 2D-canvas glitch in `website/components/home/about/glitch/` to one
//! textured fullscreen draw: the portrait uploads ONCE via
//! `bitmap_texture` (GPU→GPU copy, no CPU readback), and every
//! corruption effect runs in the fragment shader against a per-frame
//! xorshift seed — the JS side's `Math.random()` band/block dice-rolls
//! become deterministic GPU hashes, so a frame costs one 16-f32
//! uniform write + one 3-vertex draw.
//!
//! # What the shader does (mirrors `glitch/fx.ts` pass-for-pass)
//!
//!   base          — texture lookup + portrait circle mask
//!   hit gate      — hash(seed) < 0.05 idle / 0.65 hover (types.ts
//!                   `_IDLE_HIT`/`_HOVER_HIT`)
//!   rgbSplit      — red channel at −dx + cyan at +dx, replacing the
//!                   base on hit frames (the JS additive composite)
//!   sliceShift    — ≤6 (2 idle) band slots, each hashed (y, h, dx);
//!                   `_HOVER_BANDS`/`_IDLE_BANDS` + min/max heights
//!   blockTear     — ≤3 rect slots duplicated at ±dx (hover, 50% gate)
//!   rollBar       — brightened band sweeping at 0.4 h/s (hover)
//!   trackBar      — thin white strip at a hashed y (hover, 50% gate)
//!   invertFlash   — 2% chance near-total inversion (hover)
//!   scanlines     — 3px-period dark lines, jittered per frame
//!   grain         — per-pixel hash speckle (superset of the JS
//!                   fillRect loop — per-pixel grain is FREE on GPU)
//!
//! # Uniform layout (f32s = 16 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[GLITCH]` and the WGSL struct)
//!
//!   [0,1]   res     canvas DEVICE px (corruption math is px-native)
//!   [2]     time    seconds — the roll-bar sweep clock
//!   [3]     hover   0 idle / 1 hover — intensity select
//!   [4]     seed    xorshift32 state (f32-bitcast u32), advanced
//!                   Rust-side every `frame()` so each frame differs
//!   [5..9]  cover   (dx, dy, dw, dh) — the cover-fit dest rect in
//!                   device px, pushed by JS `coverRect()` parity
//!   [9..16] pad     64B alignment
//!
//! # Bindings
//!
//!   0 uniform · 1 portrait texture · 2 linear sampler
//!
//! # Why the RNG lives Rust-side
//!
//! `frame()` advances one xorshift32 — three shifts of state — so the
//! JS loop never passes randoms and every hash input is a single f32
//! slot. The shader reproduces "random per frame" by hashing
//! (seed, slot) pairs: same seed → same bands → coherent corruption.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{gpu_buffer_usage, GpuDevice, HtmlCanvasElement};

/// WGSL — the corruption shader (all passes documented inline).
const WGSL: &str = include_str!("glitch.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 16;

/// GPU state — surface + the texture-dependent bind group (rebuilt
/// when the portrait uploads).
struct Gpu {
    /// The live canvas surface.
    surface: gpu_mini::Surface,
    /// The glitch pipeline (fullscreen triangle, textured).
    pipeline: web_sys::GpuRenderPipeline,
    /// Uniform block — the 16-f32 layout above.
    ubuf: web_sys::GpuBuffer,
    /// Shared linear sampler.
    sampler: web_sys::GpuSampler,
    /// The portrait texture (uploaded once per image change).
    tex: RefCell<Option<web_sys::GpuTexture>>,
    /// The bind-group layout (kept so `rebind` can rebuild `bind`).
    layout: web_sys::GpuBindGroupLayout,
    /// Bound resources — rebuilt by `rebind` after texture uploads.
    bind: RefCell<Option<web_sys::GpuBindGroup>>,
}

/// Scalar state repacked into the uniform block per frame.
struct State {
    /// Canvas DEVICE px — `res` (all corruption math is px-native;
    /// CSS px is not used anywhere in this shader).
    dev: [f32; 2],
    /// Intensity — 0 idle ambient / 1 hover VHS.
    hover: f32,
    /// Cover-fit dest rect in device px (dx, dy, dw, dh) — the DOM
    /// `coverRect()` result pushed down so frag→uv mapping matches
    /// `object-fit: cover` + `object-position` exactly.
    cover: [f32; 4],
}

/// Public wasm surface — one glitch canvas per portrait.
#[wasm_bindgen]
pub struct Glitch {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Elapsed seconds — the roll-bar sweep clock.
    clock: Cell<f32>,
    /// xorshift32 RNG — advanced every `frame()`; the seed f32 the
    /// shader hashes all per-frame decisions from.
    rng: Cell<u32>,
}

#[wasm_bindgen]
impl Glitch {
    /// `Glitch::new` — un-initialized widget; `init` must follow.
    ///
    /// The RNG seeds from a fixed constant — determinism is a feature:
    /// replayed corruption is stable across mounts, and entropy comes
    /// from the frame counter, not wall time.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Glitch {
        Glitch {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                dev: [1.0, 1.0],
                hover: 0.0,
                cover: [0.0, 0.0, 1.0, 1.0],
            }),
            clock: Cell::new(0.0),
            rng: Cell::new(0x9E37_79B9),
        }
    }

    /// `init` — device + pipeline + sampler; the bind group builds
    /// once the portrait lands (`upload` → `rebind`).
    ///
    /// @param canvas The overlay canvas (positioned over the <img>).
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // binding 0 uniform · 1 texture · 2 sampler — the WGSL's
        // declared contract; indices must not drift.
        let layout = gpu_mini::new_layout(
            &surface.device,
            &[
                gpu_mini::uniform_entry(0),
                gpu_mini::texture_entry(1),
                gpu_mini::sampler_entry(2),
            ],
        )?;

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

        let sampler = gpu_mini::linear_sampler(&surface.device);

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
            sampler,
            tex: RefCell::new(None),
            layout,
            bind: RefCell::new(None),
        });

        Ok(())
    }

    /// `upload` — stage the decoded portrait. `ImageBitmap` input —
    /// the browser decodes off-main-thread and
    /// `copyExternalImageToTexture` moves pixels GPU→GPU, so no CPU
    /// pixel round-trip ever touches the glitch path.
    ///
    /// NOTE on CORS: the portrait is cross-origin — the copy REQUIRES
    /// the image be fetched with `crossorigin` or the bitmap is
    /// tainted and the copy throws. The JS arm catches that and keeps
    /// the 2D fallback alive — the <img> stays the source of truth.
    ///
    /// @param bitmap The decoded portrait (`createImageBitmap(img)`).
    /// @returns true when the texture landed, false before init.
    pub fn upload(&self, bitmap: &web_sys::ImageBitmap) -> bool {
        let Ok(gpu) = self.gpu.try_borrow() else {
            return false;
        };

        let Some(g) = gpu.as_ref() else {
            return false;
        };

        // A device fault mid-upload reports false — the JS arm keeps
        // the 2D fallback alive rather than a wasm panic poisoning
        // the mount closures.
        let Ok(tex) = gpu_mini::bitmap_texture(
            &g.surface.device,
            &g.surface.queue,
            bitmap,
            bitmap.width(),
            bitmap.height(),
        ) else {
            return false;
        };

        let Ok(v) = gpu_mini::view(&tex) else {
            return false;
        };

        let Ok(mut t) = g.tex.try_borrow_mut() else {
            return false;
        };
        *t = Some(tex);
        drop(t);

        // Rebuild the bind group with the fresh view — bind groups are
        // immutable by spec, so a texture swap means a new group.
        let bind = gpu_mini::new_bind_group(
            &g.surface.device,
            &g.layout,
            &[
                gpu_mini::bind_buffer_entry(0, &g.ubuf),
                gpu_mini::bind_view_entry(1, &v),
                gpu_mini::bind_sampler_entry(2, &g.sampler),
            ],
        );

        let Ok(mut b) = g.bind.try_borrow_mut() else {
            return false;
        };
        *b = Some(bind);

        true
    }

    /// `set_cover` — push the cover-fit dest rect (device px) — the
    /// DOM `coverRect()` parity value so the shader's frag→uv mapping
    /// matches `object-fit: cover` + `object-position` exactly.
    ///
    /// @param dx @param dy Dest rect origin (device px).
    /// @param dw @param dh Dest rect size (device px).
    pub fn set_cover(&self, dx: f32, dy: f32, dw: f32, dh: f32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.cover = [dx, dy, dw, dh];
        }
    }

    /// `set_hover` — the intensity latch: 0 = idle ambient, 1 = full
    /// VHS. Mirrors `session.hover`.
    ///
    /// @param hover 0 or 1.
    pub fn set_hover(&self, hover: f32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.hover = if hover > 0.5 { 1.0 } else { 0.0 };
        }
    }

    /// `set_size` — CSS px (kept for bridge contract parity; the
    /// shader reads device px from `resize` only).
    ///
    /// @param _w @param _h CSS size — unused.
    pub fn set_size(&self, _w: f32, _h: f32) {}

    /// `resize` — device px — the `res` uniform (corruption math is
    /// px-native: band heights, tear distances, grain).
    ///
    /// @param w @param h Backing size in device pixels.
    pub fn resize(&self, w: f32, h: f32) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.dev = [w, h];
        }
    }

    /// `frame` — advance the clock + RNG, repack the uniform, draw.
    ///
    /// The RNG: one xorshift32 step per frame — `x ^= x<<13; x ^= x>>17;
    /// x ^= x<<5` (Marsaglia) — maximal period 2³²−1, three shifts of
    /// state. The seed goes down as an f32; the shader bitcasts it
    /// back to u32 for the hash mixer (bit-exact round trip — the
    /// seed's low 24 bits are the only ones that survive f32, which is
    /// why the mixer uses the HIGH bits via the second shift).
    ///
    /// @param dt Frame delta in seconds (roll-bar clock).
    /// @returns false once the surface is stale/lost.
    pub fn frame(&self, dt: f32) -> bool {
        let Ok(gpu) = self.gpu.try_borrow() else {
            return false;
        };

        let Some(g) = gpu.as_ref() else {
            return false;
        };

        let Ok(bind) = g.bind.try_borrow() else {
            return false;
        };

        let Some(b) = bind.as_ref() else {
            return false;
        };

        // Advance the frame RNG — see doc: three shifts of state.
        let mut x = self.rng.get();

        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;

        self.rng.set(x);

        let t = self.clock.get() + dt;

        self.clock.set(t);

        let Ok(s) = self.state.try_borrow() else {
            return false;
        };
        let mut u = [0.0f32; UNIFORM_F32S];

        u[0] = s.dev[0];
        u[1] = s.dev[1];
        u[2] = t;
        u[3] = s.hover;
        u[4] = f32::from_bits(x);
        u[5..9].copy_from_slice(&s.cover);

        gpu_mini::write_f32s(&g.surface.queue, &g.ubuf, &u);

        gpu_mini::draw(&g.surface, &g.pipeline, b, None)
    }
}

/// `destroy` parity — forget the GPU state instead of destructing
/// it: the FinalizationRegistry's `__wbg_*_free` walks every stored
/// JsValue's externref dealloc — on a torn-down table that traps
/// `memory access out of bounds` during GC. GPU objects die with the
/// shared device; leaking the handles is safe and un-panickable.
impl Drop for Glitch {
    fn drop(&mut self) {
        if let Some(g) = self.gpu.get_mut().take() {
            std::mem::forget(g);
        }
    }
}
