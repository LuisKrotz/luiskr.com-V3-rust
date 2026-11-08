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
//!   [0,1]   resolution_dev canvas DEVICE px (corruption math is
//!                          px-native)
//!   [2]     time_seconds   seconds — the roll-bar sweep clock
//!   [3]     hover_amount   0 idle / 1 hover — intensity select
//!   [4]     seed           xorshift32 state (f32-bitcast u32), advanced
//!                          Rust-side every `frame()` so each frame differs
//!   [5..9]  cover_rect     (dx, dy, dw, dh) — the cover-fit dest rect
//!                          in device px, pushed by JS `coverRect()` parity
//!   [9..16] reserved       64B alignment
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

/// WGSL — the corruption shader fragment chunk (the shared fullscreen
/// vs_main arrives via the gpu-mini library).
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
    uniform_buffer: web_sys::GpuBuffer,
    /// Shared linear sampler.
    sampler: web_sys::GpuSampler,
    /// The portrait texture (uploaded once per image change).
    portrait: RefCell<Option<web_sys::GpuTexture>>,
    /// The bind-group layout (kept so `rebind` can rebuild `bind_group`).
    layout: web_sys::GpuBindGroupLayout,
    /// Bound resources — rebuilt by `rebind` after texture uploads.
    bind_group: RefCell<Option<web_sys::GpuBindGroup>>,
}

/// Scalar state repacked into the uniform block per frame.
struct State {
    /// Canvas DEVICE px — `resolution_dev` (all corruption math is
    /// px-native; CSS px is not used anywhere in this shader).
    device_size: [f32; 2],
    /// Intensity — 0 idle ambient / 1 hover VHS.
    hover_amount: f32,
    /// Cover-fit dest rect in device px (dx, dy, dw, dh) — the DOM
    /// `coverRect()` result pushed down so frag→uv mapping matches
    /// `object-fit: cover` + `object-position` exactly.
    cover_rect: [f32; 4],
}

/// Public wasm surface — one glitch canvas per portrait.
#[wasm_bindgen]
pub struct Glitch {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Elapsed seconds — the roll-bar sweep clock.
    clock_seconds: Cell<f32>,
    /// xorshift32 RNG — advanced every `frame()`; the seed f32 the
    /// shader hashes all per-frame decisions from.
    rng_state: Cell<u32>,
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
        // Pre-init state: no GPU yet; idle intensity, a unit cover
        // rect, and the fixed xorshift seed (0x9E3779B9 — the golden-
        // ratio Weyl constant; deterministic by design, see doc).
        Glitch {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                device_size: [1.0, 1.0],
                hover_amount: 0.0,
                cover_rect: [0.0, 0.0, 1.0, 1.0],
            }),
            clock_seconds: Cell::new(0.0),
            rng_state: Cell::new(0x9E37_79B9),
        }
    }

    /// `init` — device + pipeline + sampler; the bind group builds
    /// once the portrait lands (`upload` → rebind).
    ///
    /// @param canvas The overlay canvas (positioned over the <img>).
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        // Step 1 — wrap the canvas in a configured surface.
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // Step 2 — bind-group layout: binding 0 uniform · 1 texture ·
        // 2 sampler — the WGSL's declared contract; indices must not
        // drift.
        let layout = gpu_mini::new_layout(
            &surface.device,
            &[
                gpu_mini::uniform_entry(0),
                gpu_mini::texture_entry(1),
                gpu_mini::sampler_entry(2),
            ],
        )?;

        // Step 3 — compile the textured fullscreen-triangle pipeline.
        // The shader is a fragment chunk, so it takes the shared
        // fullscreen source (helpers + triangle vs_main).
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

        // Step 5 — the linear sampler the portrait texture reads.
        let sampler = gpu_mini::linear_sampler(&surface.device);

        // try_borrow everywhere: a JS exception crossing a wasm frame
        // mid-call leaks the RefCell guard — a later borrow() panics
        // `unreachable` (the observed widget-crash cascade).
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };
        // Publish the scaffold — portrait/bind_group stay `None` until
        // `upload` lands the portrait; `frame` skips draws until then.
        *slot = Some(Gpu {
            surface,
            pipeline,
            uniform_buffer,
            sampler,
            portrait: RefCell::new(None),
            layout,
            bind_group: RefCell::new(None),
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
        // Bail while init is in flight — `false` keeps the JS arm on
        // the 2D fallback.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return false;
        };

        // Still `None` → init hasn't resolved; nothing to upload into.
        let Some(gpu_state) = gpu.as_ref() else {
            return false;
        };

        // A device fault mid-upload reports false — the JS arm keeps
        // the 2D fallback alive rather than a wasm panic poisoning
        // the mount closures.
        let Ok(portrait) = gpu_mini::bitmap_texture(
            &gpu_state.surface.device,
            &gpu_state.surface.queue,
            bitmap,
            bitmap.width(),
            bitmap.height(),
        ) else {
            return false;
        };

        // A view for the bind group — same keep-2D-fallback policy.
        let Ok(view) = gpu_mini::view(&portrait) else {
            return false;
        };

        // Swap in the portrait texture — borrow, write, release before
        // the bind-group rebuild below.
        let Ok(mut portrait_slot) = gpu_state.portrait.try_borrow_mut() else {
            return false;
        };

        *portrait_slot = Some(portrait);

        drop(portrait_slot);

        // Rebuild the bind group with the fresh view — bind groups are
        // immutable by spec, so a texture swap means a new group.
        let bind_group = gpu_mini::new_bind_group(
            &gpu_state.surface.device,
            &gpu_state.layout,
            &[
                gpu_mini::bind_buffer_entry(0, &gpu_state.uniform_buffer),
                gpu_mini::bind_view_entry(1, &view),
                gpu_mini::bind_sampler_entry(2, &gpu_state.sampler),
            ],
        );

        // Publish the rebuilt group — `frame` starts corrupting the
        // portrait on its very next tick.
        let Ok(mut bind_slot) = gpu_state.bind_group.try_borrow_mut() else {
            return false;
        };

        *bind_slot = Some(bind_group);

        true
    }

    /// `set_cover` — push the cover-fit dest rect (device px) — the
    /// DOM `coverRect()` parity value so the shader's frag→uv mapping
    /// matches `object-fit: cover` + `object-position` exactly.
    ///
    /// @param dest_x @param dest_y Dest rect origin (device px).
    /// @param dest_w @param dest_h Dest rect size (device px).
    pub fn set_cover(&self, dest_x: f32, dest_y: f32, dest_w: f32, dest_h: f32) {
        // Store the dest rect verbatim — JS already computed device
        // px via `coverRect()`; a contended slot drops the update.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.cover_rect = [dest_x, dest_y, dest_w, dest_h];
        }
    }

    /// `set_hover` — the intensity latch: 0 = idle ambient, 1 = full
    /// VHS. Mirrors `session.hover`.
    ///
    /// @param hover_amount 0 or 1.
    pub fn set_hover(&self, hover_amount: f32) {
        // Binarize the intensity latch — the shader branches on
        // `hover_amount < 0.5`, not a lerp.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.hover_amount = if hover_amount > 0.5 { 1.0 } else { 0.0 };
        }
    }

    /// `set_size` — CSS px (kept for bridge contract parity; the
    /// shader reads device px from `resize` only).
    ///
    /// @param _width @param _height CSS size — unused.
    pub fn set_size(&self, _width: f32, _height: f32) {}

    /// `resize` — device px — the `resolution_dev` uniform (corruption
    /// math is px-native: band heights, tear distances, grain).
    ///
    /// @param width @param height Backing size in device pixels.
    pub fn resize(&self, width: f32, height: f32) {
        // Store the device-px backing size — `resolution_dev` (the
        // corruption math's pixel domain; no clamp needed, WGSL
        // guards itself).
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.device_size = [width, height];
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
    /// @param delta_seconds Frame delta in seconds (roll-bar clock).
    /// @returns false once the surface is stale/lost.
    pub fn frame(&self, delta_seconds: f32) -> bool {
        // Bail while init is in flight — `false` tells JS to stop
        // pumping (or keep the 2D fallback).
        let Ok(gpu) = self.gpu.try_borrow() else {
            return false;
        };

        // Still `None` → init hasn't resolved.
        let Some(gpu_state) = gpu.as_ref() else {
            return false;
        };

        // Borrow the bind group — contended → skip the tick.
        let Ok(bind_group) = gpu_state.bind_group.try_borrow() else {
            return false;
        };

        // `None` → the portrait hasn't uploaded; an unbound draw
        // would be a validation error, so skip silently.
        let Some(bound) = bind_group.as_ref() else {
            return false;
        };

        // Advance the frame RNG — see doc: three shifts of state.
        let mut seed = self.rng_state.get();

        seed ^= seed << 13;

        seed ^= seed >> 17;

        seed ^= seed << 5;

        self.rng_state.set(seed);

        // Advance the roll-bar clock — summing dt absorbs rAF jitter.
        let elapsed = self.clock_seconds.get() + delta_seconds;

        self.clock_seconds.set(elapsed);

        // Snapshot the scalar state — the guard lives only through
        // the repack below, never across the queue write.
        let Ok(state) = self.state.try_borrow() else {
            return false;
        };

        // Zero-init the block ([9..16] reserved stays 0), then fill…
        let mut uniforms = [0.0f32; UNIFORM_F32S];

        // [0,1] resolution_dev · [2] time_seconds · [3] hover_amount.
        uniforms[0] = state.device_size[0];

        uniforms[1] = state.device_size[1];

        uniforms[2] = elapsed;

        uniforms[3] = state.hover_amount;

        // [4] the fresh RNG state — bitcast to f32 for transport;
        // the shader bitcasts back to u32 for the hash mixer.
        uniforms[4] = f32::from_bits(seed);

        // [5..9] cover-fit dest rect (dx, dy, dw, dh).
        uniforms[5..9].copy_from_slice(&state.cover_rect);

        // Push the block to the GPU…
        gpu_mini::write_f32s(&gpu_state.surface.queue, &gpu_state.uniform_buffer, &uniforms);

        // …and draw; the bool reports whether the surface is still live.
        gpu_mini::draw(&gpu_state.surface, &gpu_state.pipeline, bound, None)
    }
}

/// `destroy` parity — forget the GPU state instead of destructing
/// it: the FinalizationRegistry's `__wbg_*_free` walks every stored
/// JsValue's externref dealloc — on a torn-down table that traps
/// `memory access out of bounds` during GC. GPU objects die with the
/// shared device; leaking the handles is safe and un-panickable.
impl Drop for Glitch {
    fn drop(&mut self) {
        if let Some(gpu_state) = self.gpu.get_mut().take() {
            std::mem::forget(gpu_state);
        }
    }
}
