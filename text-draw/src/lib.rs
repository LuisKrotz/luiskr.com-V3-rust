//! text-draw — the per-character writing animation (draw-text).
//!
//! Standalone per-widget artifact — see `rust/README.md`. Replaces
//! the CSS per-char `<span>` animation (`char-draw` keyframes:
//! opacity 0→1 + translateY 0.12em→0, cubic-bezier(0.22,1,0.36,1),
//! delay = offset + i·char-delay + word cascade) with ONE instanced
//! draw over a rasterized atlas:
//!
//! - JS rasterizes the rendered text into an atlas canvas ONCE (the
//!   browser's font engine does the raster — fonts don't exist in
//!   wasm) → `ImageBitmap` → `upload_atlas`.
//! - JS pushes per-char records through `set_chars`: dest rect
//!   (device px), atlas uv rect, per-char delay (the exact CSS value:
//!   offset + i·char-delay + word cascade folded in), duration.
//! - The vertex shader synthesizes each char's quad from
//!   vertex_index+instance_index against the storage sheet — no
//!   vertex buffer — applies the eased rise + a settling hand tremor,
//!   and the fragment stage reveals the atlas pixels through a
//!   wobbled stroke-direction mask so every letter reads as drawn by
//!   a pen, not faded in (see text_draw.wgsl's header for the mask
//!   math; delays/durations are the unchanged CSS values).
//!
//! The DOM text stays mounted (aria-hidden char spans + the real
//! text sibling) — indexability, screen readers, and reader mode are
//! unaffected; the canvas is the visual layer only, like the glitch
//! overlay over the portrait <img>.
//!
//! # Char-sheet layout (12 f32s per char — storage `array<CharRecord>`)
//!
//!   [0..4)  dest_rect    x, y, w, h — device px on the overlay canvas
//!   [4..8)  uv_rect      u0, v0, u1, v1 — the char's atlas sub-rect
//!   [8]     delay_ms     ms until this char starts (CSS delay folded:
//!                        offset + i·char-delay + word cascade)
//!   [9]     duration_ms  ms — the char's animation duration
//!   [10]    em_px        the char's font-size px — `translateY(0.12em)`
//!                        unit
//!   [11]    pad
//!
//! # Uniform layout (f32s = 16 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[TEXT_DRAW]` and the WGSL struct)
//!
//!   [0,1]   resolution   canvas DEVICE px
//!   [2]     time_ms      ms since the draw was triggered (ms — delays
//!                        are ms-denominated to match the CSS values)
//!   [3]     char_count   live char count (instance count mirror —
//!                        the shader early-outs past it)
//!   [4..8]  ink_rgba     text color rgba (computed style pushed once)
//!   [8..16] pad
//!
//! # Bindings
//!
//!   0 uniform · 1 char sheet (read-only storage) ·
//!   2 atlas texture · 3 linear sampler
//!
//! # Frame cost
//!
//! One 16-f32 uniform write + one `draw(6, count)` — the whole
//! stagger, any char count, is a single draw call. When `time` passes
//! the last char's delay+dur the JS side stops driving frames (the
//! last presented frame persists) and marks the block done.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{gpu_buffer_usage, GpuDevice, HtmlCanvasElement};

/// WGSL — instanced char-quad shader.
const WGSL: &str = include_str!("text_draw.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 16;

/// Char-sheet stride in f32s — 12 per char (see module doc).
pub const CHAR_F32S: usize = 12;

/// GPU state — surface + the storage/texture-dependent bind group
/// (rebuilt when the atlas or sheet lands).
struct Gpu {
    /// The live canvas surface.
    surface: gpu_mini::Surface,
    /// The instanced char pipeline.
    pipeline: web_sys::GpuRenderPipeline,
    /// Uniform block — the 16-f32 layout above.
    uniform_buffer: web_sys::GpuBuffer,
    /// Per-char sheet — read-only storage; RefCell'd because
    /// `set_chars` recreates the buffer on growth (the same interior-
    /// mutability pattern `atlas_tex`/`bind_group` use).
    sheet: RefCell<web_sys::GpuBuffer>,
    /// Sheet capacity in chars — recreated on growth (sheets are
    /// written once per trigger, so recreate-on-grow is free).
    sheet_capacity: Cell<u32>,
    /// Shared linear sampler for the atlas.
    sampler: web_sys::GpuSampler,
    /// The atlas texture (uploaded once per raster).
    atlas_tex: RefCell<Option<web_sys::GpuTexture>>,
    /// The bind-group layout (kept so `rebind` can rebuild
    /// `bind_group`).
    layout: web_sys::GpuBindGroupLayout,
    /// Bound resources — rebuilt by `rebind` after uploads.
    bind_group: RefCell<Option<web_sys::GpuBindGroup>>,
}

/// Scalar state repacked into the uniform block per frame.
struct State {
    /// Canvas DEVICE px — `resolution`.
    resolution_dev: [f32; 2],
    /// Ink color rgba — the text's computed color.
    ink_rgba: [f32; 4],
    /// Live char count — mirrors the sheet's populated rows.
    char_count: u32,
    /// ms timestamp the draw triggered at — `frame` reports elapsed
    /// ms as (clock − t0)·1000 so delay/dur stay in the CSS's units.
    trigger_ms: f32,
}

/// Public wasm surface — one text-draw canvas per animated block.
#[wasm_bindgen]
pub struct TextDraw {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Elapsed seconds — the stagger clock.
    clock: Cell<f32>,
}

#[wasm_bindgen]
impl TextDraw {
    /// `TextDraw::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> TextDraw {
        // Pre-init state: no GPU yet, empty char list; white ink is a
        // placeholder — `set_ink` pushes the computed color on mount.
        TextDraw {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                resolution_dev: [1.0, 1.0],
                ink_rgba: [1.0, 1.0, 1.0, 1.0],
                char_count: 0,
                trigger_ms: 0.0,
            }),
            clock: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline + sampler; the bind group builds
    /// once the atlas + first sheet land (`upload_atlas`/`set_chars`
    /// → `rebind`).
    ///
    /// @param canvas The overlay canvas.
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        // Step 1 — wrap the canvas in a configured surface.
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // Step 2 — bind-group layout: binding 0 uniform · 1 sheet ·
        // 2 atlas · 3 sampler — the vertex stage reads the uniform
        // (res/time) and the char sheet (gpu-mini's entries are
        // VERTEX+FRAGMENT for exactly this reason).
        let layout = gpu_mini::new_layout(
            &surface.device,
            &[
                gpu_mini::uniform_entry(0),
                gpu_mini::storage_entry(1),
                gpu_mini::texture_entry(2),
                gpu_mini::sampler_entry(3),
            ],
        )?;

        // Step 3 — compile the instanced char-quad pipeline. The
        // shader keeps its own vs_main (storage-sheet instancing), so
        // it takes the helpers prelude (quad_corner, hash_sin_1d),
        // not the fullscreen triangle source.
        let source = gpu_mini::wgsl::helpers_source(WGSL);

        let pipeline = gpu_mini::fullscreen_pipeline(
            &surface.device,
            surface.format,
            &source,
            &[layout.clone()],
        )?;

        // Step 4 — the 16-f32 uniform block (CPU-writable each frame).
        let uniform_buffer = gpu_mini::new_buffer(
            &surface.device,
            (UNIFORM_F32S * 4) as u32,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        )?;

        // Initial sheet: capacity for one char — `set_chars` recreates
        // on growth (sheets are written once per trigger, so the
        // recreate path is cold).
        let sheet = gpu_mini::new_buffer(
            &surface.device,
            (CHAR_F32S * 4) as u32,
            gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
        )?;

        // Step 6 — the atlas's linear sampler.
        let sampler = gpu_mini::linear_sampler(&surface.device);

        // try_borrow everywhere below: a JS exception that crossed a
        // wasm frame mid-call leaks the RefCell guard (wasm has no
        // unwinding) — a later borrow() then panics `unreachable`.
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };
        // Publish the scaffold — atlas_tex/bind_group stay `None`
        // until the atlas and first sheet land; `frame` skips draws
        // until then.
        *slot = Some(Gpu {
            surface,
            pipeline,
            uniform_buffer,
            sheet: RefCell::new(sheet),
            sheet_capacity: Cell::new(1),
            sampler,
            atlas_tex: RefCell::new(None),
            layout,
            bind_group: RefCell::new(None),
        });

        Ok(())
    }

    /// `upload_atlas` — stage the rasterized char atlas. `ImageBitmap`
    /// input — `copyExternalImageToTexture` moves pixels GPU→GPU.
    /// Rebinds so the new atlas goes live next frame.
    ///
    /// @param bitmap The decoded atlas bitmap.
    /// @returns true when the texture landed, false before init.
    pub fn upload_atlas(&self, bitmap: &web_sys::ImageBitmap) -> bool {
        // Bail while init is in flight — `false` keeps the DOM text.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return false;
        };

        // Still `None` → init hasn't resolved.
        let Some(gpu_state) = gpu.as_ref() else {
            return false;
        };

        // A device fault reports false — the JS arm keeps the DOM
        // text visible instead of a wasm panic inside the callback.
        let Ok(atlas) = gpu_mini::bitmap_texture(
            &gpu_state.surface.device,
            &gpu_state.surface.queue,
            bitmap,
            bitmap.width(),
            bitmap.height(),
        ) else {
            return false;
        };

        // Swap in the atlas texture — borrow, write, release before
        // the rebind below borrows the same cell.
        let Ok(mut tex_slot) = gpu_state.atlas_tex.try_borrow_mut() else {
            return false;
        };

        *tex_slot = Some(atlas);

        drop(tex_slot);

        // Rebuild the bind group against the fresh atlas (immutable
        // groups — a texture swap means a new group).
        self.rebind(gpu_state);

        true
    }

    /// `set_chars` — upload the per-char sheet (12 f32s per char —
    /// see module doc). Recreates the storage buffer when the sheet
    /// outgrows capacity, then rebinds.
    ///
    /// @param values The flat f32 sheet (len = count·12).
    /// @returns The char count adopted (0 on malformed input).
    pub fn set_chars(&self, values: &[f32]) -> u32 {
        let Ok(gpu) = self.gpu.try_borrow() else {
            return 0;
        };

        // Still `None` → init hasn't resolved; report 0 chars so JS
        // keeps the DOM text visible.
        let Some(gpu_state) = gpu.as_ref() else {
            return 0;
        };

        // Whole char records in the flat array — a malformed length
        // floors out; the leftover tail is ignored.
        let count = (values.len() / CHAR_F32S) as u32;

        let Ok(mut state) = self.state.try_borrow_mut() else {
            return 0;
        };

        // Publish the live count before the upload — `frame` reads it
        // as the instance count.
        state.char_count = count;

        drop(state);

        // Empty sheet → nothing to upload; 0 already reported above.
        if count == 0 {
            return 0;
        }

        if count > gpu_state.sheet_capacity.get() {
            // A device fault mid-regrow keeps the old sheet — 0 tells
            // the JS arm the sheet wasn't adopted rather than a wasm
            // panic inside the set_chars callback.
            let Ok(sheet) = gpu_mini::new_buffer(
                &gpu_state.surface.device,
                (values.len() * 4) as u32,
                gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
            ) else {
                return 0;
            };

            // Swap in the grown buffer…
            let Ok(mut sheet_slot) = gpu_state.sheet.try_borrow_mut() else {
                return 0;
            };

            *sheet_slot = sheet;

            drop(sheet_slot);

            // …and record the new capacity so the next set_chars only
            // regrows when it actually exceeds it.
            gpu_state.sheet_capacity.set(count);
        }

        // Upload the whole flat sheet (viewed as raw bytes) into the
        // storage buffer — capacity is guaranteed by the grow path.
        let Ok(sheet_ref) = gpu_state.sheet.try_borrow() else {
            return 0;
        };

        let _ = gpu_state.surface.queue.write_buffer_with_u32_and_u8_slice(
            &sheet_ref,
            0,
            gpu_mini::f32_bytes(values),
        );

        drop(sheet_ref);

        // Rebind only once the atlas exists — a sheet-only group
        // would bind an empty texture slot (validation error); until
        // upload_atlas lands, `frame` keeps skipping draws.
        if gpu_state.atlas_tex.try_borrow().map_or(false, |tex| tex.is_some()) {
            self.rebind(gpu_state);
        }

        count
    }

    /// `set_ink` — push the text color (computed style read once).
    ///
    /// @param red @param green @param blue @param alpha rgba 0-1.
    pub fn set_ink(&self, red: f32, green: f32, blue: f32, alpha: f32) {
        // Store the computed text color verbatim — the shader tints
        // the atlas samples with it, never reading CSS vars.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.ink_rgba = [red, green, blue, alpha];
        }
    }

    /// `trigger` — start the stagger: the clock's current value
    /// becomes t0 so `frame` reports elapsed ms from HERE.
    pub fn trigger(&self) {
        // Stamp t0 on the ms clock — `frame` reports elapsed ms from
        // HERE so per-char delays/durations stay in CSS units.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.trigger_ms = self.clock.get() * 1000.0;
        }
    }

    /// `set_size` — CSS px (contract parity; unused — device px only).
    ///
    /// @param _width @param _height CSS size — unused.
    pub fn set_size(&self, _width: f32, _height: f32) {}

    /// `resize` — device px — the `resolution` uniform.
    ///
    /// @param width @param height Backing size in device pixels.
    pub fn resize(&self, width: f32, height: f32) {
        // Store the device-px backing size — `resolution`.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.resolution_dev = [width, height];
        }

        // Canvas-bitmap change → the context must re-configure or the
        // next get_current_texture lands on the stale-size (or zero-
        // size, when init ran before the overlay was laid out) surface
        // — every frame after resize drew into an invalid texture.
        if let Ok(gpu) = self.gpu.try_borrow() {
            if let Some(gpu_state) = gpu.as_ref() {
                gpu_mini::resize(&gpu_state.surface);
            }
        }
    }

    /// `rebind` — rebuild the bind group after a sheet/atlas swap.
    /// Bind groups are immutable by spec — resource swaps mean a new
    /// group, cheap at trigger-time (never mid-frame).
    fn rebind(&self, gpu_state: &Gpu) {
        // Borrow the atlas slot…
        let Ok(tex) = gpu_state.atlas_tex.try_borrow() else {
            return;
        };

        // …`None` → no upload has landed yet; nothing to bind.
        let Some(atlas) = tex.as_ref() else {
            return;
        };

        // A device fault skips the rebind — the previous group keeps
        // drawing rather than a wasm panic inside the callback.
        let Ok(atlas_view) = gpu_mini::view(atlas) else {
            return;
        };

        // Borrow the sheet buffer for the group's binding 1…
        let Ok(sheet_ref) = gpu_state.sheet.try_borrow() else {
            return;
        };

        // Rebuild the group: same uniform_buffer/sampler, current
        // sheet + atlas view — all four bindings in one group.
        let bind_group = gpu_mini::new_bind_group(
            &gpu_state.surface.device,
            &gpu_state.layout,
            &[
                gpu_mini::bind_buffer_entry(0, &gpu_state.uniform_buffer),
                gpu_mini::bind_buffer_entry(1, &sheet_ref),
                gpu_mini::bind_view_entry(2, &atlas_view),
                gpu_mini::bind_sampler_entry(3, &gpu_state.sampler),
            ],
        );

        // Release both borrows BEFORE publishing — the bind write
        // borrows a different cell but the same one-at-a-time rule
        // keeps the crash-cascade impossible.
        drop(sheet_ref);
        drop(tex);

        // Publish — `frame` picks up the new resources next tick.
        if let Ok(mut bind_slot) = gpu_state.bind_group.try_borrow_mut() {
            *bind_slot = Some(bind_group);
        }
    }

    /// `frame` — advance the clock, repack, draw all live chars.
    ///
    /// @param delta_seconds Frame delta in seconds.
    /// @returns false once the surface is stale/lost (JS disarms).
    pub fn frame(&self, delta_seconds: f32) -> bool {
        // Bail while init is in flight — `false` tells JS to disarm.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return false;
        };

        // Still `None` → init hasn't resolved.
        let Some(gpu_state) = gpu.as_ref() else {
            return false;
        };

        // Borrow the bind group — contended → skip the tick.
        let Ok(bind_guard) = gpu_state.bind_group.try_borrow() else {
            return false;
        };

        // `None` → atlas or sheet hasn't landed; an unbound draw
        // would be a validation error, so skip silently.
        let Some(bound) = bind_guard.as_ref() else {
            return false;
        };

        // Advance the stagger clock — summing the delta absorbs rAF
        // jitter.
        let clock_seconds = self.clock.get() + delta_seconds;

        self.clock.set(clock_seconds);

        // Snapshot the scalar state — the guard lives only through
        // the repack below, never across the queue write.
        let Ok(state) = self.state.try_borrow() else {
            return false;
        };

        // Zero-init the block, then fill the documented layout…
        let mut uniforms = [0.0f32; UNIFORM_F32S];

        // [0,1] resolution — device px.
        uniforms[0] = state.resolution_dev[0];

        uniforms[1] = state.resolution_dev[1];

        // [2] time_ms — elapsed MS since `trigger` stamped t0 (the
        // CSS delays are ms-denominated, so the shader's domain is ms).
        uniforms[2] = clock_seconds * 1000.0 - state.trigger_ms;

        // [3] live char count — the shader early-outs past it.
        uniforms[3] = state.char_count as f32;

        // [4..8] ink rgba — the text's computed color.
        uniforms[4..8].copy_from_slice(&state.ink_rgba);

        // Push the block to the GPU…
        gpu_mini::write_f32s(&gpu_state.surface.queue, &gpu_state.uniform_buffer, &uniforms);

        // Zero live chars → nothing to draw, but the surface is still
        // live — report true so JS keeps pumping.
        if state.char_count == 0 {
            return true;
        }

        // …then the instanced draw: 6 verts per char quad × count.
        gpu_mini::draw_n(&gpu_state.surface, &gpu_state.pipeline, bound, None, 6, state.char_count)
    }
}

/// `destroy` parity — forget the GPU state instead of destructing
/// it: the FinalizationRegistry's `__wbg_textdraw_free` would walk
/// every stored JsValue's externref dealloc — on a torn-down table
/// that traps `memory access out of bounds` during GC (the observed
/// `TextDrawFinalization` crash). GPU objects die with the shared
/// device; leaking the handles is safe and un-panickable.
impl Drop for TextDraw {
    fn drop(&mut self) {
        if let Some(gpu_state) = self.gpu.get_mut().take() {
            std::mem::forget(gpu_state);
        }
    }
}
