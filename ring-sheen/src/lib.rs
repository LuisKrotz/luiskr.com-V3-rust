//! ring-sheen — the home-mosaic / related-item border rings with a
//! drifting diagonal sheen.
//!
//! Standalone per-widget artifact — see `rust/README.md`. Replaces the
//! CSS masked-gradient ring + 9s `background-position` sweep with ONE
//! overlay canvas per grid: per-tile rects stream down a storage
//! sheet and the shader draws every ring as an instanced SDF — one
//! draw call for the whole grid instead of N compositor layers.
//!
//! # What the shader reproduces (home-mosaic/_item.scss parity)
//!
//!   ring   — hairline band at each tile's rounded-rect edge.
//!   sheen  — a narrow brightness band sliding along the 120°
//!            diagonal on the 9s alternate loop.
//!   breath — the ::before opacity cross-fade 0.15↔0.9, ease-in-out
//!            alternate over 9s (the two-layer trick reduces to one
//!            alpha term on GPU).
//!
//! # Tile-sheet layout (8 f32s per tile — storage `array<TileRecord>`)
//!
//!   [0..4)  rect_px      x, y, w, h — device px on the overlay canvas
//!   [4]     corner_radius (device px — the tiles share --radius
//!                        tokens, but per-tile keeps the sheet honest
//!                        when variants diverge)
//!   [5]     phase_offset per-tile phase offset 0..1 (0 in the CSS —
//!                        all tiles pulse in unison; the slot exists
//!                        so a staggered variant is one push away)
//!   [6,7]   reserved
//!
//! # Uniform layout (f32s = 16 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[RING_SHEEN]` and the WGSL struct)
//!
//!   [0,1]   resolution_dev canvas DEVICE px
//!   [2]     time_seconds   seconds — the 18s round-trip sweep clock
//!   [3]     hairline_half  hairline half-width (device px)
//!   [4..8]  edge_rgba      ring edge rgba (var(--border-item-hover))
//!   [8..12] accent_rgba    sheen-core rgba (var(--color-accent-contrast))
//!   [12..16] reserved
//!
//! # Bindings
//!
//!   0 uniform · 1 tile sheet (read-only storage, VERTEX+FRAGMENT)
//!
//! # Frame cost
//!
//! One 16-f32 uniform write + one `draw(6, tiles)` — every ring in
//! the grid is a single instanced draw.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{gpu_buffer_usage, GpuDevice, HtmlCanvasElement};

/// WGSL — instanced ring-sheen shader; keeps its own vertex stage, so
/// it composes with `wgsl::helpers_source` (SDF + quad + easing only).
const WGSL: &str = include_str!("ring_sheen.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 16;

/// Tile-sheet stride in f32s — 8 per tile (see module doc).
pub const TILE_F32S: usize = 8;

/// GPU state — surface + the storage-dependent bind group (rebuilt
/// when the sheet buffer is recreated on growth).
struct Gpu {
    /// The live canvas surface.
    surface: gpu_mini::Surface,
    /// The instanced ring pipeline.
    pipeline: web_sys::GpuRenderPipeline,
    /// Uniform block — the 16-f32 layout above.
    uniform_buffer: web_sys::GpuBuffer,
    /// Per-tile sheet — read-only storage; RefCell'd because
    /// `set_tiles` recreates the buffer on growth.
    tile_sheet: RefCell<web_sys::GpuBuffer>,
    /// Sheet capacity in tiles — recreated on growth.
    sheet_capacity: Cell<u32>,
    /// The bind-group layout (kept so the group can rebuild after
    /// sheet swaps).
    layout: web_sys::GpuBindGroupLayout,
    /// Bound resources — rebuilt after sheet swaps.
    bind_group: RefCell<web_sys::GpuBindGroup>,
}

/// Scalar state repacked into the uniform block per frame.
struct State {
    /// Canvas DEVICE px — `resolution_dev`.
    device_size: [f32; 2],
    /// Hairline half-width (device px) — the ring's band edge.
    hairline_half: f32,
    /// Live tile count — mirrors the sheet's populated rows.
    tile_count: u32,
    /// Ring edge color — var(--border-item-hover).
    edge_rgba: [f32; 4],
    /// Sheen core color — var(--color-accent-contrast).
    accent_rgba: [f32; 4],
}

/// Public wasm surface — one ring canvas per tile container.
#[wasm_bindgen]
pub struct RingSheen {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Elapsed seconds — the 9s sweep clock.
    clock_seconds: Cell<f32>,
}

#[wasm_bindgen]
impl RingSheen {
    /// `RingSheen::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> RingSheen {
        // Pre-init state: no GPU yet, empty tile list; the colors are
        // placeholders — `set_colors` pushes theme values on mount.
        RingSheen {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                device_size: [1.0, 1.0],
                hairline_half: 1.0,
                tile_count: 0,
                edge_rgba: [0.5, 0.5, 0.5, 0.8],
                accent_rgba: [0.4, 0.7, 1.0, 1.0],
            }),
            clock_seconds: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline + the initial bind group.
    ///
    /// @param canvas The overlay canvas (positioned over the grid).
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        // Step 1 — wrap the canvas in a configured surface.
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // Step 2 — bind-group layout: binding 0 uniform · 1 tile
        // sheet — vs reads both to place the ring quads, fs reads
        // them for the SDF (gpu-mini's entries are VERTEX+FRAGMENT).
        let layout = gpu_mini::new_layout(
            &surface.device,
            &[gpu_mini::uniform_entry(0), gpu_mini::storage_entry(1)],
        )?;

        // Step 3 — compile the instanced-quad pipeline. The shader
        // keeps its own vs_main, so it takes the helpers prelude
        // (sd_round_box + quad_corner + easing), not the fullscreen
        // triangle source.
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

        // Step 5 — the tile sheet, 1-tile minimum capacity (0-size
        // buffers are invalid); `set_tiles` regrows it on demand.
        let tile_sheet = gpu_mini::new_buffer(
            &surface.device,
            (TILE_F32S * 4) as u32,
            gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
        )?;

        // Step 6 — bind both buffers; the group is rebuilt when the
        // sheet regrows (bind groups are immutable).
        let bind_group = gpu_mini::new_bind_group(
            &surface.device,
            &layout,
            &[
                gpu_mini::bind_buffer_entry(0, &uniform_buffer),
                gpu_mini::bind_buffer_entry(1, &tile_sheet),
            ],
        );

        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };

        // Publish the bundle — sheet_capacity starts at 1 (the minimum
        // alloc above), bind_group is populated (not Option — the
        // sheet is always bound, even at 0 live tiles).
        *slot = Some(Gpu {
            surface,
            pipeline,
            uniform_buffer,
            tile_sheet: RefCell::new(tile_sheet),
            sheet_capacity: Cell::new(1),
            layout,
            bind_group: RefCell::new(bind_group),
        });

        Ok(())
    }

    /// `set_tiles` — upload the per-tile sheet (8 f32s per tile — see
    /// module doc). Recreates the storage buffer on growth, rebinds.
    ///
    /// @param values The flat f32 sheet (len = tiles·8).
    /// @returns The tile count adopted (0 on malformed input).
    pub fn set_tiles(&self, values: &[f32]) -> u32 {
        // try_borrow everywhere below: a JS exception that crossed a
        // wasm call frame mid-method leaks the RefCell guard (wasm has
        // no unwinding) — plain borrow() would then panic
        // `unreachable` on every later call. try_borrow turns that
        // poisoned-widget state into a graceful no-op.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return 0;
        };

        // Still `None` → init hasn't resolved; report 0 tiles so JS
        // stays on the CSS path.
        let Some(gpu_state) = gpu.as_ref() else {
            return 0;
        };

        // Whole tiles in the flat array — a malformed length floors
        // out; the leftover tail is ignored.
        let tile_count = (values.len() / TILE_F32S) as u32;

        let Ok(mut state) = self.state.try_borrow_mut() else {
            return 0;
        };

        // Publish the live count before the upload — `frame` reads it
        // as the instance count.
        state.tile_count = tile_count;

        drop(state);

        // Empty sheet → nothing to upload; 0 already reported above.
        if tile_count == 0 {
            return 0;
        }

        if tile_count > gpu_state.sheet_capacity.get() {
            // A device fault mid-regrow keeps the old sheet — the
            // upload below clips to the old capacity instead of a
            // wasm panic inside the tile callback.
            let Ok(tile_sheet) = gpu_mini::new_buffer(
                &gpu_state.surface.device,
                (values.len() * 4) as u32,
                gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
            ) else {
                return 0;
            };

            // Swap in the grown buffer…
            let Ok(mut sheet_slot) = gpu_state.tile_sheet.try_borrow_mut() else {
                return 0;
            };

            *sheet_slot = tile_sheet;

            drop(sheet_slot);

            // …and record the new capacity so the next set_tiles only
            // regrows when it actually exceeds it.
            gpu_state.sheet_capacity.set(tile_count);

            // Bind groups are immutable — the new buffer needs a new
            // group (cold path: only on growth, once per trigger).
            let Ok(sheet_ref) = gpu_state.tile_sheet.try_borrow() else {
                return 0;
            };
            let bind_group = gpu_mini::new_bind_group(
                &gpu_state.surface.device,
                &gpu_state.layout,
                &[
                    gpu_mini::bind_buffer_entry(0, &gpu_state.uniform_buffer),
                    gpu_mini::bind_buffer_entry(1, &sheet_ref),
                ],
            );
            drop(sheet_ref);

            // Publish the rebuilt group — `frame` binds it next tick.
            let Ok(mut bind_slot) = gpu_state.bind_group.try_borrow_mut() else {
                return 0;
            };

            *bind_slot = bind_group;
        }

        // Upload the whole flat sheet (viewed as raw bytes) into the
        // storage buffer — capacity is guaranteed by the grow path.
        let Ok(sheet_ref) = gpu_state.tile_sheet.try_borrow() else {
            return 0;
        };

        let _ = gpu_state.surface.queue.write_buffer_with_u32_and_u8_slice(
            &sheet_ref,
            0,
            gpu_mini::f32_bytes(values),
        );

        tile_count
    }

    /// `set_colors` — push the live theme colors (computed style read
    /// once per theme flip — the shader never sees CSS vars).
    ///
    /// @param edge_red @param edge_green @param edge_blue
    ///   @param edge_alpha Ring edge rgba 0-1.
    /// @param accent_red @param accent_green @param accent_blue
    ///   @param accent_alpha Sheen core rgba 0-1.
    /// @param hairline_half Hairline half-width in device px.
    #[allow(clippy::too_many_arguments)]
    pub fn set_colors(
        &self,
        edge_red: f32,
        edge_green: f32,
        edge_blue: f32,
        edge_alpha: f32,
        accent_red: f32,
        accent_green: f32,
        accent_blue: f32,
        accent_alpha: f32,
        hairline_half: f32,
    ) {
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };

        // Store the theme-resolved colors + hairline verbatim — the
        // shader modulates them on the sweep loop, never reading CSS.
        state.edge_rgba = [edge_red, edge_green, edge_blue, edge_alpha];

        state.accent_rgba = [accent_red, accent_green, accent_blue, accent_alpha];

        state.hairline_half = hairline_half;
    }

    /// `set_size` — CSS px (contract parity; unused — device px only).
    ///
    /// @param _width @param _height CSS size — unused.
    pub fn set_size(&self, _width: f32, _height: f32) {}

    /// `resize` — device px — the `resolution_dev` uniform.
    ///
    /// @param width @param height Backing size in device pixels.
    pub fn resize(&self, width: f32, height: f32) {
        // Store the device-px backing size — `resolution_dev`.
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };

        state.device_size = [width, height];
    }

    /// `frame` — advance the sweep clock, repack, draw all tiles.
    ///
    /// @param delta_seconds Frame delta in seconds.
    /// @returns false once the surface is stale/lost.
    pub fn frame(&self, delta_seconds: f32) -> bool {
        // Bail while init is in flight — `false` tells JS to stop
        // pumping (or keep the CSS path).
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

        // Zero-init the block, then fill the documented layout…
        let mut uniforms = [0.0f32; UNIFORM_F32S];

        // [0,1] resolution_dev · [2] time_seconds · [3] hairline half.
        uniforms[0] = state.device_size[0];

        uniforms[1] = state.device_size[1];

        uniforms[2] = elapsed;

        uniforms[3] = state.hairline_half;

        // [4..8] ring edge rgba, [8..12] sheen-core rgba.
        uniforms[4..8].copy_from_slice(&state.edge_rgba);

        uniforms[8..12].copy_from_slice(&state.accent_rgba);

        // Push the block to the GPU…
        gpu_mini::write_f32s(&gpu_state.surface.queue, &gpu_state.uniform_buffer, &uniforms);

        // Zero live tiles → nothing to draw, but the surface is still
        // live — report true so JS keeps pumping.
        if state.tile_count == 0 {
            return true;
        }

        // …then the instanced draw: 6 verts per ring quad × count.
        let Ok(bind_group) = gpu_state.bind_group.try_borrow() else {
            return false;
        };

        let ok = gpu_mini::draw_n(
            &gpu_state.surface,
            &gpu_state.pipeline,
            &bind_group,
            None,
            6,
            state.tile_count,
        );

        ok
    }
}

/// `destroy` parity — the struct drop releases the device refs via
/// forget, not destruction: when the JS FinalizationRegistry runs
/// `__wbg_ringsheen_free` the stored JsValues would each dealloc an
/// externref slab slot — a table already torn down (device lost,
/// module unloading) turns that walk into `memory access out of
/// bounds`. GPU objects die with the shared device anyway, so leaking
/// the handles is both safe and un-panickable.
impl Drop for RingSheen {
    fn drop(&mut self) {
        if let Some(gpu_state) = self.gpu.get_mut().take() {
            std::mem::forget(gpu_state);
        }
    }
}
