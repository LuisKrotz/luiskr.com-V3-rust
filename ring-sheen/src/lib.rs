//! ring-sheen — the tile border sheen ring (mosaic + related items).
//!
//! Standalone per-widget artifact — see `rust/README.md`. Replaces
//! the masked-gradient `::before`/`::after` ring on
//! `.home-mosaic-item` and `.related-item` (the `mosaic-border-sheen`
//! / `related-border-sheen` keyframes) with ONE overlay canvas per
//! container: per-tile rects stream down a storage sheet and the
//! shader draws every ring as an instanced SDF quad — one draw call
//! for the whole grid instead of 2 pseudo-element layers × N tiles,
//! each a compositor layer of its own.
//!
//! # What the shader reproduces (home-mosaic/_item.scss parity)
//!
//!   ring   — hairline band at the tile's rounded-rect edge
//!            (`padding: var(--space-hairline)` mask-composite).
//!   sheen  — a diagonal brightness band (the 120° gradient stops:
//!            transparent 32% → border-item-hover 45% →
//!            accent-contrast 50% → border-item-hover 55% →
//!            transparent 68%), its position drifting along the
//!            diagonal on the 9s alternate loop.
//!   breath — the ::before opacity cross-fade 0.15↔0.9, ease-in-out
//!            alternate over 9s (the two-layer trick reduces to one
//!            alpha term on GPU).
//!
//! # Tile-sheet layout (8 f32s per tile — storage `array<TileRec>`)
//!
//!   [0..4)  rect    x, y, w, h — device px on the overlay canvas
//!   [4]     radius  corner radius (device px — the tiles share
//!                   --radius tokens, but per-tile keeps the sheet
//!                   honest when variants diverge)
//!   [5]     phase   per-tile phase offset 0..1 (0 in the CSS — all
//!                   tiles pulse in unison; the slot exists so a
//!                   staggered variant is one push away)
//!   [6,7]   pad
//!
//! # Uniform layout (f32s = 16 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[RING_SHEEN]` and the WGSL struct)
//!
//!   [0,1]   res     canvas DEVICE px
//!   [2]     time    seconds — the 9s sweep clock
//!   [3]     hair    hairline half-width (device px)
//!   [4..8]  edge    ring edge rgba (var(--border-item-hover))
//!   [8..12] accent  sheen-core rgba (var(--color-accent-contrast))
//!   [12..16] pad
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

/// WGSL — instanced ring-sheen shader.
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
    ubuf: web_sys::GpuBuffer,
    /// Per-tile sheet — read-only storage; RefCell'd because
    /// `set_tiles` recreates the buffer on growth.
    sheet: RefCell<web_sys::GpuBuffer>,
    /// Sheet capacity in tiles — recreated on growth.
    sheet_cap: Cell<u32>,
    /// The bind-group layout (kept so `rebind` can rebuild `bind`).
    layout: web_sys::GpuBindGroupLayout,
    /// Bound resources — rebuilt by `rebind` after sheet swaps.
    bind: RefCell<web_sys::GpuBindGroup>,
}

/// Scalar state repacked into the uniform block per frame.
struct State {
    /// Canvas DEVICE px — `res`.
    dev: [f32; 2],
    /// Hairline half-width (device px) — the ring's band edge.
    hair: f32,
    /// Live tile count — mirrors the sheet's populated rows.
    count: u32,
    /// Ring edge color — var(--border-item-hover).
    edge: [f32; 4],
    /// Sheen core color — var(--color-accent-contrast).
    accent: [f32; 4],
}

/// Public wasm surface — one ring canvas per tile container.
#[wasm_bindgen]
pub struct RingSheen {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Elapsed seconds — the 9s sweep clock.
    clock: Cell<f32>,
}

#[wasm_bindgen]
impl RingSheen {
    /// `RingSheen::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> RingSheen {
        RingSheen {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                dev: [1.0, 1.0],
                hair: 1.0,
                count: 0,
                edge: [0.5, 0.5, 0.5, 0.8],
                accent: [0.4, 0.7, 1.0, 1.0],
            }),
            clock: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline + the initial bind group.
    ///
    /// @param canvas The overlay canvas (positioned over the grid).
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // binding 0 uniform · 1 tile sheet — vs reads both to place
        // the ring quads, fs reads them for the SDF (gpu-mini's
        // entries are VERTEX+FRAGMENT).
        let layout = gpu_mini::new_layout(
            &surface.device,
            &[gpu_mini::uniform_entry(0), gpu_mini::storage_entry(1)],
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

        let sheet = gpu_mini::new_buffer(
            &surface.device,
            (TILE_F32S * 4) as u32,
            gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
        )?;

        let bind = gpu_mini::new_bind_group(
            &surface.device,
            &layout,
            &[
                gpu_mini::bind_buffer_entry(0, &ubuf),
                gpu_mini::bind_buffer_entry(1, &sheet),
            ],
        );

        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };
        *slot = Some(Gpu {
            surface,
            pipeline,
            ubuf,
            sheet: RefCell::new(sheet),
            sheet_cap: Cell::new(1),
            layout,
            bind: RefCell::new(bind),
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

        let Some(g) = gpu.as_ref() else {
            return 0;
        };

        let count = (values.len() / TILE_F32S) as u32;

        let Ok(mut s) = self.state.try_borrow_mut() else {
            return 0;
        };

        s.count = count;
        drop(s);

        if count == 0 {
            return 0;
        }

        if count > g.sheet_cap.get() {
            // A device fault mid-regrow keeps the old sheet — the
            // upload below clips to the old capacity instead of a
            // wasm panic inside the tile callback.
            let Ok(sheet) = gpu_mini::new_buffer(
                &g.surface.device,
                (values.len() * 4) as u32,
                gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
            ) else {
                return 0;
            };

            let Ok(mut sh) = g.sheet.try_borrow_mut() else {
                return 0;
            };
            *sh = sheet;
            drop(sh);
            g.sheet_cap.set(count);

            // Bind groups are immutable — the new buffer needs a new
            // group (cold path: only on growth, once per trigger).
            let Ok(sheet_ref) = g.sheet.try_borrow() else {
                return 0;
            };
            let bind = gpu_mini::new_bind_group(
                &g.surface.device,
                &g.layout,
                &[
                    gpu_mini::bind_buffer_entry(0, &g.ubuf),
                    gpu_mini::bind_buffer_entry(1, &sheet_ref),
                ],
            );
            drop(sheet_ref);

            let Ok(mut b) = g.bind.try_borrow_mut() else {
                return 0;
            };
            *b = bind;
        }

        let Ok(sheet_ref) = g.sheet.try_borrow() else {
            return 0;
        };
        let _ = g.surface.queue.write_buffer_with_u32_and_u8_slice(
            &sheet_ref,
            0,
            gpu_mini::f32_bytes(values),
        );

        count
    }

    /// `set_colors` — push the live theme colors (computed style read
    /// once per theme flip — the shader never sees CSS vars).
    ///
    /// @param er @param eg @param eb @param ea Ring edge rgba 0-1.
    /// @param ar @param ag @param ab @param aa Sheen core rgba 0-1.
    /// @param hair Hairline half-width in device px.
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
        hair: f32,
    ) {
        let Ok(mut s) = self.state.try_borrow_mut() else {
            return;
        };

        s.edge = [er, eg, eb, ea];
        s.accent = [ar, ag, ab, aa];
        s.hair = hair;
    }

    /// `set_size` — CSS px (contract parity; unused — device px only).
    ///
    /// @param _w @param _h CSS size — unused.
    pub fn set_size(&self, _w: f32, _h: f32) {}

    /// `resize` — device px — the `res` uniform.
    ///
    /// @param w @param h Backing size in device pixels.
    pub fn resize(&self, w: f32, h: f32) {
        let Ok(mut s) = self.state.try_borrow_mut() else {
            return;
        };

        s.dev = [w, h];
    }

    /// `frame` — advance the sweep clock, repack, draw all tiles.
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
        u[3] = s.hair;
        u[4..8].copy_from_slice(&s.edge);
        u[8..12].copy_from_slice(&s.accent);

        gpu_mini::write_f32s(&g.surface.queue, &g.ubuf, &u);

        if s.count == 0 {
            return true;
        }

        let Ok(bind) = g.bind.try_borrow() else {
            return false;
        };
        let ok = gpu_mini::draw_n(&g.surface, &g.pipeline, &bind, None, 6, s.count);

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
        if let Some(g) = self.gpu.get_mut().take() {
            std::mem::forget(g);
        }
    }
}
