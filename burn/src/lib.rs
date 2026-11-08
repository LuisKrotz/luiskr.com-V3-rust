//! burn — the mosaic title's paper-combustion particle sim.
//!
//! Standalone per-widget artifact — see `rust/README.md`. The CPU sim
//! (`sim.rs`: embers on buoyancy, char flecks on gravity, rebuild
//! deposits on the burn front) lives fully in wasm — JS only pushes
//! letter geometry, ink occupancy, tint ramp, and per-frame stage.
//!
//! # Why instanced quads via storage buffer
//!
//! The sim emits hundreds–thousands of particles per frame. A
//! per-particle draw would be thousands of calls; instancing makes it
//! ONE. The instances travel in a read-only storage buffer indexed by
//! `@builtin(instance_index)` — chosen over vertex-buffer instancing
//! because it needs no vertex-layout descriptors: one `write_buffer`
//! per frame, one struct shape to keep in sync (see burn.wgsl's
//! 4×vec4 record doc).
//!
//! # Buffers
//!
//!   0 uniform   viewport px (vec2) — the CSS→NDC projection divisor
//!   1 storage   instance records — 16 f32s each (see `sim::QUAD_F32S`)
//!
//! The instance buffer grows geometrically (double on overflow) so
//! steady frames never re-allocate — frame-budget rule.
//!
//! # Frame cost
//!
//! Sim: O(particles) scalar math in wasm (~µs for 10³).
//! GPU: one ~64B-uniform write + one instance-buffer write + one
//! `draw(6, n)` call. Nothing crosses the JS boundary per particle.

// The sim is pure math — host-testable; everything below (wasm-bindgen
// exports, web_sys device types, gpu_mini) only exists for wasm32, so
// it is cfg-gated like scene3d's renderer/wasm modules.
pub mod sim;

#[cfg(test)]
mod sim_tests;

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::{gpu_buffer_usage, GpuDevice, HtmlCanvasElement};

/// WGSL — storage-buffer instanced tinted quads.
#[cfg(target_arch = "wasm32")]
const WGSL: &str = include_str!("burn.wgsl");

/// Starting instance capacity — doubles on overflow. 512 × 64 B =
/// 32 KB covers a typical burn; the first overflow is a one-time
/// realloc, not a per-frame cost.
#[cfg(target_arch = "wasm32")]
const INITIAL_INSTANCE_CAPACITY: usize = 512;

/// GPU state — surface + the instanced-quad resources.
#[cfg(target_arch = "wasm32")]
struct Gpu {
    /// The live canvas surface.
    surface: gpu_mini::Surface,
    /// The instanced-quad pipeline.
    pipeline: web_sys::GpuRenderPipeline,
    /// Viewport uniform (16 B — vec2 + pad).
    uniform_buffer: web_sys::GpuBuffer,
    /// Instance storage — capacity `instance_capacity` records.
    instance_buffer: web_sys::GpuBuffer,
    /// Instance capacity in records — geometric growth.
    instance_capacity: usize,
    /// Bound resources — rebuilt whenever `instance_buffer` regrows.
    bind_group: web_sys::GpuBindGroup,
    /// Bind-group layout — kept for regrowth rebinds.
    layout: web_sys::GpuBindGroupLayout,
}

/// Public wasm surface — one burn canvas per instance (the mosaic
/// mounts one per title interaction).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct Burn {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// The CPU sim — letters/ink/colors/dir/stage live inside.
    sim: RefCell<Option<sim::BurnSim>>,
    /// Reusable instance scratch — cleared+refilled per frame so the
    /// emit path never allocates (zero-allocation frame rule).
    scratch: RefCell<Vec<f32>>,
    /// Canvas CSS px — the viewport uniform.
    viewport_css: RefCell<[f32; 2]>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Burn {
    /// `Burn::new` — un-initialized surface; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Burn {
        // Pre-init state: no GPU, no sim session. The scratch Vec is
        // pre-sized to the starting capacity so the steady-state emit
        // path never grows it (zero-allocation frame rule).
        Burn {
            gpu: RefCell::new(None),
            sim: RefCell::new(None),
            scratch: RefCell::new(Vec::with_capacity(INITIAL_INSTANCE_CAPACITY * sim::QUAD_F32S)),
            viewport_css: RefCell::new([1.0, 1.0]),
        }
    }

    /// `init` — device + pipeline + buffers. Rejection → JS keeps the
    /// 2D-canvas burn; mounting can never regress the page.
    ///
    /// @param canvas The widget's dedicated canvas.
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        // Step 1 — wrap the canvas in a configured surface.
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // Step 2 — binding 0 uniform viewport · 1 storage instances —
        // the two binding slots burn.wgsl declares.
        let layout = gpu_mini::new_layout(
            &surface.device,
            &[gpu_mini::uniform_entry(0), gpu_mini::storage_entry(1)],
        )?;

        // Step 3 — compile the instanced-quad pipeline. The shader
        // keeps its own vs_main (storage instancing), so it takes the
        // helpers prelude (quad_corner), not the fullscreen triangle
        // source.
        let source = gpu_mini::wgsl::helpers_source(WGSL);

        let pipeline = gpu_mini::fullscreen_pipeline(
            &surface.device,
            surface.format,
            &source,
            &[layout.clone()],
        )?;

        // Step 4 — the 16 B viewport uniform (vec2 + pad).
        let uniform_buffer = gpu_mini::new_buffer(
            &surface.device,
            16,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        )?;

        // Step 5 — the instance storage buffer at starting capacity
        // (512 records × 64 B = 32 KB; regrowth is the cold path).
        let instance_capacity = INITIAL_INSTANCE_CAPACITY;
        let instance_buffer = gpu_mini::new_buffer(
            &surface.device,
            (instance_capacity * sim::QUAD_F32S * 4) as u32,
            gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
        )?;

        // Step 6 — build the bind group once; regrowth rebinds.
        let bind_group = gpu_mini::new_bind_group(
            &surface.device,
            &layout,
            &[
                gpu_mini::bind_buffer_entry(0, &uniform_buffer),
                gpu_mini::bind_buffer_entry(1, &instance_buffer),
            ],
        );

        // try_borrow everywhere: a JS exception crossing a wasm frame
        // mid-call leaks the RefCell guard — a later borrow() panics
        // `unreachable` (the observed widget-crash cascade).
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };
        // Publish the fully-bound scaffold — `frame` may now draw.
        *slot = Some(Gpu {
            surface,
            pipeline,
            uniform_buffer,
            instance_buffer,
            instance_capacity,
            bind_group,
            layout,
        });

        Ok(())
    }

    /// `begin` — start a sim session: `dir` +1 burns, −1 rebuilds;
    /// `origin` is the widget's CSS-px offset inside the canvas;
    /// `seed` makes the xorshift deterministic per session.
    ///
    /// @param direction +1.0 burn / −1.0 rebuild.
    /// @param origin_x Widget origin X, canvas CSS px.
    /// @param origin_y Widget origin Y, canvas CSS px.
    /// @param seed Deterministic xorshift seed (0 → session default).
    pub fn begin(&self, direction: f32, origin_x: f32, origin_y: f32, seed: u32) {
        // Mount a fresh sim session — replaces any prior one wholesale
        // (a new hover starts a new burn, not a resume).
        if let Ok(mut sim_slot) = self.sim.try_borrow_mut() {
            *sim_slot = Some(sim::BurnSim::new(direction, [origin_x, origin_y], seed));
        }
    }

    /// `resize` — canvas-bitmap change → surface re-configure.
    ///
    /// @param _width Device-pixel width (element-side).
    /// @param _height Device-pixel height.
    pub fn resize(&self, _width: u32, _height: u32) {
        // Bail while init is in flight — the surface configures
        // itself when it arrives.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Re-configure the live surface to match the new bitmap — the
        // sim works in CSS px so only the surface needs the size.
        if let Some(gpu_state) = gpu.as_ref() {
            gpu_mini::resize(&gpu_state.surface);
        }
    }

    /// `set_size` — the canvas's CSS-pixel size (viewport uniform).
    ///
    /// @param width Canvas width in CSS pixels.
    /// @param height Canvas height in CSS pixels.
    pub fn set_size(&self, width: f32, height: f32) {
        // CSS-px viewport — ≥1 clamp; the shader's CSS→NDC projection
        // divides by it, so 0 would poison every particle position.
        if let Ok(mut viewport) = self.viewport_css.try_borrow_mut() {
            *viewport = [width.max(1.0), height.max(1.0)];
        }
    }

    /// `set_letters` — the letter table: 6 f32s per glyph (rect +
    /// speeds — see `sim::BurnSim::set_letters` for the exact layout).
    ///
    /// @param flat Flat f32 letter table.
    pub fn set_letters(&self, flat: &[f32]) {
        let Ok(mut sim_ref) = self.sim.try_borrow_mut() else {
            return;
        };

        // Forward to the mounted session — `None` (pre-`begin`) drops
        // the table silently; the next mount pushes it again.
        if let Some(session) = sim_ref.as_mut() {

            session.set_letters(flat);
        }
    }

    /// `set_ink` — per-column ink occupancy bytes (the front's paper
    /// mask — where burn front edges can live).
    ///
    /// @param ink One byte per pixel-column.
    pub fn set_ink(&self, ink: &[u8]) {
        let Ok(mut sim_ref) = self.sim.try_borrow_mut() else {
            return;
        };

        // Forward the column-occupancy mask to the live session.
        if let Some(session) = sim_ref.as_mut() {

            session.set_ink(ink);
        }
    }

    /// `set_colors` — the ember ramp + scorch tint: [core, mid, edge,
    /// scorch] × rgba = 16 f32s.
    ///
    /// @param flat Flat f32 color table.
    pub fn set_colors(&self, flat: &[f32]) {
        let Ok(mut sim_ref) = self.sim.try_borrow_mut() else {
            return;
        };

        // Forward the tint table to the live session.
        if let Some(session) = sim_ref.as_mut() {

            session.set_colors(flat);
        }
    }

    /// `set_dir` — +1 burn, −1 rebuild.
    ///
    /// @param direction Direction discriminant.
    pub fn set_dir(&self, direction: f32) {
        let Ok(mut sim_ref) = self.sim.try_borrow_mut() else {
            return;
        };

        // Forward the direction flip — the sim drains its pools
        // internally (burn ↔ rebuild hover toggle).
        if let Some(session) = sim_ref.as_mut() {

            session.set_dir(direction);
        }
    }

    /// `set_stage` — per-frame stage + jagged amplitude pushed by the
    /// bridge before `frame`.
    ///
    /// @param stage Global burn progress 0..1.
    /// @param jagged_amplitude Jagged-edge amplitude in px.
    pub fn set_stage(&self, stage: f32, jagged_amplitude: f32) {
        let Ok(mut sim_ref) = self.sim.try_borrow_mut() else {
            return;
        };

        // Per-frame stage push — `tick` reads it parameter-free.
        if let Some(session) = sim_ref.as_mut() {

            session.set_stage(stage, jagged_amplitude);
        }
    }

    /// `set_origin` — move the burn origin (the widget rect moved).
    ///
    /// @param origin_x New origin X, canvas CSS px.
    /// @param origin_y New origin Y, canvas CSS px.
    pub fn set_origin(&self, origin_x: f32, origin_y: f32) {
        let Ok(mut sim_ref) = self.sim.try_borrow_mut() else {
            return;
        };

        // Forward the origin move — a scrolling tile drags its whole
        // flame front rather than burning in place.
        if let Some(session) = sim_ref.as_mut() {

            session.set_origin([origin_x, origin_y]);
        }
    }

    /// `live_particles` — the sim's live-particle count. The JS loop
    /// polls this for its settle check: the burn canvas can idle once
    /// the front is spent AND the last ember died (mirroring the JS
    /// sim's `particles.length === 0` condition).
    ///
    /// @returns Live particle count (0 when no sim is mounted).
    pub fn live_particles(&self) -> u32 {
        // Chain of Options: borrow → session → count; any missing link
        // (unborrowed, un-mounted) reports 0 — which conveniently IS
        // the "idle" answer the JS loop wants anyway.
        self.sim
            .try_borrow()
            .ok()
            .and_then(|sim_ref| sim_ref.as_ref().map(|session| session.live_count() as u32))
            .unwrap_or(0)
    }

    /// `frame` — one animation tick: advance the sim, emit instances,
    /// upload, draw. Per-frame cost: O(particles) scalar math + one
    /// instance write + one `draw(6, n)` — one JS→wasm crossing total.
    ///
    /// @param delta_seconds Seconds since the previous call.
    pub fn frame(&self, delta_seconds: f32) {
        // Advance + emit — the scratch Vec keeps its heap capacity
        // across frames (zero-allocation rule).
        let Ok(mut scratch) = self.scratch.try_borrow_mut() else {
            return;
        };
        scratch.clear();

        // Advance the sim one tick — it appends quad records into the
        // scratch; no session mounted → the Vec stays empty.
        if let Ok(mut sim_ref) = self.sim.try_borrow_mut() {
            if let Some(session) = sim_ref.as_mut() {
                session.tick(delta_seconds, &mut scratch);
            }
        }

        // Whole instance records emitted this tick.
        let count = scratch.len() / sim::QUAD_F32S;

        // Mutable scope — viewport write + instance-buffer regrowth.
        {
            let Ok(mut gpu_mut) = self.gpu.try_borrow_mut() else {
                return;
            };

            // Still `None` → init hasn't resolved.
            let Some(gpu_state) = gpu_mut.as_mut() else {
                return;
            };

            // Viewport uniform — viewport_css + 2 f32 pad = 16 B.
            // Copy the CSS size out of its cell, release the guard,
            // THEN write — no RefCell guard crosses the queue call.
            let Ok(viewport_ref) = self.viewport_css.try_borrow() else {
                return;
            };

            let viewport = *viewport_ref;

            drop(viewport_ref);

            gpu_mini::write_f32s(
                &gpu_state.surface.queue,
                &gpu_state.uniform_buffer,
                &[viewport[0], viewport[1], 0.0, 0.0],
            );

            // Grow the instance buffer geometrically when the emit
            // overflows — bind groups are immutable, so regrow ⇒ rebind.
            if count > 0 && count > gpu_state.instance_capacity {
                // Remember the old capacity so a failed regrow can
                // roll the bookkeeping back to the still-live buffer.
                let old_capacity = gpu_state.instance_capacity;

                // Geometric growth — double until the emit fits; the
                // realloc cost lands once per scale-up, never steadily.
                while count > gpu_state.instance_capacity {
                    gpu_state.instance_capacity *= 2;
                }

                // A device fault mid-regrow keeps the old (too-small)
                // buffer — the frame draws truncated rather than
                // panicking inside the rAF callback.
                if let Ok(instance_buffer) = gpu_mini::new_buffer(
                    &gpu_state.surface.device,
                    (gpu_state.instance_capacity * sim::QUAD_F32S * 4) as u32,
                    gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
                ) {
                    // Swap in the grown buffer…
                    gpu_state.instance_buffer = instance_buffer;

                    // …and rebuild the bind group against it — groups
                    // are immutable, so a buffer swap means a rebind.
                    gpu_state.bind_group = gpu_mini::new_bind_group(
                        &gpu_state.surface.device,
                        &gpu_state.layout,
                        &[
                            gpu_mini::bind_buffer_entry(0, &gpu_state.uniform_buffer),
                            gpu_mini::bind_buffer_entry(1, &gpu_state.instance_buffer),
                        ],
                    );
                } else {
                    // Regrow failed — restore the capacity so the
                    // bookkeeping still describes the old buffer.
                    gpu_state.instance_capacity = old_capacity;
                }
            }

            // Upload this frame's instance records — skipped when the
            // emit was empty (nothing to write).
            if count > 0 {
                gpu_mini::write_f32s(&gpu_state.surface.queue, &gpu_state.instance_buffer, &scratch);
            }
        }

        // Release the scratch borrow BEFORE the draw section — the
        // same no-guard-across-JS-boundary rule.
        drop(scratch);

        // Immutable scope — draw with whatever bind group is live.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Still `None` → init hasn't resolved.
        let Some(gpu_state) = gpu.as_ref() else {
            return;
        };

        // One instanced draw for every live particle; a stale/lost
        // surface re-configures itself for next frame. The `.max(1)`
        // keeps the pass alive on an empty emit so the clear still
        // composites (a fully-dead burn must erase its last frame,
        // not freeze it).
        if !gpu_mini::draw_n(
            &gpu_state.surface,
            &gpu_state.pipeline,
            &gpu_state.bind_group,
            None,
            6,
            (count as u32).max(1),
        ) {
            gpu_mini::resize(&gpu_state.surface);
        }
    }
}

/// `destroy` parity — forget the GPU state instead of destructing
/// it: the FinalizationRegistry's `__wbg_*_free` walks every stored
/// JsValue's externref dealloc — on a torn-down table that traps
/// `memory access out of bounds` during GC. GPU objects die with the
/// shared device; leaking the handles is safe and un-panickable.
#[cfg(target_arch = "wasm32")]
impl Drop for Burn {
    fn drop(&mut self) {
        if let Some(gpu_state) = self.gpu.get_mut().take() {
            std::mem::forget(gpu_state);
        }
    }
}
