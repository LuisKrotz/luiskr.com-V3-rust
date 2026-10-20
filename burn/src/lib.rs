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
const START_CAP: usize = 512;

/// GPU state — surface + the instanced-quad resources.
#[cfg(target_arch = "wasm32")]
struct Gpu {
    /// The live canvas surface.
    surface: gpu_mini::Surface,
    /// The instanced-quad pipeline.
    pipeline: web_sys::GpuRenderPipeline,
    /// Viewport uniform (16 B — vec2 + pad).
    ubuf: web_sys::GpuBuffer,
    /// Instance storage — capacity `inst_cap` records.
    inst_buf: web_sys::GpuBuffer,
    /// Instance capacity in records — geometric growth.
    inst_cap: usize,
    /// Bound resources — rebuilt whenever `inst_buf` regrows.
    bind: web_sys::GpuBindGroup,
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
    css: RefCell<[f32; 2]>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Burn {
    /// `Burn::new` — un-initialized surface; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Burn {
        Burn {
            gpu: RefCell::new(None),
            sim: RefCell::new(None),
            scratch: RefCell::new(Vec::with_capacity(START_CAP * sim::QUAD_F32S)),
            css: RefCell::new([1.0, 1.0]),
        }
    }

    /// `init` — device + pipeline + buffers. Rejection → JS keeps the
    /// 2D-canvas burn; mounting can never regress the page.
    ///
    /// @param canvas The widget's dedicated canvas.
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // binding 0 uniform viewport · 1 storage instances — burn.wgsl.
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
            16,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        )?;

        let inst_cap = START_CAP;
        let inst_buf = gpu_mini::new_buffer(
            &surface.device,
            (inst_cap * sim::QUAD_F32S * 4) as u32,
            gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
        )?;

        let bind = gpu_mini::new_bind_group(
            &surface.device,
            &layout,
            &[
                gpu_mini::bind_buffer_entry(0, &ubuf),
                gpu_mini::bind_buffer_entry(1, &inst_buf),
            ],
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
            inst_buf,
            inst_cap,
            bind,
            layout,
        });

        Ok(())
    }

    /// `begin` — start a sim session: `dir` +1 burns, −1 rebuilds;
    /// `origin` is the widget's CSS-px offset inside the canvas;
    /// `seed` makes the xorshift deterministic per session.
    ///
    /// @param dir +1.0 burn / −1.0 rebuild.
    /// @param ox Widget origin X, canvas CSS px.
    /// @param oy Widget origin Y, canvas CSS px.
    /// @param seed Deterministic xorshift seed (0 → session default).
    pub fn begin(&self, dir: f32, ox: f32, oy: f32, seed: u32) {
        if let Ok(mut s) = self.sim.try_borrow_mut() {
            *s = Some(sim::BurnSim::new(dir, [ox, oy], seed));
        }
    }

    /// `resize` — canvas-bitmap change → surface re-configure.
    ///
    /// @param _w Device-pixel width (element-side).
    /// @param _h Device-pixel height.
    pub fn resize(&self, _w: u32, _h: u32) {
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        if let Some(g) = gpu.as_ref() {
            gpu_mini::resize(&g.surface);
        }
    }

    /// `set_size` — the canvas's CSS-pixel size (viewport uniform).
    ///
    /// @param w Canvas width in CSS pixels.
    /// @param h Canvas height in CSS pixels.
    pub fn set_size(&self, w: f32, h: f32) {
        if let Ok(mut c) = self.css.try_borrow_mut() {
            *c = [w.max(1.0), h.max(1.0)];
        }
    }

    /// `set_letters` — the letter table: 6 f32s per glyph (rect +
    /// speeds — see `sim::BurnSim::set_letters` for the exact layout).
    ///
    /// @param flat Flat f32 letter table.
    pub fn set_letters(&self, flat: &[f32]) {
        let Ok(mut s_ref) = self.sim.try_borrow_mut() else {
            return;
        };

        if let Some(s) = s_ref.as_mut() {

            s.set_letters(flat);
        }
    }

    /// `set_ink` — per-column ink occupancy bytes (the front's paper
    /// mask — where burn front edges can live).
    ///
    /// @param ink One byte per pixel-column.
    pub fn set_ink(&self, ink: &[u8]) {
        let Ok(mut s_ref) = self.sim.try_borrow_mut() else {
            return;
        };

        if let Some(s) = s_ref.as_mut() {

            s.set_ink(ink);
        }
    }

    /// `set_colors` — the ember ramp + scorch tint: [core, mid, edge,
    /// scorch] × rgba = 16 f32s.
    ///
    /// @param flat Flat f32 color table.
    pub fn set_colors(&self, flat: &[f32]) {
        let Ok(mut s_ref) = self.sim.try_borrow_mut() else {
            return;
        };

        if let Some(s) = s_ref.as_mut() {

            s.set_colors(flat);
        }
    }

    /// `set_dir` — +1 burn, −1 rebuild.
    ///
    /// @param dir Direction discriminant.
    pub fn set_dir(&self, dir: f32) {
        let Ok(mut s_ref) = self.sim.try_borrow_mut() else {
            return;
        };

        if let Some(s) = s_ref.as_mut() {

            s.set_dir(dir);
        }
    }

    /// `set_stage` — per-frame stage + jagged amplitude pushed by the
    /// bridge before `frame`.
    ///
    /// @param stage Global burn progress 0..1.
    /// @param jag_amp Jagged-edge amplitude in px.
    pub fn set_stage(&self, stage: f32, jag_amp: f32) {
        let Ok(mut s_ref) = self.sim.try_borrow_mut() else {
            return;
        };

        if let Some(s) = s_ref.as_mut() {

            s.set_stage(stage, jag_amp);
        }
    }

    /// `set_origin` — move the burn origin (the widget rect moved).
    ///
    /// @param ox New origin X, canvas CSS px.
    /// @param oy New origin Y, canvas CSS px.
    pub fn set_origin(&self, ox: f32, oy: f32) {
        let Ok(mut s_ref) = self.sim.try_borrow_mut() else {
            return;
        };

        if let Some(s) = s_ref.as_mut() {

            s.set_origin([ox, oy]);
        }
    }

    /// `live_particles` — the sim's live-particle count. The JS loop
    /// polls this for its settle check: the burn canvas can idle once
    /// the front is spent AND the last ember died (mirroring the JS
    /// sim's `particles.length === 0` condition).
    ///
    /// @returns Live particle count (0 when no sim is mounted).
    pub fn live_particles(&self) -> u32 {
        self.sim
            .try_borrow()
            .ok()
            .and_then(|r| r.as_ref().map(|s| s.live_count() as u32))
            .unwrap_or(0)
    }

    /// `frame` — one animation tick: advance the sim, emit instances,
    /// upload, draw. Per-frame cost: O(particles) scalar math + one
    /// instance write + one `draw(6, n)` — one JS→wasm crossing total.
    ///
    /// @param dt Seconds since the previous call.
    pub fn frame(&self, dt: f32) {
        // Advance + emit — the scratch Vec keeps its heap capacity
        // across frames (zero-allocation rule).
        let Ok(mut scratch) = self.scratch.try_borrow_mut() else {
            return;
        };
        scratch.clear();

        if let Ok(mut s_ref) = self.sim.try_borrow_mut() {
            if let Some(s) = s_ref.as_mut() {
                s.tick(dt, &mut scratch);
            }
        }

        let count = scratch.len() / sim::QUAD_F32S;

        // Mutable scope — viewport write + instance-buffer regrowth.
        {
            let Ok(mut g_mut) = self.gpu.try_borrow_mut() else {
                return;
            };

            let Some(g) = g_mut.as_mut() else {
                return;
            };

            // Viewport uniform — viewport_px + 2 f32 pad = 16 B.
            let Ok(css_ref) = self.css.try_borrow() else {
                return;
            };
            let css = *css_ref;
            drop(css_ref);
            gpu_mini::write_f32s(&g.surface.queue, &g.ubuf, &[css[0], css[1], 0.0, 0.0]);

            // Grow the instance buffer geometrically when the emit
            // overflows — bind groups are immutable, so regrow ⇒ rebind.
            if count > 0 && count > g.inst_cap {
                let old_cap = g.inst_cap;

                while count > g.inst_cap {
                    g.inst_cap *= 2;
                }

                // A device fault mid-regrow keeps the old (too-small)
                // buffer — the frame draws truncated rather than
                // panicking inside the rAF callback.
                if let Ok(inst_buf) = gpu_mini::new_buffer(
                    &g.surface.device,
                    (g.inst_cap * sim::QUAD_F32S * 4) as u32,
                    gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
                ) {
                    g.inst_buf = inst_buf;

                    g.bind = gpu_mini::new_bind_group(
                        &g.surface.device,
                        &g.layout,
                        &[
                            gpu_mini::bind_buffer_entry(0, &g.ubuf),
                            gpu_mini::bind_buffer_entry(1, &g.inst_buf),
                        ],
                    );
                } else {
                    g.inst_cap = old_cap;
                }
            }

            if count > 0 {
                gpu_mini::write_f32s(&g.surface.queue, &g.inst_buf, &scratch);
            }
        }

        drop(scratch);

        // Immutable scope — draw with whatever bind group is live.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        let Some(g) = gpu.as_ref() else {
            return;
        };

        if !draw_instanced(&g.surface, &g.pipeline, &g.bind, count as u32) {
            gpu_mini::resize(&g.surface);
        }
    }
}

/// `draw_instanced` — the burn draw: fullscreen-quad topology, 6 verts
/// × `instances`. Like `gpu_mini::draw` but with an instance count.
///
/// @param surface The live surface.
/// @param pipeline The instanced-quad pipeline.
/// @param bind Its bind group.
/// @param instances Live instance count.
/// @returns true when a frame was submitted.
#[cfg(target_arch = "wasm32")]
fn draw_instanced(
    surface: &gpu_mini::Surface,
    pipeline: &web_sys::GpuRenderPipeline,
    bind: &web_sys::GpuBindGroup,
    instances: u32,
) -> bool {
    let tex = match surface.context.get_current_texture() {
        Ok(t) => t,
        Err(_) => return false,
    };

    let view = match tex.create_view() {
        Ok(v) => v,
        Err(_) => return false,
    };

    let clear = web_sys::GpuColorDict::new(0.0, 0.0, 0.0, 0.0);
    let attach = web_sys::GpuRenderPassColorAttachment::new_with_gpu_texture_view(
        web_sys::GpuLoadOp::Clear,
        web_sys::GpuStoreOp::Store,
        &view,
    );
    attach.set_clear_value_gpu_color_dict(&clear);

    let attachments = [js_sys::JsNullable::wrap(attach)];
    let pass_desc = web_sys::GpuRenderPassDescriptor::new(&attachments);

    let encoder = surface
        .device
        .create_command_encoder_with_descriptor(&web_sys::GpuCommandEncoderDescriptor::new());

    match encoder.begin_render_pass(&pass_desc) {
        Ok(pass) => {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(bind));
            pass.draw_with_instance_count(6, instances.max(1));
            pass.end();
        }
        Err(_) => return false,
    }

    surface.queue.submit(&[encoder.finish()]);
    true
}

/// `destroy` parity — forget the GPU state instead of destructing
/// it: the FinalizationRegistry's `__wbg_*_free` walks every stored
/// JsValue's externref dealloc — on a torn-down table that traps
/// `memory access out of bounds` during GC. GPU objects die with the
/// shared device; leaking the handles is safe and un-panickable.
#[cfg(target_arch = "wasm32")]
impl Drop for Burn {
    fn drop(&mut self) {
        if let Some(g) = self.gpu.get_mut().take() {
            std::mem::forget(g);
        }
    }
}
