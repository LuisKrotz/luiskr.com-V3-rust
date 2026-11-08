//! flag — the locale flag widget.
//!
//! Standalone per-widget artifact — see `rust/README.md`. The shader
//! warps one or two flag textures: wind sine envelope (pole-fixed →
//! free-fly edge), horizontal foreshortening, specular sheen, seam
//! darkening on split flags, per-locale `flutter`/`specular`
//! personalities (`GPU_FLAG_ANIMS` on the JS side, 12 variants).
//!
//! # Uniform layout (f32s = 12 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[FLAG]` and the WGSL struct)
//!
//!   [0,1]  resolution_css canvas CSS px (contract parity — unread)
//!   [2]    time_seconds   seconds — wind phase advance
//!   [3]    hover_amount   0..1 hover amount (amplitude boost)
//!   [4]    anim_kind      locale personality 0..11 (`GPU_FLAG_ANIMS`)
//!   [5]    split_flag     0 single flag / 1 split flag (two textures)
//!   [6]    split_x        split seam position 0..1 (0.5 = halves)
//!   [7]    reserved       16B alignment pad
//!   [8..11] clip_rect     (0,0,w,h) on the dedicated canvas
//!
//! # Bindings (must match the WGSL `@binding` indices)
//!
//!   0 uniform · 1 flag_a (primary) · 2 flag_b (secondary —
//!   single-flag widgets bind flag_a twice) · 3 linear sampler
//!
//! # Frame cost
//!
//! One ~48 B uniform write + one 3-vertex draw; textures upload once
//! via `upload_textures`, not per frame.

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use web_sys::{gpu_buffer_usage, GpuDevice, HtmlCanvasElement};

/// WGSL — wind-warp + specular flag fragment chunk (the shared
/// fullscreen vs_main arrives via the gpu-mini library).
const WGSL: &str = include_str!("flag.wgsl");

/// Uniform f32 count — see module doc.
const UNIFORM_F32S: usize = 12;

/// GPU state — surface + the texture-dependent bind group (rebuilt
/// when `upload_textures` lands new pixels).
struct Gpu {
    /// The live canvas surface.
    surface: gpu_mini::Surface,
    /// The flag pipeline (fullscreen triangle, textured).
    pipeline: web_sys::GpuRenderPipeline,
    /// Uniform block — the 12-f32 layout above.
    uniform_buffer: web_sys::GpuBuffer,
    /// Shared linear sampler for both flag textures.
    sampler: web_sys::GpuSampler,
    /// Primary flag texture (current flag).
    flag_a: RefCell<Option<web_sys::GpuTexture>>,
    /// Secondary texture (split-flag second half; flag_a when absent).
    flag_b: RefCell<Option<web_sys::GpuTexture>>,
    /// The bind-group layout (kept so `rebind` can rebuild `bind_group`).
    layout: web_sys::GpuBindGroupLayout,
    /// Bound resources — rebuilt by `rebind` after texture uploads.
    bind_group: RefCell<Option<web_sys::GpuBindGroup>>,
}

/// Flag state repacked into the uniform block per frame.
struct State {
    /// Canvas CSS px — contract parity (the shader works in device
    /// px; the field is packed but the shader reads `clip_rect`).
    css_size: [f32; 2],
    /// Canvas DEVICE px — the `clip_rect.zw` divisor mapping
    /// `pixel_coord` (device px) into widget uv 0..1. Written by `resize`.
    device_size: [f32; 2],
    /// 0..1 hover amount — drives the wind-amplitude boost.
    hover_amount: f32,
    /// Locale personality discriminant (0..11).
    anim_kind: f32,
    /// 0 single flag / 1 split flag.
    split_flag: f32,
    /// Split seam position 0..1 (0.5 = halves).
    split_x: f32,
}

/// Public wasm surface — one flag canvas per instance.
#[wasm_bindgen]
pub struct Flag {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<Gpu>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Accumulated seconds — the wind phase clock.
    clock_seconds: Cell<f32>,
}

#[wasm_bindgen]
impl Flag {
    /// `Flag::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Flag {
        // Pre-init state: no GPU yet; inert defaults — personality 0,
        // unsplit geometry (seam at halves, unused until split_flag=1).
        Flag {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                css_size: [1.0, 1.0],
                device_size: [1.0, 1.0],
                hover_amount: 0.0,
                anim_kind: 0.0,
                split_flag: 0.0,
                split_x: 0.5,
            }),
            clock_seconds: Cell::new(0.0),
        }
    }

    /// `init` — device + pipeline + sampler; the bind group builds
    /// once textures land (`upload_textures` → rebind).
    ///
    /// @param canvas The widget's dedicated canvas.
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        // Step 1 — wrap the canvas in a configured surface.
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // Step 2 — bind-group layout: binding 0 uniform · 1 flag_a ·
        // 2 flag_b · 3 sampler — the WGSL's declared contract; indices
        // must not drift.
        let layout = gpu_mini::new_layout(
            &surface.device,
            &[
                gpu_mini::uniform_entry(0),
                gpu_mini::texture_entry(1),
                gpu_mini::texture_entry(2),
                gpu_mini::sampler_entry(3),
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

        // Step 4 — uniform buffer; 256 B (not 48) keeps headroom for
        // layout growth without reallocating the GPU allocation.
        let uniform_buffer = gpu_mini::new_buffer(
            &surface.device,
            256,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        )?;

        // Step 5 — one linear sampler shared by both flag textures.
        let sampler = gpu_mini::linear_sampler(&surface.device);

        // try_borrow everywhere: a JS exception crossing a wasm frame
        // mid-call leaks the RefCell guard — a later borrow() panics
        // `unreachable` (the observed widget-crash cascade).
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };
        // Publish the scaffold — textures/bind_group arrive later via
        // `upload_textures` → rebind; `frame` skips draws until then.
        *slot = Some(Gpu {
            surface,
            pipeline,
            uniform_buffer,
            sampler,
            flag_a: RefCell::new(None),
            flag_b: RefCell::new(None),
            layout,
            bind_group: RefCell::new(None),
        });

        Ok(())
    }

    /// `upload_textures` — stage one or two RGBA8 flag images. Single-
    /// flag widgets pass `bitmap_b` identical to `bitmap_a` (the shader
    /// reads flag_b only when split_flag=1 — a duplicate binding is the
    /// cheapest no-op). Rebuilds the bind group so the new textures go
    /// live next frame.
    ///
    /// `ImageBitmap` inputs — the browser decodes the SVG off-main-
    /// thread and `copyExternalImageToTexture` moves pixels GPU→GPU,
    /// so no CPU pixel round-trip ever touches the flag path.
    ///
    /// @param bitmap_a The primary flag's decoded bitmap.
    /// @param bitmap_b The secondary flag's decoded bitmap (same as
    ///   `bitmap_a` for unsplit locales — a duplicate binding is the
    ///   cheapest no-op).
    pub fn upload_textures(&self, bitmap_a: web_sys::ImageBitmap, bitmap_b: web_sys::ImageBitmap) {
        // Bail while init is in flight — the upload lands too early.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Still `None` → init hasn't resolved; nothing to upload into.
        let Some(gpu_state) = gpu.as_ref() else {
            return;
        };

        // A device fault mid-upload keeps the previous textures/bind —
        // the flag keeps waving with stale pixels rather than
        // panicking inside the upload callback.
        let (Ok(tex_a), Ok(tex_b)) = (
            gpu_mini::bitmap_texture(
                &gpu_state.surface.device,
                &gpu_state.surface.queue,
                &bitmap_a,
                bitmap_a.width(),
                bitmap_a.height(),
            ),
            gpu_mini::bitmap_texture(
                &gpu_state.surface.device,
                &gpu_state.surface.queue,
                &bitmap_b,
                bitmap_b.width(),
                bitmap_b.height(),
            ),
        ) else {
            return;
        };

        // Views for the bind group — same keep-stale policy on fault.
        let (Ok(view_a), Ok(view_b)) = (gpu_mini::view(&tex_a), gpu_mini::view(&tex_b)) else {
            return;
        };

        // Swap in the primary texture — each slot is borrowed,
        // written, and released in turn so no two guards overlap.
        let Ok(mut slot_a) = gpu_state.flag_a.try_borrow_mut() else {
            return;
        };

        *slot_a = Some(tex_a);

        drop(slot_a);

        // Swap in the secondary texture (flag_a's duplicate for
        // unsplit locales — the shader never samples it anyway).
        let Ok(mut slot_b) = gpu_state.flag_b.try_borrow_mut() else {
            return;
        };

        *slot_b = Some(tex_b);

        drop(slot_b);

        // Rebuild the bind group with the fresh views — bind groups are
        // immutable by spec, so a texture swap means a new group.
        let bind_group = gpu_mini::new_bind_group(
            &gpu_state.surface.device,
            &gpu_state.layout,
            &[
                gpu_mini::bind_buffer_entry(0, &gpu_state.uniform_buffer),
                gpu_mini::bind_view_entry(1, &view_a),
                gpu_mini::bind_view_entry(2, &view_b),
                gpu_mini::bind_sampler_entry(3, &gpu_state.sampler),
            ],
        );

        // Publish the rebuilt group — `frame` starts drawing with the
        // new textures on its very next tick.
        let Ok(mut bind_slot) = gpu_state.bind_group.try_borrow_mut() else {
            return;
        };

        *bind_slot = Some(bind_group);
    }

    /// `resize` — canvas-bitmap change → surface re-configure.
    ///
    /// @param width Device-pixel width (element-side).
    /// @param height Device-pixel height.
    pub fn resize(&self, width: u32, height: u32) {
        // Store the device-px size (clip_rect.zw); clamped ≥1 so the
        // frag→uv divisor never sees a zero-sized canvas.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.device_size = [(width.max(1)) as f32, (height.max(1)) as f32];
        }

        // Bail while init is in flight — the surface configures
        // itself when it arrives.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Re-configure the live surface to match the new bitmap.
        if let Some(gpu_state) = gpu.as_ref() {
            gpu_mini::resize(&gpu_state.surface);
        }
    }

    /// `set_size` — the canvas's CSS-pixel size (contract parity).
    ///
    /// @param width Canvas width in CSS pixels.
    /// @param height Canvas height in CSS pixels.
    pub fn set_size(&self, width: f32, height: f32) {
        // CSS-px pair — same ≥1 clamp so `resolution_dev` never scales
        // by zero.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.css_size = [width.max(1.0), height.max(1.0)];
        }
    }

    /// `set_state` — bundled scalar update: hover amount, the locale
    /// personality, and the split-flag geometry.
    ///
    /// @param hover_amount 0..1 hover amount.
    /// @param anim_kind Locale personality 0..11 (`GPU_FLAG_ANIMS`).
    /// @param split_flag 0 single / 1 split flag.
    /// @param split_x Seam position 0..1.
    pub fn set_state(&self, hover_amount: f32, anim_kind: f32, split_flag: f32, split_x: f32) {
        let Ok(mut state) = self.state.try_borrow_mut() else {
            return;
        };

        // Clamp the eased hover amount into 0..1.
        state.hover_amount = hover_amount.clamp(0.0, 1.0);

        // Personality discriminant — stored verbatim (0..11 by contract).
        state.anim_kind = anim_kind;

        // Binarize the split flag — the shader gates on >0.5.
        state.split_flag = if split_flag > 0.5 { 1.0 } else { 0.0 };

        // Clamp the seam position into the unit interval.
        state.split_x = split_x.clamp(0.0, 1.0);
    }

    /// `frame` — one animation tick: pack, write, draw. Skips silently
    /// until textures land (no bind group yet — an unbound draw would
    /// be a validation error).
    ///
    /// @param delta_seconds Seconds since the previous call.
    pub fn frame(&self, delta_seconds: f32) {
        // Bail while init is in flight — a contended slot skips a tick.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return;
        };

        // Still `None` → init hasn't resolved; nothing to draw yet.
        let Some(gpu_state) = gpu.as_ref() else {
            return;
        };

        // Borrow the bind group — contended → skip the tick.
        let Ok(bind_group) = gpu_state.bind_group.try_borrow() else {
            return;
        };

        // `None` → textures haven't landed yet; an unbound draw would
        // be a validation error, so skip silently.
        let Some(bound) = bind_group.as_ref() else {
            return;
        };

        // Advance the wind clock — summing dt absorbs rAF jitter.
        self.clock_seconds
            .set(self.clock_seconds.get() + delta_seconds);

        // Snapshot the scalar state under the guard…
        let Ok(state) = self.state.try_borrow() else {
            return;
        };

        // [0,1] resolution_css [2] time_seconds [3] hover_amount
        // [4] anim_kind [5] split_flag [6] split_x [7] reserved
        // [8..11] clip_rect = (0,0,w,h)
        let uniforms: [f32; UNIFORM_F32S] = [
            state.css_size[0],
            state.css_size[1],
            self.clock_seconds.get(),
            state.hover_amount,
            state.anim_kind,
            state.split_flag,
            state.split_x,
            0.0,
            0.0,
            0.0,
            state.device_size[0],
            state.device_size[1],
        ];

        // Release the state borrow BEFORE the GPU calls — no RefCell
        // guard may be held across the JS-boundary work.
        drop(state);

        // Push the packed block to the GPU…
        gpu_mini::write_f32s(&gpu_state.surface.queue, &gpu_state.uniform_buffer, &uniforms);

        // …and draw; a stale/lost surface re-configures so the next
        // frame recovers cleanly.
        if !gpu_mini::draw(&gpu_state.surface, &gpu_state.pipeline, bound, None) {
            gpu_mini::resize(&gpu_state.surface);
        }
    }
}

/// `destroy` parity — forget the GPU state instead of destructing
/// it: the FinalizationRegistry's `__wbg_*_free` walks every stored
/// JsValue's externref dealloc — on a torn-down table that traps
/// `memory access out of bounds` during GC. GPU objects die with the
/// shared device; leaking the handles is safe and un-panickable.
impl Drop for Flag {
    fn drop(&mut self) {
        if let Some(gpu_state) = self.gpu.get_mut().take() {
            std::mem::forget(gpu_state);
        }
    }
}
