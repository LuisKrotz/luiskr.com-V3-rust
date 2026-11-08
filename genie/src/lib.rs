//! genie — the expand-modal media mesh-warp transition.
//!
//! Standalone per-widget artifact — see `rust/README.md`. Replaces the
//! CSS modal zoom with the macOS genie field: the composited modal
//! sheet texture is drawn as a 192-row synthesized mesh whose vertices
//! deform through the shared per-row staggered eased lerp (the
//! ui-layouts mac-genie reference field — `core/tokens/motion/genie.ts`
//! holds the identical math the 2D backend runs), plus the reference's
//! dock-flash accent in the fragment stage. JS owns the clock and
//! pushes funnel progress + rects per frame; the crate owns the draw.
//!
//! # Geometry without buffers
//!
//! The pipeline synthesizes vertices from `vertex_index` — no vertex
//! or index buffers: the mesh is ROWS-1 quads × 6 verts (570 draws).
//!
//! # Uniform layout (f32s = 24 — mirrors
//! `GPU_WIDGET_UNIFORM_F32S[GENIE]` and the WGSL struct)
//!
//!   [0,1]    resolution_dev  canvas DEVICE px
//!   [2]      time_seconds    seconds
//!   [3]      progress        funnel progress 0..1 (0 = modal sheet,
//!                            1 = figure funnel — JS-owned linear clock)
//!   [4..8]   modal_rect      modal rect (x,y,w,h) CSS px
//!   [8..12]  figure_rect     figure rect
//!   [12,13]  reserved        (kept for the 24-f32 ABI)
//!   [14]     dpr             device/css ratio (rects arrive in CSS px)
//!   [15]     flags           bit0 opening · bit1 nearBottom · bit2 hasTex
//!   [16..20] ghost_tint      accent rgba (untex'd ghost fill)
//!   [20..24] uv_rect         texture uv scale.xy + offset.xy
//!
//! # Bindings
//!
//!   Mesh pipeline:      0 uniform · 1 texture · 2 sampler
//!
//! # Reduced motion
//!
//! JS skips the widget entirely under reduced-motion (the transition
//! simply doesn't run) — the crate has no motion gate.

#[cfg(target_arch = "wasm32")]
use std::cell::{Cell, RefCell};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::{GpuBindGroupLayout, GpuDevice, HtmlCanvasElement, ImageBitmap};

/// WGSL — the mesh deform + dock-flash shader chunk (the shared easing
/// comes from gpu-mini's prelude via `helpers_source`).
#[cfg(target_arch = "wasm32")]
const WGSL: &str = include_str!("genie.wgsl");

/// Uniform f32 count — see module doc.
#[cfg(target_arch = "wasm32")]
const UNIFORM_F32S: usize = 24;

/// Mesh vertex count — (ROWS-1) quads × 6 verts, synthesized in WGSL
/// (ROWS = 192 → 191 quads; ~4.7px/row on a 900px sheet keeps the
/// funnel's sloped walls raster-smooth — 96 rows stair-stepped them).
#[cfg(target_arch = "wasm32")]
const MESH_VERTS: u32 = 1146;

/// Flag bit 0 — set while the modal is opening (collapse plays the
/// identical funnel in reverse via the flipped progress).
#[cfg(target_arch = "wasm32")]
const FLAG_OPENING: u32 = 1u32;

/// Flag bit 1 — the figure sits below the modal (down-space rows
/// measure toward the bottom edge instead of the top).
#[cfg(target_arch = "wasm32")]
const FLAG_NEAR_BOTTOM: u32 = 2u32;

/// Flag bit 2 — the sheet texture has been uploaded; the shader
/// switches from the ghost tint to sampling the real bitmap.
#[cfg(target_arch = "wasm32")]
const FLAG_HAS_TEXTURE: u32 = 4u32;

/// GPU state — surface, pipeline, buffers, bind group, texture.
#[cfg(target_arch = "wasm32")]
struct GpuState {
    /// The live canvas surface.
    surface: gpu_mini::Surface,
    /// The mesh-deform pipeline (texture+sampler+uniform).
    mesh_pipeline: web_sys::GpuRenderPipeline,
    /// The shared 24-f32 uniform block.
    uniform_buffer: web_sys::GpuBuffer,
    /// Mesh bind-group layout — kept so `upload_bitmap` can rebuild.
    mesh_bind_layout: web_sys::GpuBindGroupLayout,
    /// Mesh bind group — rebuilt on texture upload.
    mesh_bind_group: web_sys::GpuBindGroup,
    /// The media texture (placeholder 1×1 until `upload_bitmap`).
    texture: web_sys::GpuTexture,
    /// The shared linear sampler.
    sampler: web_sys::GpuSampler,
}

/// Scalar state repacked into the uniform block per frame.
#[cfg(target_arch = "wasm32")]
struct State {
    /// Canvas device px — `resolution_dev`.
    device_px: [f32; 2],
    /// Canvas CSS px — derives `dpr` against `device_px`.
    css_px: [f32; 2],
    /// Modal rect, CSS px — the field's source (S).
    modal_rect: [f32; 4],
    /// Figure rect, CSS px — the funnel's target (D).
    figure_rect: [f32; 4],
    /// bit0 opening · bit1 nearBottom · bit2 hasTex.
    flags: u32,
    /// Accent rgba — tints the untextured ghost fill.
    ghost_tint: [f32; 4],
    /// Texture uv transform (1,1,0,0 — the modal fits the whole frame).
    uv_rect: [f32; 4],
    /// Eased funnel progress — pushed by the JS driver each frame
    /// (0 = modal sheet, 1 = figure funnel).
    progress: f32,
}

/// Public wasm surface — one genie transition per expand modal.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct Genie {
    /// GPU state — `None` until `init` resolves.
    gpu: RefCell<Option<GpuState>>,
    /// Scalar state repacked per frame.
    state: RefCell<State>,
    /// Elapsed seconds — the `time_seconds` uniform's clock.
    clock_seconds: Cell<f32>,
}

/// `pipeline` — one shader module, one entry-point pair, premult-over
/// blend. Mirrors `gpu_mini::fullscreen_pipeline` minus the vertex-
/// buffer-free fullscreen assumption (same shape — no buffers).
///
/// @param device The shared GPUDevice.
/// @param format The surface's swap format.
/// @param vertex_entry Vertex entry point name.
/// @param fragment_entry Fragment entry point name.
/// @param layouts Bind-group layouts for the pipeline.
/// @returns The render pipeline, or Err on a device fault.
#[cfg(target_arch = "wasm32")]
fn pipeline(
    device: &GpuDevice,
    format: web_sys::GpuTextureFormat,
    vertex_entry: &str,
    fragment_entry: &str,
    layouts: &[GpuBindGroupLayout],
) -> Result<web_sys::GpuRenderPipeline, JsValue> {
    use web_sys::*;

    // Compile the WGSL module — one module serves both entry points.
    // helpers_source prepends the shared prelude (ease_in_out_cubic &c).
    let source = gpu_mini::wgsl::helpers_source(WGSL);
    let shader = device.create_shader_module(&GpuShaderModuleDescriptor::new(&source));

    // Wrap each bind-group layout in JsNullable (the web_sys descriptor
    // takes a nullable array), then build the pipeline layout.
    let layout_arr: Vec<js_sys::JsNullable<GpuBindGroupLayout>> = layouts
        .iter()
        .map(|layout| js_sys::JsNullable::wrap(layout.clone()))
        .collect();

    let layout = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&layout_arr));

    // Vertex stage — entry point picked by the caller (vs_mesh here;
    // the mesh synthesizes vertices from vertex_index, no buffers).
    let vertex = GpuVertexState::new(&shader);
    vertex.set_entry_point(vertex_entry);

    // Premult-over blend: src×1 + dst×(1−src.a) — the warp draws a
    // premultiplied sheet so the ghost fill blends correctly.
    let component = GpuBlendComponent::new();
    component.set_src_factor(GpuBlendFactor::One);
    component.set_dst_factor(GpuBlendFactor::OneMinusSrcAlpha);
    component.set_operation(GpuBlendOperation::Add);
    let blend = GpuBlendState::new(&component, &component);

    // One color target in the surface's negotiated format.
    let target = GpuColorTargetState::new(format);
    target.set_blend(&blend);
    let targets = [js_sys::JsNullable::wrap(target)];

    // Fragment stage — fs_mesh (deform sample + dock-flash accent).
    let fragment = GpuFragmentState::new(&shader, &targets);
    fragment.set_entry_point(fragment_entry);

    // Assemble + create; a device fault surfaces as Err(JsValue) so
    // JS can keep the DOM modal instead of the canvas.
    let desc = GpuRenderPipelineDescriptor::new(&layout, &vertex);
    desc.set_fragment(&fragment);
    device
        .create_render_pipeline(&desc)
        .map_err(|e| JsValue::from_str(&format!("create_render_pipeline: {e:?}")))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Genie {
    /// `Genie::new` — un-initialized widget; `init` must follow.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Genie {
        // Pre-init state: no GPU yet; unit rects and identity uv keep
        // the warp degenerate-but-valid until JS pushes real geometry.
        Genie {
            gpu: RefCell::new(None),
            state: RefCell::new(State {
                device_px: [1.0, 1.0],
                css_px: [1.0, 1.0],
                modal_rect: [0.0, 0.0, 1.0, 1.0],
                figure_rect: [0.0, 0.0, 1.0, 1.0],
                flags: 0,
                ghost_tint: [1.0, 1.0, 1.0, 1.0],
                uv_rect: [1.0, 1.0, 0.0, 0.0],
                progress: 0.0,
            }),
            clock_seconds: Cell::new(0.0),
        }
    }

    /// `init` — device + the mesh pipeline + the placeholder texture.
    ///
    /// @param canvas The overlay canvas.
    /// @param device The shared GPUDevice (one per page, from the JS bridge).
    /// @returns Ok on a live surface, Err(JsValue) on failure.
    pub fn init(&self, canvas: HtmlCanvasElement, device: GpuDevice) -> Result<(), JsValue> {
        use web_sys::*;

        // Step 1 — wrap the canvas in a configured surface.
        let surface = gpu_mini::init_surface(&canvas, &device)?;

        // Step 2 — mesh bind-group layout: 0 uniform · 1 texture ·
        // 2 sampler — genie.wgsl's declared contract.
        let mesh_bind_layout = gpu_mini::new_layout(
            &surface.device,
            &[
                gpu_mini::uniform_entry(0),
                gpu_mini::texture_entry(1),
                gpu_mini::sampler_entry(2),
            ],
        )?;

        // Step 3 — compile the mesh-deform pipeline (the crate-local
        // `pipeline` helper — the mesh isn't a fullscreen triangle).
        let mesh_pipeline = pipeline(
            &surface.device,
            surface.format,
            "vs_mesh",
            "fs_mesh",
            &[mesh_bind_layout.clone()],
        )?;

        // Step 4 — the 24-f32 uniform block (CPU-writable each frame).
        let uniform_buffer = gpu_mini::new_buffer(
            &surface.device,
            (UNIFORM_F32S * 4) as u32,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
        )?;

        // Step 5 — linear sampler for the sheet texture.
        let sampler = gpu_mini::linear_sampler(&surface.device);

        // 1×1 placeholder — the mesh pipeline always has a texture to
        // bind; `has_tex` stays clear until a real bitmap uploads.
        let texture = gpu_mini::new_texture(&surface.device, 1, 1)?;

        gpu_mini::upload_rgba(&surface.queue, &texture, 1, 1, &[0u8, 0, 0, 0]);

        // Step 6 — a view for the placeholder texture…
        let texture_view = gpu_mini::view(&texture)?;

        // …and the bind group — rebuilt by `upload_bitmap` once the
        // real sheet texture lands.
        let mesh_bind_group = gpu_mini::new_bind_group(
            &surface.device,
            &mesh_bind_layout,
            &[
                gpu_mini::bind_buffer_entry(0, &uniform_buffer),
                gpu_mini::bind_view_entry(1, &texture_view),
                gpu_mini::bind_sampler_entry(2, &sampler),
            ],
        );

        // try_borrow everywhere: a JS exception crossing a wasm frame
        // mid-call leaks the RefCell guard — a later borrow() panics
        // `unreachable` (the observed widget-crash cascade).
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return Err(JsValue::from_str("init: gpu slot poisoned"));
        };
        // Publish the assembled GPU bundle — `frame` may now borrow it.
        *slot = Some(GpuState {
            surface,
            mesh_pipeline,
            uniform_buffer,
            mesh_bind_layout,
            mesh_bind_group,
            texture,
            sampler,
        });

        Ok(())
    }

    /// `set_rects` — the warp's origin + destination rects, CSS px.
    ///
    /// @param modal_x @param modal_y @param modal_width @param modal_height Origin rect.
    /// @param figure_x @param figure_y @param figure_width @param figure_height Destination rect.
    pub fn set_rects(
        &self,
        modal_x: f32,
        modal_y: f32,
        modal_width: f32,
        modal_height: f32,
        figure_x: f32,
        figure_y: f32,
        figure_width: f32,
        figure_height: f32,
    ) {
        // Store both rects verbatim — JS resolves CSS px each frame;
        // a contended slot just drops the update (pushed every frame).
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.modal_rect = [modal_x, modal_y, modal_width, modal_height];

            state.figure_rect = [figure_x, figure_y, figure_width, figure_height];
        }
    }

    /// `set_flags` — direction + facing edge.
    ///
    /// @param opening 1.0 while opening, 0.0 on the collapse.
    /// @param near_bottom 1.0 when the origin sits below the destination.
    pub fn set_flags(&self, opening: f32, near_bottom: f32) {
        if let Ok(mut state) = self.state.try_borrow_mut() {
            // Rewrite bits 0–1 only (`& !3` clears just them): bit0 =
            // opening, bit1 = nearBottom; bit2 (hasTex) is untouched.
            state.flags = (state.flags & !(FLAG_OPENING | FLAG_NEAR_BOTTOM))
                | (opening > 0.5) as u32
                | (((near_bottom > 0.5) as u32) << 1);
        }
    }

    /// `set_tint` — the accent rgba (0-1, theme-resolved in JS) the
    /// untextured ghost fill reads.
    ///
    /// @param red @param green @param blue @param alpha Rgba channels.
    pub fn set_tint(&self, red: f32, green: f32, blue: f32, alpha: f32) {
        // Store the theme-resolved rgba — only read while hasTex=0
        // (the ghost fill before the sheet texture uploads).
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.ghost_tint = [red, green, blue, alpha];
        }
    }

    /// `set_progress` — the JS-owned spring progress 0..1.
    ///
    /// @param progress Funnel progress.
    pub fn set_progress(&self, progress: f32) {
        // Store the spring progress verbatim — JS owns the ease and
        // re-pushes every frame, so a dropped write self-heals.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.progress = progress;
        }
    }

    /// `set_size` — canvas CSS px (drives the dpr uniform against
    /// `device_px`).
    ///
    /// @param width @param height CSS size.
    pub fn set_size(&self, width: f32, height: f32) {
        // CSS-px pair — ≥1 clamp so the `dpr` division below never
        // divides by a zero-sized canvas.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.css_px = [width.max(1.0), height.max(1.0)];
        }
    }

    /// `resize` — device px — the `resolution_dev` uniform + surface
    /// re-configure.
    ///
    /// @param width @param height Backing size in device pixels.
    pub fn resize(&self, width: f32, height: f32) {
        // Store the device-px backing size (`resolution_dev`); clamped
        // ≥1 like `css_px` so the dpr ratio stays finite.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.device_px = [width.max(1.0), height.max(1.0)];
        }

        // Re-configure the live surface — skipped entirely while init
        // is still in flight (nested try-borrow, no early return needed
        // since this is the function tail).
        if let Ok(gpu) = self.gpu.try_borrow() {
            if let Some(gpu_state) = gpu.as_ref() {
                gpu_mini::resize(&gpu_state.surface);
            }
        }
    }

    /// `upload_bitmap` — decode happened off-thread (createImageBitmap);
    /// GPU→GPU copy into a fresh texture, then the mesh bind group is
    /// rebuilt against it and `has_tex` goes live.
    ///
    /// @param bitmap The decoded frame.
    /// @param width @param height Bitmap pixel size.
    pub fn upload_bitmap(&self, bitmap: ImageBitmap, width: u32, height: u32) {
        // Mutable borrow — we replace `texture`/`mesh_bind_group` in
        // place.
        let Ok(mut slot) = self.gpu.try_borrow_mut() else {
            return;
        };

        // Still `None` → init hasn't resolved; drop the upload.
        let Some(gpu_state) = slot.as_mut() else {
            return;
        };

        // GPU→GPU copy of the decoded bitmap — a device fault keeps the
        // placeholder texture rather than panicking mid-upload.
        let Ok(texture) = gpu_mini::bitmap_texture(
            &gpu_state.surface.device,
            &gpu_state.surface.queue,
            &bitmap,
            width,
            height,
        ) else {
            return;
        };

        // A view for the fresh texture — same keep-stale policy.
        let Ok(texture_view) = gpu_mini::view(&texture) else {
            return;
        };

        // Bind groups are immutable — rebuild the mesh group against
        // the new view (uniform + sampler carry over unchanged).
        gpu_state.mesh_bind_group = gpu_mini::new_bind_group(
            &gpu_state.surface.device,
            &gpu_state.mesh_bind_layout,
            &[
                gpu_mini::bind_buffer_entry(0, &gpu_state.uniform_buffer),
                gpu_mini::bind_view_entry(1, &texture_view),
                gpu_mini::bind_sampler_entry(2, &gpu_state.sampler),
            ],
        );

        // Swap the texture handle — the old 1×1 placeholder drops.
        gpu_state.texture = texture;

        // Set bit2 (hasTex) — the shader switches from the tinted
        // ghost fill to sampling the real sheet.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.flags |= FLAG_HAS_TEXTURE;
        }
    }

    /// `frame` — repack uniforms + one render pass (the mesh draw).
    ///
    /// @param delta_seconds Frame delta in seconds.
    /// @returns false once the surface is stale/lost.
    pub fn frame(&self, delta_seconds: f32) -> bool {
        // Bail while init is in flight — `false` tells JS to stop
        // pumping, correct for a not-yet-live widget too.
        let Ok(gpu) = self.gpu.try_borrow() else {
            return false;
        };

        // Still `None` → init hasn't resolved.
        let Some(gpu_state) = gpu.as_ref() else {
            return false;
        };

        // Advance the `time_seconds` clock — summing dt absorbs rAF
        // jitter.
        let time_seconds = self.clock_seconds.get() + delta_seconds;

        self.clock_seconds.set(time_seconds);

        // Scoped block: the state guard must drop BEFORE the render
        // pass below — no RefCell borrow may cross the JS-boundary work.
        {
            let Ok(state) = self.state.try_borrow() else {
                return false;
            };

            // Uniform repack — the 24-f32 block documented above.
            // dpr scales the CSS-px rects into device px.
            let dpr = if state.css_px[0] > 0.0 {
                state.device_px[0] / state.css_px[0]
            } else {
                1.0
            };

            let mut uniforms = [0.0f32; UNIFORM_F32S];

            // [0,1] resolution_dev · [2] time_seconds · [3] progress.
            uniforms[0] = state.device_px[0];

            uniforms[1] = state.device_px[1];

            uniforms[2] = time_seconds;

            uniforms[3] = state.progress;

            // [4..8] modal rect, [8..12] figure rect (CSS px).
            uniforms[4..8].copy_from_slice(&state.modal_rect);

            uniforms[8..12].copy_from_slice(&state.figure_rect);

            // uniforms[12], uniforms[13] reserved — the funnel field
            // has no tunables.
            // [14] dpr, [15] flags (opening · nearBottom · hasTex).
            uniforms[14] = dpr;

            uniforms[15] = state.flags as f32;

            // [16..20] ghost tint, [20..24] uv scale+offset.
            uniforms[16..20].copy_from_slice(&state.ghost_tint);

            uniforms[20..24].copy_from_slice(&state.uv_rect);

            // Push the packed block — the write happens while the
            // state guard is still held but released at block end.
            gpu_mini::write_f32s(&gpu_state.surface.queue, &gpu_state.uniform_buffer, &uniforms);
        }

        use web_sys::*;

        // Acquire this frame's swap texture — a lost/stale surface
        // returns false so JS stops pumping.
        let texture = match gpu_state.surface.context.get_current_texture() {
            Ok(texture) => texture,
            Err(_) => return false,
        };

        // …and its view for the color attachment.
        let texture_view = match texture.create_view() {
            Ok(view) => view,
            Err(_) => return false,
        };

        // Clear to transparent then store — the mesh pass composites
        // the whole canvas.
        let clear = GpuColorDict::new(0.0, 0.0, 0.0, 0.0);
        let attachment = GpuRenderPassColorAttachment::new_with_gpu_texture_view(
            GpuLoadOp::Clear,
            GpuStoreOp::Store,
            &texture_view,
        );
        attachment.set_clear_value_gpu_color_dict(&clear);

        let attachments = [js_sys::JsNullable::wrap(attachment)];
        let pass_desc = GpuRenderPassDescriptor::new(&attachments);

        // One command encoder for the single-pass frame.
        let encoder = gpu_state
            .surface
            .device
            .create_command_encoder_with_descriptor(&GpuCommandEncoderDescriptor::new());

        // Record the mesh draw: pipeline + bind group + 1146
        // synthesized vertices (191 quads × 6, no vertex buffers).
        match encoder.begin_render_pass(&pass_desc) {
            Ok(pass) => {
                pass.set_pipeline(&gpu_state.mesh_pipeline);
                pass.set_bind_group(0, Some(&gpu_state.mesh_bind_group));
                pass.draw(MESH_VERTS);

                pass.end();
            }
            Err(_) => return false,
        }

        // Submit the recorded pass — the frame is on its way.
        gpu_state.surface.queue.submit(&[encoder.finish()]);

        true
    }
}

/// `destroy` parity — forget the GPU state instead of destructing it:
/// the FinalizationRegistry's `__wbg_*_free` walks every stored
/// JsValue's externref dealloc — on a torn-down table that traps
/// `memory access out of bounds` during GC. GPU objects die with the
/// shared device; leaking the handles is safe and un-panickable.
#[cfg(target_arch = "wasm32")]
impl Drop for Genie {
    fn drop(&mut self) {
        if let Some(gpu_state) = self.gpu.get_mut().take() {
            std::mem::forget(gpu_state);
        }
    }
}
