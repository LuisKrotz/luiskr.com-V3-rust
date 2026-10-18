//! gpu-mini — the shared WebGPU boilerplate every widget crate links.
//!
//! # Why this exists (and why it's NOT wgpu)
//!
//! `wgpu` on wasm32 links the whole Rust WebGPU stack — naga, the hal,
//! validation layers — ~3 MB into every artifact. The widgets here
//! need maybe 30 WebGPU calls total; `web-sys` emits thin JS bindings
//! where unused functions vanish at link time. A widget crate built on
//! gpu-mini lands at tens of KB of wasm — small enough that the
//! skeleton rides the critical path and every other animation lazy-
//! loads only when its DOM surface mounts.
//!
//! # How it differs from a traditional structure
//!
//! A traditional renderer owns one context and multiplexes surfaces —
//! here EVERY widget is its own wasm module with its own GPUDevice on
//! its own canvas (the DOM architecture already gave each widget a
//! dedicated canvas; GPUDevices are cheap, and isolated contexts mean
//! a shader compile or device loss in one widget can never stall
//! another). What IS shared is this crate: context init, pipeline
//! assembly, uniform/sheet buffers, the clear→draw→submit frame —
//! written once, audited once, linked into each artifact with dead
//! code stripped.
//!
//! # The frame contract every widget implements
//!
//! ```text
//! JS bridge (rAF owner, single clock)
//!   │  Widget::frame(dt)
//!   ▼
//! rust: repack uniform f32s → write_buffer (~64 B)
//!       encoder → begin_pass → set_pipeline → set_bind_group →
//!       draw(3) → end → submit → the canvas presents the texture
//!       when the current task ends (no explicit present in WebGPU JS).
//! ```

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::*;

/// `Surface` — a live WebGPU canvas: context + device + queue + the
/// negotiated format. Every widget crate holds one; creation is async
/// because adapter/device requests are browser promises.
pub struct Surface {
    /// The canvas's `webgpu` context — get_current_texture/configure.
    pub context: GpuCanvasContext,
    /// The logical device — factory for every GPU object.
    pub device: GpuDevice,
    /// The submission queue — write_buffer/submit.
    pub queue: GpuQueue,
    /// The canvas-preferred format (usually bgra8unorm).
    pub format: GpuTextureFormat,
}

/// `init_surface` — the one async step: `navigator.gpu` → adapter →
/// device → canvas context → configure. Any failure returns Err so
/// the JS bridge falls back to the WebGL/CSS widget — mounting a
/// widget crate can never regress the page.
///
/// @param canvas The dedicated canvas this widget draws into.
/// @returns A configured `Surface`, or a JsValue error string.
pub async fn init_surface(canvas: &HtmlCanvasElement) -> Result<Surface, JsValue> {
    let window = web_sys::window().ok_or("no window")?;
    let gpu = window.navigator().gpu();

    let adapter = JsFuture::from(gpu.request_adapter())
        .await
        .map_err(|e| JsValue::from_str(&format!("adapter: {e:?}")))?
        .dyn_into::<GpuAdapter>()
        .map_err(|_| JsValue::from_str("adapter: unavailable"))?;

    let device = JsFuture::from(adapter.request_device())
        .await
        .map_err(|e| JsValue::from_str(&format!("device: {e:?}")))?;
    let queue = device.queue();

    // `get_context("webgpu")` returns a generic Object; unchecked_into
    // is safe because the browser only returns a GPUCanvasContext for
    // that context name.
    let context = canvas
        .get_context("webgpu")
        .map_err(|e| JsValue::from_str(&format!("context: {e:?}")))?
        .ok_or("context: webgpu unavailable")?
        .unchecked_into::<GpuCanvasContext>();

    let format = gpu.get_preferred_canvas_format();

    configure(&context, &device, format);

    Ok(Surface {
        context,
        device,
        queue,
        format,
    })
}

/// `configure` — (re)bind the canvas context to the device. Called on
/// init AND on every canvas resize: WebGPU requires `configure` after
/// the canvas bitmap resizes or `getCurrentTexture` yields stale-size
/// textures. `alphaMode: "premultiplied"` matches the widgets'
/// premultiplied shader output.
///
/// @param context The canvas's webgpu context.
/// @param device The widget's GPUDevice.
/// @param format The preferred canvas format captured at init.
pub fn configure(context: &GpuCanvasContext, device: &GpuDevice, format: GpuTextureFormat) {
    let desc = GpuCanvasConfiguration::new(device, format);
    desc.set_alpha_mode(GpuCanvasAlphaMode::Premultiplied);
    let _ = context.configure(&desc);
}

/// `resize` — canvas-bitmap change → context re-configure. The bridge
/// sets the canvas element's width/height to CSS×DPR first; this just
/// re-runs configure so the next `get_current_texture` matches.
///
/// @param surface The live surface.
pub fn resize(surface: &Surface) {
    configure(&surface.context, &surface.device, surface.format);
}

// ── Buffers ─────────────────────────────────────────────────────────

/// `new_buffer` — a GPU buffer by (size, usage). `usage` takes the
/// `GpuBufferUsage` bitmask constants (UNIFORM|COPY_DST etc.).
///
/// @param device The factory device.
/// @param size Byte size.
/// @param usage GpuBufferUsage bitmask.
/// @returns The created buffer.
pub fn new_buffer(device: &GpuDevice, size: u32, usage: u32) -> GpuBuffer {
    device.create_buffer(&GpuBufferDescriptor::new(size, usage)).expect("create_buffer")
}

/// `write_f32s` — upload an f32 slice into a buffer. The browser
/// copies the bytes synchronously; the copy is ~64 B per widget
/// uniform block — the per-frame upload cost is a rounding error.
///
/// @param queue Submission queue.
/// @param buffer Target buffer.
/// @param values Source f32s (viewed as raw bytes).
pub fn write_f32s(queue: &GpuQueue, buffer: &GpuBuffer, values: &[f32]) {
    let _ = queue.write_buffer_with_u32_and_u8_slice(buffer, 0, f32_bytes(values));
}

/// `f32_bytes` — view an f32 slice as bytes without copying. f32 is
/// Pod: any bit pattern is valid and the cast is alignment-safe.
///
/// @param values Source f32 slice.
/// @returns The same memory as `&[u8]`.
pub fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32→u8 is a Pod reinterpret; len*4 can't overflow.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

// ── Bind groups ─────────────────────────────────────────────────────

/// `uniform_entry` — a `@binding(n) var<uniform>` layout entry,
/// FRAGMENT-visible (every widget reads uniforms in fs_main only).
///
/// @param binding The @binding index in the WGSL.
/// @returns The layout entry.
pub fn uniform_entry(binding: u32) -> GpuBindGroupLayoutEntry {
    let e = GpuBindGroupLayoutEntry::new(binding, gpu_shader_stage::FRAGMENT);
    let l = GpuBufferBindingLayout::new();
    l.set_type(GpuBufferBindingType::Uniform);
    l.set_has_dynamic_offset(false);
    e.set_buffer(&l);
    e
}

/// `storage_entry` — a `@binding(n) var<storage, read>` layout entry
/// (the skeleton sheet is the only user today).
///
/// @param binding The @binding index in the WGSL.
/// @returns The layout entry.
pub fn storage_entry(binding: u32) -> GpuBindGroupLayoutEntry {
    let e = GpuBindGroupLayoutEntry::new(binding, gpu_shader_stage::FRAGMENT);
    let l = GpuBufferBindingLayout::new();
    l.set_type(GpuBufferBindingType::ReadOnlyStorage);
    l.set_has_dynamic_offset(false);
    e.set_buffer(&l);
    e
}

/// `texture_entry` — a filterable float `@binding(n) var texture_2d`
/// layout entry (flag halves, media frames).
///
/// @param binding The @binding index.
/// @returns The layout entry.
pub fn texture_entry(binding: u32) -> GpuBindGroupLayoutEntry {
    let e = GpuBindGroupLayoutEntry::new(binding, gpu_shader_stage::FRAGMENT);
    let l = GpuTextureBindingLayout::new();
    e.set_texture(&l);
    e
}

/// `sampler_entry` — a filtering `@binding(n) var sampler` entry.
///
/// @param binding The @binding index.
/// @returns The layout entry.
pub fn sampler_entry(binding: u32) -> GpuBindGroupLayoutEntry {
    let e = GpuBindGroupLayoutEntry::new(binding, gpu_shader_stage::FRAGMENT);
    let l = GpuSamplerBindingLayout::new();
    l.set_type(GpuSamplerBindingType::Filtering);
    e.set_sampler(&l);
    e
}

/// `new_layout` — assemble a bind-group layout from entries.
///
/// @param device The factory device.
/// @param entries Layout entries (see `uniform_entry` etc.).
/// @returns The bind-group layout.
pub fn new_layout(device: &GpuDevice, entries: &[GpuBindGroupLayoutEntry]) -> GpuBindGroupLayout {
    device.create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(entries)).expect("create_bind_group_layout")
}

/// `bind_buffer_entry` — a bind-group entry pointing at a buffer.
///
/// @param binding The @binding index.
/// @param buffer The bound buffer.
/// @returns The entry.
pub fn bind_buffer_entry(binding: u32, buffer: &GpuBuffer) -> GpuBindGroupEntry {
    GpuBindGroupEntry::new_with_gpu_buffer_binding(binding, &GpuBufferBinding::new(buffer))
}

/// `bind_view_entry` — a bind-group entry pointing at a texture view.
///
/// @param binding The @binding index.
/// @param view The texture view.
/// @returns The entry.
pub fn bind_view_entry(binding: u32, view: &GpuTextureView) -> GpuBindGroupEntry {
    GpuBindGroupEntry::new_with_gpu_texture_view(binding, view)
}

/// `bind_sampler_entry` — a bind-group entry pointing at a sampler.
///
/// @param binding The @binding index.
/// @param sampler The sampler.
/// @returns The entry.
pub fn bind_sampler_entry(binding: u32, sampler: &GpuSampler) -> GpuBindGroupEntry {
    GpuBindGroupEntry::new(binding, sampler)
}

/// `new_bind_group` — assemble a bind group from entries + layout.
///
/// @param device The factory device.
/// @param layout Matching layout.
/// @param entries Resource entries.
/// @returns The bind group.
pub fn new_bind_group(
    device: &GpuDevice,
    layout: &GpuBindGroupLayout,
    entries: &[GpuBindGroupEntry],
) -> GpuBindGroup {
    device.create_bind_group(&GpuBindGroupDescriptor::new(entries, layout))
}

// ── Pipeline ────────────────────────────────────────────────────────

/// `fullscreen_pipeline` — the widget-family pipeline: a fullscreen
/// triangle (3 verts synthesized from @builtin(vertex_index), no
/// vertex buffers), premultiplied-alpha blending, no depth.
///
/// Why premultiplied: every widget shader emits `rgb × a` already —
/// `src + dst·(1−src.a)` is the correct composite and matches the
/// canvas's premultiplied alpha mode.
///
/// @param device The factory device.
/// @param format Canvas format.
/// @param wgsl The shader source.
/// @param layouts Bind-group layouts (usually exactly one).
/// @returns The compiled render pipeline.
pub fn fullscreen_pipeline(
    device: &GpuDevice,
    format: GpuTextureFormat,
    wgsl: &str,
    layouts: &[GpuBindGroupLayout],
) -> GpuRenderPipeline {
    let shader = device.create_shader_module(&GpuShaderModuleDescriptor::new(wgsl));

    let layout_arr: Vec<js_sys::JsNullable<GpuBindGroupLayout>> = layouts
        .iter()
        .map(|l| js_sys::JsNullable::wrap(l.clone()))
        .collect();
    let layout = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&layout_arr));

    let vertex = GpuVertexState::new(&shader);
    vertex.set_entry_point("vs_main");

    // Premultiplied-over blend: src·1 + dst·(1−src.a), same for alpha.
    let comp = GpuBlendComponent::new();
    comp.set_src_factor(GpuBlendFactor::One);
    comp.set_dst_factor(GpuBlendFactor::OneMinusSrcAlpha);
    comp.set_operation(GpuBlendOperation::Add);
    let blend = GpuBlendState::new(&comp, &comp);

    let target = GpuColorTargetState::new(format);
    target.set_blend(&blend);
    let targets = [js_sys::JsNullable::wrap(target)];

    let fragment = GpuFragmentState::new(&shader, &targets);
    fragment.set_entry_point("fs_main");

    let desc = GpuRenderPipelineDescriptor::new(&layout, &vertex);
    desc.set_fragment(&fragment);
    device
        .create_render_pipeline(&desc)
        .expect("create_render_pipeline")
}

/// `linear_sampler` — the shared bilinear sampler for textured
/// widgets: linear min/mag, clamp-to-edge (media UV crops never wrap;
/// flag textures mustn't bleed across their edge).
///
/// @param device The factory device.
/// @returns The sampler.
pub fn linear_sampler(device: &GpuDevice) -> GpuSampler {
    let d = GpuSamplerDescriptor::new();
    d.set_address_mode_u(GpuAddressMode::ClampToEdge);
    d.set_address_mode_v(GpuAddressMode::ClampToEdge);
    d.set_min_filter(GpuFilterMode::Linear);
    d.set_mag_filter(GpuFilterMode::Linear);
    device.create_sampler_with_descriptor(&d)
}

// ── Textures ────────────────────────────────────────────────────────

/// `new_texture` — an RGBA8 texture for widget uploads (flag halves,
/// media frames). Usage = TEXTURE_BINDING | COPY_DST — bilinear samples
/// a single level, no render-attachment needed.
///
/// @param device The factory device.
/// @param w Texture width px.
/// @param h Texture height px.
/// @returns The texture (single mip level).
pub fn new_texture(device: &GpuDevice, w: u32, h: u32) -> GpuTexture {
    let size = [
        js_sys::Number::from(w),
        js_sys::Number::from(h),
        js_sys::Number::from(1u32),
    ];
    device
        .create_texture(&GpuTextureDescriptor::new(
            GpuTextureFormat::Rgba8unorm,
            &size,
            gpu_texture_usage::TEXTURE_BINDING | gpu_texture_usage::COPY_DST,
        ))
        .expect("create_texture")
}

/// `upload_rgba` — write a packed RGBA8 byte slice into a texture.
/// `bytes_per_row` = w×4 — the tightly-packed default.
///
/// @param queue Submission queue.
/// @param texture Target texture.
/// @param w Texture width px.
/// @param h Texture height px.
/// @param rgba Packed RGBA8 bytes (w·h·4 long).
pub fn upload_rgba(queue: &GpuQueue, texture: &GpuTexture, w: u32, h: u32, rgba: &[u8]) {
    let dest = GpuTexelCopyTextureInfo::new(texture);
    let layout = GpuTexelCopyBufferLayout::new();
    layout.set_bytes_per_row(w * 4);
    layout.set_rows_per_image(h);
    let size = GpuExtent3dDict::new(w);
    size.set_height(h);

    let _ = queue.write_texture_with_u8_slice_and_gpu_extent_3d_dict(&dest, rgba, &layout, &size);
}

/// `view` — a texture's default view (all mips, 2D) for binding.
///
/// @param texture The texture.
/// @returns Its view.
pub fn view(texture: &GpuTexture) -> GpuTextureView {
    texture.create_view().expect("create_view")
}

/// `bitmap_texture` — create an RGBA8 texture and copy a decoded
/// `ImageBitmap` into it via `copyExternalImageToTexture`. The browser
/// does the decode→GPU path entirely off-main-thread — no `image`
/// crate, no CPU pixel round-trip.
///
/// @param device The factory device.
/// @param queue Submission queue.
/// @param bitmap The decoded bitmap (JS `createImageBitmap`).
/// @param w Bitmap width px.
/// @param h Bitmap height px.
/// @returns The populated texture.
pub fn bitmap_texture(
    device: &GpuDevice,
    queue: &GpuQueue,
    bitmap: &ImageBitmap,
    w: u32,
    h: u32,
) -> GpuTexture {
    let tex = new_texture(device, w.max(1), h.max(1));

    let src = GpuCopyExternalImageSourceInfo::new(bitmap);
    let dest = GpuCopyExternalImageDestInfo::new(&tex);
    let size = GpuExtent3dDict::new(w.max(1));
    size.set_height(h.max(1));

    let _ = queue.copy_external_image_to_texture_with_gpu_extent_3d_dict(&src, &dest, &size);
    tex
}

// ── Widget scaffold ─────────────────────────────────────────────────

/// `Widget` — the uniform-only widget scaffold most crates share:
/// one pipeline, one uniform buffer, one bind group. Textured/storage
/// widgets (flag, skeleton, media) build on the primitives directly —
/// this scaffold is for the 90% case.
pub struct Widget {
    /// The live canvas surface.
    pub surface: Surface,
    /// The widget's fullscreen-triangle pipeline.
    pub pipeline: GpuRenderPipeline,
    /// Uniform block (256 B — every widget block fits).
    pub ubuf: GpuBuffer,
    /// Pre-built bind group (ubuf at binding 0).
    pub bind: GpuBindGroup,
}

/// `init_widget` — surface + layout(binding 0 uniform) + pipeline +
/// uniform buffer + bind group, in one call.
///
/// @param canvas The widget's dedicated canvas.
/// @param wgsl The widget's shader.
/// @returns A ready `Widget`, or Err for the JS fallback path.
pub async fn init_widget(canvas: &HtmlCanvasElement, wgsl: &str) -> Result<Widget, JsValue> {
    let surface = init_surface(canvas).await?;

    let layout = new_layout(&surface.device, &[uniform_entry(0)]);
    let pipeline = fullscreen_pipeline(&surface.device, surface.format, wgsl, &[layout.clone()]);
    let ubuf = new_buffer(
        &surface.device,
        256,
        gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
    );
    let bind = new_bind_group(&surface.device, &layout, &[bind_buffer_entry(0, &ubuf)]);

    Ok(Widget {
        surface,
        pipeline,
        ubuf,
        bind,
    })
}

/// `widget_frame` — the per-frame body for uniform-only widgets:
/// write the packed block + draw. Returns false on a stale surface
/// (caller re-configures and recovers next rAF).
///
/// @param w The live widget.
/// @param uniforms The packed f32 block (layout per each widget's doc).
/// @returns true when a frame was submitted.
pub fn widget_frame(w: &Widget, uniforms: &[f32]) -> bool {
    write_f32s(&w.surface.queue, &w.ubuf, uniforms);
    draw(&w.surface, &w.pipeline, &w.bind, None)
}

// ── Frame ───────────────────────────────────────────────────────────

/// `draw` — the per-frame body shared by every widget:
/// `get_current_texture` → pass → pipeline → bind → scissor → 3 verts
/// → submit. Returns false when the surface is lost/stale (resize
/// mid-frame, occluded tab) — the caller re-configures and retries
/// next rAF; a missed frame beats a panic.
///
/// `scissor` is optional [x,y,w,h] in device px — None draws the whole
/// canvas (per-widget canvases are already widget-sized, so most
/// callers pass None; the param exists for sub-rect surfaces).
///
/// @param surface The live surface.
/// @param pipeline The widget's pipeline.
/// @param bind Its bind group.
/// @param scissor Optional clip rect in device px.
/// @returns true when a frame was submitted.
pub fn draw(
    surface: &Surface,
    pipeline: &GpuRenderPipeline,
    bind: &GpuBindGroup,
    scissor: Option<[u32; 4]>,
) -> bool {
    let tex = match surface.context.get_current_texture() {
        Ok(t) => t,
        Err(_) => return false,
    };

    let view = match tex.create_view() {
        Ok(v) => v,
        Err(_) => return false,
    };

    let clear = GpuColorDict::new(0.0, 0.0, 0.0, 0.0);
    let attach = GpuRenderPassColorAttachment::new_with_gpu_texture_view(
        GpuLoadOp::Clear,
        GpuStoreOp::Store,
        &view,
    );
    attach.set_clear_value_gpu_color_dict(&clear);

    let attachments = [js_sys::JsNullable::wrap(attach)];
    let pass_desc = GpuRenderPassDescriptor::new(&attachments);

    let encoder = surface
        .device
        .create_command_encoder_with_descriptor(&GpuCommandEncoderDescriptor::new());

    match encoder.begin_render_pass(&pass_desc) {
        Ok(pass) => {
            pass.set_pipeline(pipeline);
            if let Some([x, y, w, h]) = scissor {
                pass.set_scissor_rect(x, y, w, h);
            }
            pass.set_bind_group(0, Some(bind));
            pass.draw(3);
            pass.end();
        }
        Err(_) => return false,
    }

    surface.queue.submit(&[encoder.finish()]);

    // The GpuTexture handle drops here — the browser presents the
    // canvas frame when the current task completes.
    true
}

/// `draw_n` — the instanced/vertex-counted variant of `draw`: same
/// acquire→pass→submit shape, but the draw call takes an explicit
/// vertex count + instance count. Used by instanced widgets
/// (text-draw's per-char quads) whose vertex shader synthesizes
/// geometry from `vertex_index`+`instance_index` against a storage
/// buffer — still no vertex buffers.
///
/// @param surface The live canvas surface.
/// @param pipeline The render pipeline.
/// @param bind The frame bind group.
/// @param scissor Optional clip rect in device px.
/// @param verts Vertices per instance.
/// @param instances Instance count (1 for non-instanced custom draws).
/// @returns true when the frame submitted; false on a stale surface.
pub fn draw_n(
    surface: &Surface,
    pipeline: &GpuRenderPipeline,
    bind: &GpuBindGroup,
    scissor: Option<[u32; 4]>,
    verts: u32,
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

    let clear = GpuColorDict::new(0.0, 0.0, 0.0, 0.0);
    let attach = GpuRenderPassColorAttachment::new_with_gpu_texture_view(
        GpuLoadOp::Clear,
        GpuStoreOp::Store,
        &view,
    );
    attach.set_clear_value_gpu_color_dict(&clear);

    let attachments = [js_sys::JsNullable::wrap(attach)];
    let pass_desc = GpuRenderPassDescriptor::new(&attachments);

    let encoder = surface
        .device
        .create_command_encoder_with_descriptor(&GpuCommandEncoderDescriptor::new());

    match encoder.begin_render_pass(&pass_desc) {
        Ok(pass) => {
            pass.set_pipeline(pipeline);
            if let Some([x, y, w, h]) = scissor {
                pass.set_scissor_rect(x, y, w, h);
            }
            pass.set_bind_group(0, Some(bind));
            pass.draw_with_instance_count(verts, instances);
            pass.end();
        }
        Err(_) => return false,
    }

    surface.queue.submit(&[encoder.finish()]);

    true
}
