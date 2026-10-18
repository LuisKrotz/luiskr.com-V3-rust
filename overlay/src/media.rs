//! Media pipeline — image decode + video frames → wgpu textures.
//!
//! # Why two ingest paths (and not one)
//!
//! * **Still images** — `decode_image` runs the `image` crate
//!   (jpeg/png/webp) inside the wasm module. A traditional page lets the
//!   browser decode (`<img>` or `createImageBitmap`) on its own threads
//!   and then pays a structured-clone/texture-import dance to get pixels
//!   into WebGPU. Decoding in wasm skips all of that: bytes arrive as
//!   one `ArrayBuffer`, leave as one `queue.write_texture`. The trade is
//!   CPU speed — Rust jpeg/webp decoders are SIMD-less on wasm32 today
//!   (no simd128 paths compiled in), so decode is ~1.5–2× slower than
//!   the browser's native codec — acceptable because the DOM `<img>`
//!   fallback still loads in parallel (the texture arrives when ready).
//! * **Video** — decode stays in WebCodecs on the JS side. Hardware h264
//!   decode in the browser is ~10-50× cheaper than any pure-Rust decoder
//!   we could ship; abandoning it would be a regression, not a port.
//!   JS does `VideoFrame.copyTo` into a shared ArrayBuffer (one GPU→CPU
//!   copy — unavoidable: WebCodecs VideoFrames are opaque GPU objects),
//!   then `push_video_frame` uploads once CPU→GPU. Total cost per frame
//!   at 1080p: one 8MB memcpy + one 8MB texture write ≈ 2–3ms on modern
//!   cores — inside the 4ms frame budget at 240Hz.
//!
//! # Texture lifetime model
//!
//! Textures live in a per-widget pool (one `MediaSlot` per registered
//! media widget). The bind group pairs the widget's OWN uniform buffer
//! (rect + uv-fit — binding 0) with the texture view + shared sampler,
//! so it is built once at upload. Frame swaps are `write_texture`
//! calls into the existing texture — the cheapest GPU update primitive
//! wgpu exposes (a DMA copy into already-allocated VRAM). A bind group
//! or pipeline rebuild per frame would be the classic mistake here —
//! those allocate driver objects and are measured in tens of µs each.

use wgpu::util::DeviceExt;

/// A decoded/uploaded texture plus its bind group for the media shader.
/// Bind-group fields stay alive as long as the slot does — dropping a
/// slot drops the texture too (wgpu handles are refcounted).
pub struct MediaSlot {
    /// The texture itself — kept so `write_frame` can stream into it.
    pub tex: wgpu::Texture,
    /// Bind group pairing the texture's view with the widget uniform +
    /// sampler. The TextureView is NOT stored separately — the bind
    /// group holds it alive (wgpu refcounts) and nothing else reads it.
    pub bind: wgpu::BindGroup,
    /// Source pixel size — the slot rebuilds its texture only when this
    /// changes (a clip with a fixed resolution never rebuilds).
    pub size: [u32; 2],
}

/// Decodes jpeg/png/webp bytes to RGBA8. Pure-Rust — runs wherever the
/// wasm module runs (worker or main), keeping the browser main thread
/// free. Returns `(rgba, w, h)` or `None` on unparseable input — callers
/// treat `None` as "keep the DOM `<img>` on screen" (the dual-layer
/// rule: accelerated paths must never hide readable content).
///
/// # Math note — RGBA8 sizing
///
/// Output bytes = w × h × 4 (8 bits × 4 channels, no padding — `image`
/// emits packed rows). A 4K image is 33MB RGBA8: decode memory is real,
/// so media widgets should prefer the CDN's sized derivatives over
/// originals — the texture budget is what matters, not file size.
pub fn decode_image(bytes: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w, h) = img.dimensions();

    Some((img.into_raw(), w, h))
}

/// Builds a media slot from RGBA8 bytes — shared by the decode path and
/// the WebCodecs push path so both cost exactly one texture creation for
/// first frame / one `write_texture` thereafter.
///
/// `ubuf` is the owning widget's uniform buffer (binding 0: rect +
/// uv-fit — see media.wgsl's `Uniforms`), folded into the bind group at
/// creation so the draw needs no rebinding.
///
/// # Texture descriptor choices, each deliberate:
///
/// - `Rgba8Unorm` (not sRGB): the pipeline samples raw values and the
///   fragment shader outputs into the surface's sRGB target; using an
///   sRGB texture here would double-convert.
/// - `mip_level_count: 1`: UI media never minifies far enough to alias —
///   a 200px widget of a 4K source benefits, but the CDN derivative is
///   already ~widget-sized; spending 33% more VRAM on mips buys nothing.
///   Add mip generation (a cheap downsample pass) only if deep-zoom
///   surfaces appear.
/// - `usage: TEXTURE_BINDING | COPY_DST`: sampled in-shader + writable
///   by `write_texture` — the minimal set; no RENDER_ATTACHMENT, no
///   COPY_SRC (we never read pixels back — readbacks stall the queue).
/// - `w.max(1)`: a 0-dimension texture is a wgpu validation error;
///   clamp to 1 — the degenerate case (empty upload) then produces a
///   legal 1×1 texture rather than a panic.
pub fn make_slot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    ubuf: &wgpu::Buffer,
    rgba: &[u8],
    w: u32,
    h: u32,
) -> MediaSlot {
    let tex = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("media"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        rgba,
    );

    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Bind group is built ONCE here — bindings 0/1/2 mirror media.wgsl's
    // `@binding` numbering. Reordering these without editing the shader
    // produces a validation error at draw time (layouts are checked).
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("media"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });

    MediaSlot {
        tex,
        bind,
        size: [w, h],
    }
}

/// Streams a decoded frame into an existing slot — the video hot path.
/// The texture object stays; only texels swap.
///
/// # Why the size check exists
///
/// `write_texture` requires the copy extent ≤ texture size — a frame
/// arriving at a new resolution mid-stream (rare: adaptive video) would
/// fail validation. Rebuild the whole slot instead — bind group included,
/// which is safe because nobody holds the old bind mid-frame.
///
/// # `bytes_per_row: w * 4`
///
/// RGBA8 rows are `width × 4` bytes with NO alignment padding required
/// (wgpu asks rows be a multiple of 256B for COPIES TO/FROM BUFFERS, but
/// `write_texture` from a slice uses `TexelCopyBufferLayout` where
/// bytes_per_row just describes the source stride — any multiple of 4
/// works). Keep `Some(w * 4)`; passing None treats the source as
/// tightly packed which happens to be identical here, but the explicit
/// stride protects the call if a padded-row source ever arrives.
pub fn write_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    ubuf: &wgpu::Buffer,
    slot: &mut MediaSlot,
    rgba: &[u8],
    w: u32,
    h: u32,
) {
    if slot.size != [w, h] {
        *slot = make_slot(device, queue, layout, sampler, ubuf, rgba, w, h);
        return;
    }

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &slot.tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
}
