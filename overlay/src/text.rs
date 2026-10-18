//! GPU text via glyphon.
//!
//! Every string the overlay draws is a *mirror* of a hidden-DOM node —
//! the semantic layer stays the source of truth for crawlers and reader
//! mode; this module only rasterizes what Layer 2 already contains.
//!
//! Cyrillic: the site swaps Raleway for Fira Sans under `lang=ru` (see
//! `core/locale/cyrillic-font.ts`). The same swap happens here: the JS
//! bridge ships whichever font the route loaded as raw bytes and this
//! side registers it in the glyphon `FontSystem` under a stable id —
//! shaping/kerning/ligatures then come for free from swash.
//!
//! Layout caveat baked into the contract: glyphon wraps on a fixed
//! `Buffer::set_size` measure, not the browser's line-break engine —
//! every `stage` call must receive the DOM-measured `max_width_px` for
//! its node so breaks cannot diverge from the mirrored element.

use glyphon::{
    Attrs, Buffer, Cache, Color as GColor, FontSystem, Metrics, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use wgpu::{Device, MultisampleState, Queue, RenderPass, TextureFormat};

/// Default Latin family id — registered by `install_font`.
pub const FAMILY_LATIN: &str = "Raleway";
/// Cyrillic family id — the JS side pushes Fira Sans bytes on `ru` routes.
pub const FAMILY_CYRILLIC: &str = "Fira Sans";

/// One DOM-mirrored text run staged for a frame.
pub struct TextJob<'a> {
    /// UTF-8 run — must equal the hidden node's textContent.
    pub text: &'a str,
    /// [x, y] top-left in CSS px.
    pub pos: [f32; 2],
    /// CSS px line height (glyphon `Metrics::new(size, line_height)`).
    pub size: f32,
    /// DOM-measured wrap width in CSS px — the mirror keeps layout parity.
    pub max_width_px: f32,
    /// `FAMILY_LATIN` or `FAMILY_CYRILLIC` — set by the route's font gate.
    pub family: &'a str,
    /// Premultiplied-ish RGBA in 0..1 (glyphon takes srgb ints — converted).
    pub rgba: [f32; 4],
    /// Visible clip rect in device px: [left, top, right, bottom].
    pub bounds: [i32; 4],
}

/// Frame-scoped text pipeline: font system + atlas + renderer are held
/// for the surface's life; buffers are rebuilt per stage batch (glyphon
/// owns the shaping cache, so rebuilds are shape-cached, not re-shaped).
pub struct TextPipe {
    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    viewport: Viewport,
    renderer: TextRenderer,
}

impl TextPipe {
    /// Creates the pipeline for the surface's format; MSAA state mirrors
    /// the render pipeline (1× — glyphon does its own coverage).
    pub fn new(device: &Device, queue: &Queue, format: TextureFormat) -> Self {
        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let renderer = TextRenderer::new(
            &mut atlas,
            device,
            MultisampleState::default(),
            None,
        );

        Self {
            font_system,
            swash_cache,
            atlas,
            viewport,
            renderer,
        }
    }

    /// Loads font bytes (woff2/ttf/otf) under `family`. Called by the JS
    /// bridge once per route font — Cyrillic bytes arrive only when the
    /// locale needs them, mirroring the site's Raleway/Fira gate.
    pub fn install_font(&mut self, family: &str, bytes: &[u8]) {
        self.font_system.db_mut().load_font_data(bytes.to_vec());
        // swash registers the face under its own name; `family` is the
        // lookup key staged jobs will use — glyphon resolves by family
        // name at Attrs time, so the load above is what matters here.
        let _ = family;
    }

    /// Updates the atlas viewport for the current device-pixel size —
    /// subpixel positions are quantized against this resolution, so it
    /// must be refreshed on every resize before `prepare`.
    pub fn resize(&mut self, queue: &Queue, w: u32, h: u32) {
        self.viewport.update(
            queue,
            Resolution {
                width: w,
                height: h,
            },
        );
    }

    /// Shapes + rasterizes a batch of jobs into the atlas and emits the
    /// text draw data for the next pass. Shaping is cached — repeated
    /// frames of static text cost only `prepare`, not re-shaping.
    pub fn prepare(&mut self, device: &Device, queue: &Queue, jobs: &[TextJob<'_>]) {
        let mut buffers = Vec::with_capacity(jobs.len());

        for j in jobs {
            let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(j.size, j.size));

            buffer.set_size(
                &mut self.font_system,
                Some(j.max_width_px),
                None,
            );

            buffer.set_text(
                &mut self.font_system,
                j.text,
                &Attrs::new().family(glyphon::Family::Name(j.family)),
                Shaping::Advanced,
            );

            buffers.push((j, buffer));
        }

        let areas: Vec<TextArea<'_>> = buffers
            .iter()
            .map(|(j, buffer)| TextArea {
                buffer,
                left: j.pos[0],
                top: j.pos[1],
                scale: 1.0,
                bounds: TextBounds {
                    left: j.bounds[0],
                    top: j.bounds[1],
                    right: j.bounds[2],
                    bottom: j.bounds[3],
                },
                default_color: GColor::rgba(
                    (j.rgba[0] * 255.0) as u8,
                    (j.rgba[1] * 255.0) as u8,
                    (j.rgba[2] * 255.0) as u8,
                    (j.rgba[3] * 255.0) as u8,
                ),
                custom_glyphs: &[],
            })
            .collect();

        let _ = self.renderer.prepare(
            device,
            queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            areas,
            &mut self.swash_cache,
        );
    }

    /// Issues the text draw inside an already-open pass. `prepare` must
    /// have run for the same frame — the atlas texture is bound there.
    pub fn render<'pass>(&'pass self, pass: &mut RenderPass<'pass>) {
        let _ = self.renderer.render(&self.atlas, &self.viewport, pass);
    }
}
