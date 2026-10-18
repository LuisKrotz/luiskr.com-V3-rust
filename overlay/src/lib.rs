//! overlay — the WebGPU paint layer of the dual-layer architecture.
//!
//! # The architecture in one picture
//!
//! ```text
//!   DOM (Layer 2)                    WASM + WebGPU (Layer 1)
//!   ┌──────────────────┐  rAF tick   ┌────────────────────────────┐
//!   │ nav, h1-h6, JSON │────────────▶│ Overlay::frame(dt)         │
//!   │ -LD, links — the │◄────────────│  drain input ring → hit test│
//!   │ SOURCE OF TRUTH  │ hover_group │  tick burn sims → quads    │
//!   │ for bots + ATs   │             │  write_buffer (one memcpy) │
//!   └──────────────────┘             │  encode: quads + widgets +  │
//!   visually-hidden, readable        │  text → present            │
//!   to everyone                      └────────────────────────────┘
//! ```
//!
//! The DOM keeps ALL semantics (links, h1-h6, JSON-LD, the screen-reader
//! tree) — this canvas only paints. If WebGPU, the wasm artifact, or
//! `init` fails, the bridge mounts nothing and the DOM renders normally
//! (zero-regression rule: the accelerated path can never hide content).
//!
//! # Why the frame loop lives in JS but the state lives here
//!
//! Governance requires ONE rAF clock owned by the bridge (so every
//! animation surface shares a tick and the budget is measurable). But
//! everything heavy — font shaping, hit-testing, particle sim, uniform
//! math — runs in wasm. Per-frame JS does exactly two things: write
//! input records into the shared ring (pointer writes, no calls) and
//! call `frame(dt)` once. Compare a traditional structure where every
//! widget owns a canvas + a rAF + a GL context: N contexts, N clocks,
//! N clears, unbounded overdraw. Here: one surface, one encoder, and N
//! tiny scissored draws sharing a command buffer.
//!
//! # dt-space, not frame-space
//!
//! Every animation integrates in SECONDS (`dt`), never frame counts —
//! the same reason the project's JS animations are dt-driven. At 60Hz
//! dt≈0.0167, at 240Hz dt≈0.0042; the visual outcome is identical
//! because position += velocity·dt, not position += perFrameStep. Any
//! number marked "/s" (px/s, rad/s, flips/s) is dt-safe; constants that
//! look like per-frame steps are not — don't add any.
//!
//! Widgets: each animation surface that used to own a small WebGL canvas
//! registers a rect region + kind once; `frame` draws every registered
//! widget scissored to its rect with its own WGSL pipeline — one
//! surface, one encoder, N tiny draws. See `widgets/mod.rs` for the
//! domain list and `../ARCHITECTURE.md` for the full map.

mod input;
mod layout;
mod media;
mod text;
mod widgets;

use input::{drain, RING_F32S, RING_META_I32S};
use layout::{Quad, Scene};
use media::MediaSlot;
use std::cell::RefCell;
use std::collections::HashMap;
use text::{TextJob, TextPipe};
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use widgets::burn::BurnSim;

/// `UnsafeCell` + manual `Sync` — the modern `static mut`. Safe here
/// under the wasm32 single-threaded contract: exactly one writer (the
/// JS event pump writes through the exported pointers) and one reader
/// (`frame`'s drain), and wasm-bindgen modules are per-thread — a
/// second JS context can't race on these addresses.
struct SyncCell<T>(std::cell::UnsafeCell<T>);

// SAFETY: the wasm module runs on exactly one thread (no shared
// memory / no threads proposal), so no two accessors can ever execute
// concurrently — the Sync impl exists only to satisfy the compiler.
unsafe impl<T> Sync for SyncCell<T> {}

/// Input ring storage — written by JS through `input_buffer_ptr`, read by
/// `drain`. Zero-copy: JS event code writes f32s straight into wasm
/// linear memory at these addresses, no call per event.
static INPUT_RING: SyncCell<[f32; RING_F32S]> = SyncCell(std::cell::UnsafeCell::new([0.0; RING_F32S]));
static INPUT_META: SyncCell<[i32; RING_META_I32S]> = SyncCell(std::cell::UnsafeCell::new([0; RING_META_I32S]));

/// Registered text mirrors — the DOM layer pushes a node here once per
/// content change (route swap, locale change), not per frame. `TextJob`s
/// are rebuilt from these at `frame` time.
struct TextNode {
    text: String,
    pos: [f32; 2],
    size: f32,
    max_width_px: f32,
    family: &'static str,
    rgba: [f32; 4],
    bounds: [i32; 4],
}

/// Per-widget capacity of the private uniform buffer — 256 B covers the
/// largest block (16 f32s) with headroom and stays above the common
/// min-uniform-offset alignment, so the same buffer can later serve
/// dynamic-offset binding without restructure.
const WIDGET_UBUF_SIZE: u64 = 256;

/// A registered widget region — one scissored draw per frame.
struct Widget {
    /// `widgets::kinds` discriminant.
    kind: u32,
    /// [x, y, w, h] canvas CSS px — the scissor + uv normalization rect.
    rect: [f32; 4],
    /// Per-widget uniform buffer (kind's Uniforms block, ≤256 B).
    ubuf: wgpu::Buffer,
    /// Bind group — None until every required resource exists (flag
    /// waits on its two textures; skeleton on its sheet buffer).
    bind: Option<wgpu::BindGroup>,
    /// Skeleton sheet storage — per-rect vec4 fields.
    sheet: Option<wgpu::Buffer>,
    /// Flag texture pair — (tex1, tex2) stages before `bind` builds.
    flag_tex: Option<(wgpu::TextureView, wgpu::TextureView)>,
}

/// wgpu surface state — everything the frame needs, none of it re-created.
struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    quad_buf: wgpu::Buffer,
    /// Capacity in quads — grown geometrically so steady frames never
    /// re-create the instance buffer.
    quad_cap: usize,
    text: TextPipe,
    /// Shared linear-filter sampler for all textured widgets + media.
    sampler: wgpu::Sampler,
    /// kind → pipeline for the uniform-only SDF family.
    sdf_layout: wgpu::BindGroupLayout,
    /// kind → pipeline cache — compiled lazily on first registration.
    pipelines: HashMap<u32, wgpu::RenderPipeline>,
    /// flag layout — uniform + 2 textures + sampler.
    flag_layout: wgpu::BindGroupLayout,
    flag_pipeline: wgpu::RenderPipeline,
    /// skeleton layout — uniform + storage sheet.
    skel_layout: wgpu::BindGroupLayout,
    /// media layout + pipeline — textured quad.
    media_layout: wgpu::BindGroupLayout,
    media_pipeline: wgpu::RenderPipeline,
}

/// Public wasm surface. `RefCell` interior-mutates around wasm-bindgen's
/// `&self`/`&mut self` exports without JS-visible ceremony.
#[wasm_bindgen]
pub struct Overlay {
    gpu: RefCell<Option<Gpu>>,
    scene: RefCell<Scene>,
    texts: RefCell<Vec<TextNode>>,
    /// Group under the pointer, from the last drain — the JS router reads
    /// it to fire navigation intents (`hit_group` == a DOM link's id).
    hover: RefCell<u32>,
    /// Widget id → instance. Ids are JS-bridge-stable handles.
    widgets: RefCell<HashMap<u32, Widget>>,
    /// Next widget id — monotonic, wraps-safe for session lifetimes.
    next_widget: RefCell<u32>,
    /// Media widget id → texture slot.
    media_slots: RefCell<HashMap<u32, MediaSlot>>,
    /// Burn sim per widget id (kind BURN — CPU sim, quad-emitting).
    burns: RefCell<HashMap<u32, BurnSim>>,
    /// Canvas size in CSS px — scissor math divides surface px by this
    /// (surface is device px; widget rects arrive in CSS px).
    css_size: RefCell<[f32; 2]>,
    /// Reusable draw-order scratch — cleared + refilled per frame so
    /// the sort allocation is once-per-session, not once-per-frame
    /// (zero-allocation frame rule).
    draw_order: RefCell<Vec<u32>>,
}

/// Instanced-quad WGSL — one draw call paints the frame. Vertex positions
/// are synthesized from the instance's pos/size (fullscreen UV-space quad
/// scaled into place), so no vertex buffer exists at all.
const QUAD_WGSL: &str = include_str!("quad.wgsl");

/// Textured-quad WGSL for media widgets (decoded images / video frames).
const MEDIA_WGSL: &str = include_str!("widgets/media.wgsl");

/// Builds the shared SDF-family bind-group layout — one uniform binding.
fn sdf_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sdf"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Flag layout — uniform + two flag textures + shared sampler.
fn flag_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let tex = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("flag"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            tex(1),
            tex(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Skeleton layout — uniform + read-only storage sheet.
fn skel_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("skeleton"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Media layout — uniform rect + texture + sampler.
fn media_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("media"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Compiles a fullscreen-triangle widget pipeline — all SDF widgets and
/// the flag share this shape: no vertex buffers, scissor does the clip.
fn widget_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    wgsl: &str,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });

    let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipe_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

#[wasm_bindgen]
impl Overlay {
    /// Constructs an un-initialized overlay — `init` must follow before
    /// `frame`. Split because wgpu init is async on wasm32.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Overlay {
        Overlay {
            gpu: RefCell::new(None),
            scene: RefCell::new(Scene::default()),
            texts: RefCell::new(Vec::new()),
            hover: RefCell::new(0),
            widgets: RefCell::new(HashMap::new()),
            next_widget: RefCell::new(1),
            media_slots: RefCell::new(HashMap::new()),
            burns: RefCell::new(HashMap::new()),
            css_size: RefCell::new([1.0, 1.0]),
            draw_order: RefCell::new(Vec::new()),
        }
    }

    /// Tells the overlay the canvas's CSS size — the bridge calls this
    /// alongside `resize(device px)` so scissor math can convert widget
    /// rects (CSS px) into surface pixels.
    pub fn set_css_size(&self, w: f32, h: f32) {
        *self.css_size.borrow_mut() = [w.max(1.0), h.max(1.0)];
    }

    /// Pointer to the f32 input ring — JS writes records directly into
    /// wasm linear memory here. No copy, no call per event.
    pub fn input_buffer_ptr(&self) -> *mut f32 {
        INPUT_RING.0.get().cast::<f32>()
    }

    /// Pointer to the [head, tail] i32 meta pair — JS bumps head after
    /// each record write, Rust drains from tail during `frame`.
    pub fn input_meta_ptr(&self) -> *mut i32 {
        INPUT_META.0.get().cast::<i32>()
    }

    /// Request adapter/device and configure the canvas surface. Called
    /// once from the bridge; failure propagates so the bridge falls back
    /// to DOM rendering.
    pub async fn init(&self, canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| JsValue::from_str(&format!("surface: {e}")))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("adapter: {e}")))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("overlay"),
                ..Default::default()
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("device: {e}")))?;

        let cap = surface.get_capabilities(&adapter);
        // sRGB surface preferred — the DOM's colors are sRGB and a
        // non-sRGB surface would shift every rendered color a gamma step
        // off from the DOM underneath. `unwrap_or` keeps exotic displays
        // (linear-only targets) alive rather than panicking.
        let format = cap
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(cap.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: canvas.width().max(1),
            height: canvas.height().max(1),
            // Mailbox present mode — the key latency choice. The three
            // modes: Fifo (vsync-locked queue — input lag grows at high
            // Hz), FifoRelaxed (tears), Mailbox (present the NEWEST
            // completed frame, drop the rest). At 120/240Hz a queued
            // frame is a stale frame — Mailbox is the only mode that
            // keeps pointer→pixel latency at one frame.
            present_mode: wgpu::PresentMode::Mailbox,
            // Max 2 frames in flight — the browser compositor already
            // owns one; deeper queues add lag without throughput benefit
            // (we're latency-bound, not throughput-bound).
            desired_maximum_frame_latency: 2,
            alpha_mode: cap.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay-quads"),
            source: wgpu::ShaderSource::Wgsl(QUAD_WGSL.into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay-quads"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Quad>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2, // pos
                        1 => Float32x2, // size
                        2 => Float32x2, // uv
                        3 => Float32x2, // uv_size
                        4 => Float32x4, // tint
                        5 => Float32,   // z
                        6 => Uint32,    // group
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let quad_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay-quads"),
            size: (std::mem::size_of::<Quad>() * 1024) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let text = TextPipe::new(&device, &queue, format);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("overlay"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sdf_layout = sdf_layout(&device);
        let flag_layout = flag_layout(&device);
        let skel_layout = skel_layout(&device);
        let media_layout = media_layout(&device);

        let flag_pipeline = widget_pipeline(
            &device,
            &flag_layout,
            widgets::flag::WGSL,
            format,
            "widget-flag",
        );
        let media_pipeline = widget_pipeline(
            &device,
            &media_layout,
            MEDIA_WGSL,
            format,
            "widget-media",
        );

        *self.gpu.borrow_mut() = Some(Gpu {
            surface,
            device,
            queue,
            config,
            pipeline,
            quad_buf,
            quad_cap: 1024,
            text,
            sampler,
            sdf_layout,
            pipelines: HashMap::new(),
            flag_layout,
            flag_pipeline,
            skel_layout,
            media_layout,
            media_pipeline,
        });

        Ok(())
    }

    /// Reconfigures the surface + text viewport after a canvas resize.
    /// `scale` is devicePixelRatio — text jobs stay in CSS px, glyphon
    /// rasterizes at device px so glyphs stay crisp at 120Hz+.
    pub fn resize(&self, w: u32, h: u32) {
        let mut gpu = self.gpu.borrow_mut();
        let Some(g) = gpu.as_mut() else { return };

        g.config.width = w.max(1);
        g.config.height = h.max(1);
        g.surface.configure(&g.device, &g.config);
        g.text.resize(&g.queue, g.config.width, g.config.height);
    }

    /// Registers a widget region on the overlay. `kind` is a
    /// `widgets::kinds` discriminant; the rect is canvas CSS px. Returns
    /// a stable id for state pushes + teardown, 0 when the kind is
    /// unknown or the GPU isn't live (bridge keeps its DOM fallback).
    pub fn register_widget(&self, kind: u32, x: f32, y: f32, w: f32, h: f32) -> u32 {
        let mut gpu = self.gpu.borrow_mut();
        let Some(g) = gpu.as_mut() else { return 0 };

        // Burn is a quad-emitting sim — no pipeline needed.
        if kind == widgets::kinds::BURN {
            let id = *self.next_widget.borrow();
            *self.next_widget.borrow_mut() = id.wrapping_add(1).max(1);
            self.burns
                .borrow_mut()
                .insert(id, BurnSim::new(1.0, [x, y], id));

            self.widgets.borrow_mut().insert(
                id,
                Widget {
                    kind,
                    rect: [x, y, w, h],
                    ubuf: g.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("widget-uniform"),
                        size: WIDGET_UBUF_SIZE,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    bind: None,
                    sheet: None,
                    flag_tex: None,
                },
            );

            return id;
        }

        let Some(spec) = widgets::spec_of(kind) else {
            return 0;
        };

        let ubuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("widget-uniform"),
            size: WIDGET_UBUF_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut sheet = None;
        let mut bind = None;

        match kind {
            widgets::kinds::FLAG => {
                // Bind waits on upload_flag_textures — a flag with no
                // image binds nothing and draws nothing (never errors).
            }
            widgets::kinds::SKELETON => {
                let sheet_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("skel-sheet"),
                    size: (widgets::skeleton::STORAGE_VEC4S * 16) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                g.pipelines
                    .entry(kind)
                    .or_insert_with(|| {
                        widget_pipeline(
                            &g.device,
                            &g.skel_layout,
                            spec.wgsl,
                            g.config.format,
                            "widget-skeleton",
                        )
                    });

                bind = Some(g.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("widget-skel"),
                    layout: &g.skel_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: ubuf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: sheet_buf.as_entire_binding(),
                        },
                    ],
                }));

                sheet = Some(sheet_buf);
            }
            _ => {
                g.pipelines
                    .entry(kind)
                    .or_insert_with(|| {
                        widget_pipeline(
                            &g.device,
                            &g.sdf_layout,
                            spec.wgsl,
                            g.config.format,
                            "widget-sdf",
                        )
                    });

                bind = Some(g.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("widget-sdf"),
                    layout: &g.sdf_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubuf.as_entire_binding(),
                    }],
                }));
            }
        }

        let id = *self.next_widget.borrow();
        *self.next_widget.borrow_mut() = id.wrapping_add(1).max(1);

        self.widgets.borrow_mut().insert(
            id,
            Widget {
                kind,
                rect: [x, y, w, h],
                ubuf,
                bind,
                sheet,
                flag_tex: None,
            },
        );

        id
    }

    /// Writes the widget's uniform block — raw f32s in the layout the
    /// widget's `*.rs` documents. Called per state change (per frame for
    /// time/progress, once per event for ripple/click stamps).
    pub fn set_widget_uniform(&self, id: u32, values: &[f32]) {
        let gpu = self.gpu.borrow();
        let widgets = self.widgets.borrow();

        let (Some(g), Some(w)) = (gpu.as_ref(), widgets.get(&id)) else {
            return;
        };

        // Clamp to the kind's declared block size — an oversized JS
        // write is truncated to the contract rather than silently
        // overflowing the uniform layout (media/kind-0 uses the buffer
        // capacity as its bound).
        let max = widgets::spec_of(w.kind)
            .map(|s| s.uniform_f32s)
            .unwrap_or(WIDGET_UBUF_SIZE as usize / 4);
        let n = values.len().min(max);
        g.queue.write_buffer(&w.ubuf, 0, bytemuck::cast_slice(&values[..n]));
    }

    /// Moves/resizes a widget region — the bridge calls this when the
    /// DOM rect it mirrors moves (layout shift, resize, scroll of the
    /// containing surface). The widget's own `res` uniform updates too
    /// so shaders keep their original px-space math.
    pub fn set_widget_rect(&self, id: u32, x: f32, y: f32, w: f32, h: f32) {
        let mut widgets = self.widgets.borrow_mut();

        if let Some(wgt) = widgets.get_mut(&id) {
            wgt.rect = [x, y, w, h];
        }

        // Burn sims anchor at the widget rect's origin — a tile that
        // moves drags its flame front with it.
        if let Some(sim) = self.burns.borrow_mut().get_mut(&id) {
            sim.set_origin([x, y]);
        }
    }

    /// Removes a widget — GPU resources drop with the handle.
    pub fn unregister_widget(&self, id: u32) {
        self.widgets.borrow_mut().remove(&id);
        self.media_slots.borrow_mut().remove(&id);
        self.burns.borrow_mut().remove(&id);
    }

    /// Uploads both flag textures (RGBA8) and finishes the flag bind
    /// group. `tex2` may repeat `tex1` for non-split flags.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_flag_textures(
        &self,
        id: u32,
        rgba_a: &[u8],
        w_a: u32,
        h_a: u32,
        rgba_b: &[u8],
        w_b: u32,
        h_b: u32,
    ) {
        let gpu = self.gpu.borrow();
        let mut widgets = self.widgets.borrow_mut();

        let (Some(g), Some(wgt)) = (gpu.as_ref(), widgets.get_mut(&id)) else {
            return;
        };

        if wgt.kind != widgets::kinds::FLAG {
            return;
        }

        let make_view = |rgba: &[u8], w: u32, h: u32| {
            let tex = g.device.create_texture_with_data(
                &g.queue,
                &wgpu::TextureDescriptor {
                    label: Some("flag"),
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

            tex.create_view(&wgpu::TextureViewDescriptor::default())
        };

        let v1 = make_view(rgba_a, w_a, h_a);
        let v2 = make_view(rgba_b, w_b, h_b);

        wgt.bind = Some(g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("widget-flag"),
            layout: &g.flag_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgt.ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&v1),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&v2),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&g.sampler),
                },
            ],
        }));

        wgt.flag_tex = Some((v1, v2));
    }

    /// Writes the skeleton sheet — the per-rect vec4 fields the shader
    /// loops (rects/meta/sbase/sink, `skeleton::RECT_FIELDS` per rect).
    pub fn write_skel_sheet(&self, id: u32, values: &[f32]) {
        let gpu = self.gpu.borrow();
        let widgets = self.widgets.borrow();

        let (Some(g), Some(w)) = (gpu.as_ref(), widgets.get(&id)) else {
            return;
        };

        if let Some(sheet) = &w.sheet {
            g.queue.write_buffer(sheet, 0, bytemuck::cast_slice(values));
        }
    }

    /// Decodes an image (jpeg/png/webp) in Rust and binds it to a media
    /// widget — decode runs in wasm, off the main thread, no
    /// createImageBitmap round trip. Returns false on decode failure so
    /// the bridge keeps the DOM <img> (manifest-gated fallback).
    pub fn upload_media_image(&self, id: u32, bytes: &[u8]) -> bool {
        let gpu = self.gpu.borrow();
        let Some(g) = gpu.as_ref() else { return false };
        let widgets = self.widgets.borrow();
        let Some(wgt) = widgets.get(&id) else { return false };

        let Some((rgba, w, h)) = media::decode_image(bytes) else {
            return false;
        };

        let slot = media::make_slot(
            &g.device,
            &g.queue,
            &g.media_layout,
            &g.sampler,
            &wgt.ubuf,
            &rgba,
            w,
            h,
        );

        self.media_slots.borrow_mut().insert(id, slot);

        true
    }

    /// Pushes a decoded video frame (WebCodecs → RGBA) into a media
    /// widget's texture — the video hot path: one write_texture, no
    /// bind-group rebuild, size changes recreate the slot.
    pub fn push_video_frame(&self, id: u32, rgba: &[u8], w: u32, h: u32) {
        let gpu = self.gpu.borrow();
        let Some(g) = gpu.as_ref() else { return };
        let widgets = self.widgets.borrow();
        let Some(wgt) = widgets.get(&id) else { return };
        let mut slots = self.media_slots.borrow_mut();

        if let Some(slot) = slots.get_mut(&id) {
            media::write_frame(
                &g.device,
                &g.queue,
                &g.media_layout,
                &g.sampler,
                &wgt.ubuf,
                slot,
                rgba,
                w,
                h,
            );
        } else {
            let slot = media::make_slot(
                &g.device,
                &g.queue,
                &g.media_layout,
                &g.sampler,
                &wgt.ubuf,
                rgba,
                w,
                h,
            );

            slots.insert(id, slot);
        }
    }

    /// Registers a media widget rect — media quads draw through the
    /// textured pipeline; the id is shared with the slot table above.
    /// Same signature family as `register_widget` but no kind — media
    /// widgets are always textured quads.
    pub fn register_media(&self, x: f32, y: f32, w: f32, h: f32) -> u32 {
        let mut gpu = self.gpu.borrow_mut();
        let Some(g) = gpu.as_mut() else { return 0 };

        let ubuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("media-uniform"),
            size: WIDGET_UBUF_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let id = *self.next_widget.borrow();
        *self.next_widget.borrow_mut() = id.wrapping_add(1).max(1);

        self.widgets.borrow_mut().insert(
            id,
            Widget {
                kind: 0, // kind 0 = media quad (no spec table entry)
                rect: [x, y, w, h],
                ubuf,
                bind: None,
                sheet: None,
                flag_tex: None,
            },
        );

        id
    }

    /// Burn sim data: letter table (f32×6 per letter).
    pub fn burn_set_letters(&self, id: u32, flat: &[f32]) {
        if let Some(sim) = self.burns.borrow_mut().get_mut(&id) {
            sim.set_letters(flat);
        }
    }

    /// Burn sim data: per-column ink occupancy bytes.
    pub fn burn_set_ink(&self, id: u32, ink: &[u8]) {
        if let Some(sim) = self.burns.borrow_mut().get_mut(&id) {
            sim.set_ink(ink);
        }
    }

    /// Burn sim data: four ink tints [core, mid, edge, scorch] × rgba.
    pub fn burn_set_colors(&self, id: u32, flat: &[f32]) {
        if let Some(sim) = self.burns.borrow_mut().get_mut(&id) {
            sim.set_colors(flat);
        }
    }

    /// Burn sim stage — pushed per frame before `frame`: global progress
    /// 0..1 + jagged-edge amplitude in device px.
    pub fn burn_set_stage(&self, id: u32, stage: f32, jag_amp: f32) {
        if let Some(sim) = self.burns.borrow_mut().get_mut(&id) {
            sim.set_stage(stage, jag_amp);
        }
    }

    /// Burn sim direction: +1 burn, −1 rebuild (un-hover replay).
    pub fn burn_set_dir(&self, id: u32, dir: f32) {
        // Direction flips restart the session's particle stream; the
        // letters/ink/colors persist.
        if let Some(sim) = self.burns.borrow_mut().get_mut(&id) {
            sim.set_dir(dir);
        }
    }

    /// Registers/replaces a mirrored text node. Called on DOM content
    /// change — not per frame. `id` is the bridge's stable slot for the
    /// DOM node; re-calling with an existing id updates in place.
    #[allow(clippy::too_many_arguments)]
    pub fn set_text(
        &self,
        id: u32,
        text: String,
        x: f32,
        y: f32,
        size: f32,
        max_width_px: f32,
        family: &str,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
        bl: i32,
        bt: i32,
        br: i32,
        bb: i32,
    ) {
        let node = TextNode {
            text,
            pos: [x, y],
            size,
            max_width_px,
            family: if family == "Fira Sans" {
                text::FAMILY_CYRILLIC
            } else {
                text::FAMILY_LATIN
            },
            rgba: [r, g, b, a],
            bounds: [bl, bt, br, bb],
        };

        let mut texts = self.texts.borrow_mut();
        let id = id as usize;

        if id >= texts.len() {
            texts.resize_with(id + 1, || TextNode {
                text: String::new(),
                pos: [0.0; 2],
                size: 0.0,
                max_width_px: 0.0,
                family: text::FAMILY_LATIN,
                rgba: [0.0; 4],
                bounds: [0; 4],
            });
        }

        texts[id] = node;
    }

    /// Loads a route font (Raleway/Fira bytes) into the glyphon font db.
    pub fn install_font(&self, family: &str, bytes: &[u8]) {
        let mut gpu = self.gpu.borrow_mut();

        if let Some(g) = gpu.as_mut() {
            g.text.install_font(family, bytes);
        }
    }

    /// The group under the pointer from the latest drain — the bridge
    /// reads it after `frame` to route click intents to the DOM router.
    pub fn hover_group(&self) -> u32 {
        *self.hover.borrow()
    }

    /// One frame — the only per-tick entry point, and the whole reason
    /// the architecture works. Five stages, each chosen to cost the
    /// minimum the hardware allows:
    ///
    /// 1. **Drain input** — replay every queued event once. Hover is
    ///    recomputed only on move/down (up and wheel don't change what's
    ///    under the pointer — skipping their hit_tests is a deliberate
    ///    cost cut).
    /// 2. **Tick sims** — burn particles advance in dt-space and emit
    ///    quads into the shared buffer (one instanced draw covers ALL
    ///    particles of ALL widgets).
    /// 3. **Upload** — ONE `write_buffer` memcpy per frame; the instance
    ///    buffer is grown geometrically (`next_power_of_two`) so steady
    ///    frames never recreate it — allocation-free like the Vecs.
    /// 4. **Encode** — clear once, then: quad pass → per-widget scissored
    ///    draws (paint order = rect.y sort, mirroring DOM order) → text.
    ///    One encoder, one render pass — state changes between widgets
    ///    are pipeline+bind swaps, cheap driver commands.
    /// 5. **Present** — mailbox mode drops stale frames rather than
    ///    queueing latency.
    ///
    /// `dt` is in SECONDS — see the module doc's dt-space note.
    pub fn frame(&self, dt: f32) {
        let mut gpu = self.gpu.borrow_mut();
        let Some(g) = gpu.as_mut() else { return };

        unsafe {
            // SAFETY: single-threaded wasm — the JS writer can only
            // interleave BETWEEN exported calls, never during `frame`.
            let meta: &mut [i32; RING_META_I32S] = &mut *INPUT_META.0.get();
            let ring: &[f32; RING_F32S] = &*INPUT_RING.0.get();
            let scene = self.scene.borrow();
            let mut hover = self.hover.borrow_mut();

            // Only move/down re-hit-test: pointer-up leaves the same
            // widget under the cursor, and wheel doesn't move the cursor
            // at all — both would just recompute the answer we have.
            drain(meta, ring, |kind, x, y, _payload| {
                if kind == input::kinds::POINTER_MOVE || kind == input::kinds::POINTER_DOWN {
                    *hover = scene.hit_test(x, y).unwrap_or(0);
                }
            });
        }

        let mut scene = self.scene.borrow_mut();
        scene.quads.clear();
        scene.hits.clear();

        // Burn sims emit their particles into the shared quad pass —
        // every particle of every burn widget lands in ONE buffer and
        // costs ONE instanced draw total.
        for sim in self.burns.borrow_mut().values_mut() {
            sim.tick(dt, &mut scene.quads);
        }

        let quad_bytes = bytemuck::cast_slice::<Quad, u8>(&scene.quads);

        // Geometric growth: the buffer only ever grows, by powers of two,
        // so after warmup no frame allocates GPU memory. Doubling (not
        // +N) means the growth sequence reaches any working size in
        // O(log) rebuilds — amortized-free like Vec::push.
        if scene.quads.len() > g.quad_cap {
            g.quad_cap = scene.quads.len().next_power_of_two();
            g.quad_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay-quads"),
                size: (g.quad_cap * std::mem::size_of::<Quad>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if !quad_bytes.is_empty() {
            g.queue.write_buffer(&g.quad_buf, 0, quad_bytes);
        }

        // Text jobs are rebuilt each frame — glyphon's shaping cache makes
        // this cheap for static strings.
        let texts = self.texts.borrow();
        let jobs: Vec<TextJob<'_>> = texts
            .iter()
            .filter(|t| !t.text.is_empty())
            .map(|t| TextJob {
                text: &t.text,
                pos: t.pos,
                size: t.size,
                max_width_px: t.max_width_px,
                family: t.family,
                rgba: t.rgba,
                bounds: t.bounds,
            })
            .collect();

        g.text.prepare(&g.device, &g.queue, &jobs);

        let Ok(frame_tex) = g.surface.get_current_texture() else {
            // Lost/outdated surface — reconfigure and skip the frame.
            g.surface.configure(&g.device, &g.config);
            return;
        };

        let view = frame_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("overlay"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if !scene.quads.is_empty() {
                pass.set_pipeline(&g.pipeline);
                pass.set_vertex_buffer(0, g.quad_buf.slice(..));
                pass.draw(0..6, 0..scene.quads.len() as u32);
            }

            // ── Per-widget scissored draws ───────────────────────────
            // Scissor rects are in SURFACE pixels (device px); widget
            // rects arrive in CSS px. `sx/sy` are the CSS→device scale —
            // equal to devicePixelRatio on a full-bleed canvas, but kept
            // as a ratio so a non-fullscreen overlay also works.
            // `.max(1.0)` guards a zero/unset css_size → scale 0 → all
            // widgets clipped to nothing (a hide-everything bug instead
            // of a divide-by-zero). The `.max(0.0)` keeps rects partially
            // offscreen-legal; `ww.min(width - x)` clamps at the right
            // edge so the scissor never leaves the surface (wgpu
            // validation would otherwise reject the draw).
            let widgets = self.widgets.borrow();
            let media_slots = self.media_slots.borrow();
            let css = *self.css_size.borrow();
            let sx = g.config.width as f32 / css[0].max(1.0);
            let sy = g.config.height as f32 / css[1].max(1.0);

            // Paint order = top-to-bottom rect.y — mirrors DOM paint
            // order for stacked widgets (the earlier-registered widget
            // of two overlapping ones paints under, matching how the
            // browser would stack their DOM equivalents). The scratch
            // Vec keeps its capacity — no allocation inside the frame.
            let mut order = self.draw_order.borrow_mut();
            order.clear();
            order.extend(widgets.keys().copied());
            order.sort_by_key(|id| widgets.get(id).map(|w| w.rect[1] as u32).unwrap_or(0));

            for id in order.iter() {
                let w = &widgets[id];
                let x = (w.rect[0] * sx).max(0.0) as u32;
                let y = (w.rect[1] * sy).max(0.0) as u32;
                let ww = (w.rect[2] * sx).max(0.0) as u32;
                let hh = (w.rect[3] * sy).max(0.0) as u32;

                // Fully-offscreen or degenerate widgets cost a draw
                // call each if not culled — the four-way reject keeps
                // them free.
                if ww == 0 || hh == 0 || x >= g.config.width || y >= g.config.height {
                    continue;
                }

                if w.kind == 0 {
                    // Media quad — textured pipeline + the slot's group.
                    if let Some(slot) = media_slots.get(id) {
                        pass.set_scissor_rect(
                            x,
                            y,
                            ww.min(g.config.width - x),
                            hh.min(g.config.height - y),
                        );
                        pass.set_pipeline(&g.media_pipeline);
                        pass.set_bind_group(0, &slot.bind, &[]);
                        pass.draw(0..6, 0..1);
                    }

                    continue;
                }

                let Some(bind) = &w.bind else { continue };

                let pipeline = match w.kind {
                    widgets::kinds::FLAG => &g.flag_pipeline,
                    _ => match g.pipelines.get(&w.kind) {
                        Some(p) => p,
                        None => continue,
                    },
                };

                pass.set_scissor_rect(
                    x,
                    y,
                    ww.min(g.config.width - x),
                    hh.min(g.config.height - y),
                );
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.draw(0..3, 0..1);
            }

            pass.set_scissor_rect(0, 0, g.config.width, g.config.height);
            g.text.render(&mut pass);
        }

        g.queue.submit([encoder.finish()]);
        frame_tex.present();
    }
}
