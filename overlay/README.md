# overlay — the WebGPU paint surface

One canvas. One render pass. Every animation widget scissored into its
own rectangle. See [`../ARCHITECTURE.md`](../ARCHITECTURE.md) for the
full system design; this file is the crate-level map.

```text
┌─ JS bridge (core/utils/gpu/webgpu-overlay.ts) ────────────────────┐
│ rAF owner · event writer · capability gate · widget registrar    │
└───────┬──────────────────────────────────────────────┬───────────┘
        │ pointers (linear memory)                     │ calls (once/frame + on change)
        ▼                                              ▼
┌─ input.rs ──────────────┐              ┌─ lib.rs ─────────────────┐
│ INPUT_RING 1024×4 f32   │              │ Overlay                  │
│ INPUT_META [head,tail]  │──drain──────▶│  frame(dt)               │
│ drain() — overflow-safe │              │   ├─ burn sims → quads   │
└─────────────────────────┘              │   ├─ write_buffer        │
                                         │   ├─ scissored widgets   │
┌─ layout.rs ─────────────┐              │   ├─ media quads         │
│ Quad (instanced vert)   │◄─emit────────│   └─ glyphon text        │
│ Hit/Scene — hit tests   │              └───────────┬──────────────┘
└─────────────────────────┘                          │ submit+present
┌─ widgets/ ──────────────┐                          ▼
│ burger/flag/switch/     │                     surface (sRGB,
│ theme/close/menu/       │                     Mailbox mode)
│ skeleton — WGSL each    │
│ burn — CPU sim → quads  │
└─────────────────────────┘
┌─ media.rs ──────────────┐   ┌─ text.rs ───────────────┐
│ image decode → RGBA8    │   │ glyphon: font db, atlas │
│ WebCodecs frame ingest  │   │ TextJob rebuild/frame   │
│ MediaSlot texture pool  │   │ (cache → cheap for      │
└─────────────────────────┘   │  static strings)        │
                              └─────────────────────────┘
```

## Exported surface (`#[wasm_bindgen]`)

| Export | When called | Cost |
| ------ | ----------- | ---- |
| `new()` | bridge boot | trivial |
| `init(canvas)` | once — async adapter/device/surface | ms, once |
| `input_buffer_ptr` / `input_meta_ptr` | once — JS keeps the pointers | free |
| `set_css_size(w,h)` | on resize | free |
| `resize(w,h)` | on resize (device px) | surface reconfigure only |
| `frame(dt)` | every rAF | the whole pipeline |
| `register_widget(kind,x,y,w,h)` | DOM mirror mounts | one ubuf + lazy pipeline |
| `register_media(x,y,w,h)` | media surface mounts | one ubuf |
| `set_widget_uniform(id,f32[])` | per state change | one write_buffer, clamped to spec |
| `set_widget_rect(id,x,y,w,h)` | mirror rect moves | field write + burn origin |
| `unregister_widget(id)` | DOM mirror unmounts | drops resources |
| `upload_flag_textures(id,…)` | flag surfaces | 2 textures, once |
| `write_skel_sheet(id,f32[])` | skeleton layout changes | one write_buffer |
| `upload_media_image(id,bytes)` | image loads | decode + one texture |
| `push_video_frame(id,rgba,w,h)` | per video frame | one write_texture |
| `burn_set_letters/ink/colors/stage/dir` | session mount + per frame | Vec copies + field writes |
| `set_text(id,…)` | DOM text changes | slot write |
| `install_font(family,bytes)` | route font arrives | fontdb insert |
| `hover_group()` | bridge reads post-frame | free |

## Files

| File | Owns |
| ---- | ---- |
| `lib.rs` | The `Overlay` type, GPU state, the frame pipeline, all exports |
| `input.rs` | Ring layout, `drain`, overflow semantics |
| `layout.rs` | `Quad` vertex layout, `Hit`/`Scene` hit-testing |
| `media.rs` | `decode_image`, `make_slot`, `write_frame` — texture pooling |
| `text.rs` | glyphon wrapper: `TextPipe`, `TextJob`, family routing |
| `widgets/mod.rs` | kind discriminants, `spec_of` registry |
| `widgets/*.rs` | per-widget uniform contract docs |
| `widgets/*.wgsl` | per-widget procedural shaders (heavily commented) |
| `quad.wgsl` | instanced-quad shader (burn particles + scene quads) |

## Build

```sh
cd rust/overlay
wasm-pack build --target web --release
# artifact → pkg/  (copy to website/public/wasm/overlay/ to serve)
```

`cargo check --target wasm32-unknown-unknown` must be clean — no
warnings tolerated (CI gate candidate once wired into `yarn verify`).

## Conventions

- Zero allocation in `frame` — persistent scratch Vecs, `with_capacity`
  at construction, geometric GPU-buffer growth only.
- Everything integrates in `dt` seconds — no per-frame constants.
- Every uniform slot documented in the owning `*.rs` with units; every
  visual constant commented inline in the `.wgsl` with a safe range.
- Clamps before every wgpu call that validates (scissor, texture dims,
  write extents) — wgpu validation panics are uncatchable on wasm.
