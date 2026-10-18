# rust/ — WebGPU + WASM surface crates

Dual-layer render architecture: WebGPU canvases paint pixels while the
prerendered HTML keeps every semantic contract (SEO, JSON-LD, routing,
screen readers, reader mode). The DOM is always the fallback — the JS
bridge returns `null` whenever WebGPU or the wasm artifact is missing.

**Start here: [`ARCHITECTURE.md`](ARCHITECTURE.md)** — the full system
design: the per-artifact model, the gpu-mini shared layer, every
widget's uniform contract, frame math, media decode paths, and the
safe-tuning tables.

## Why one artifact per animation (and not a monolith)

The first iteration put every widget in a single `overlay` crate. It
compiled to **~4 MB** — not because the widgets are big, but because
the `wgpu` Rust framework statically links ~3 MB of validation/hal/
naga into every artifact. Splitting that framework across N widgets
would have multiplied the cost, so the architecture flipped:

- **`gpu-mini`** — a shared `rlib` of raw `web-sys` WebGPU bindings
  (~30 API calls total). web-sys emits thin JS bindings; dead code is
  stripped per artifact.
- **One crate per animation** — each widget gets its own cdylib, its
  own GPUDevice, its own canvas. Every artifact lands at **~46-54 KB**,
  fetched only when its DOM surface mounts.
- **`skeleton` loads FIRST** — it is the loading placeholder itself,
  the only artifact allowed on the critical path.
- **Fallback stays the existing WebGL widgets** — no wgpu `webgl`
  feature needed anywhere; a missing artifact or `navigator.gpu` is a
  clean no-op.

## Workspace

| crate              | artifact (~KB) | surface                                                                                                              |
| ------------------ | -------------- | -------------------------------------------------------------------------------------------------------------------- |
| `gpu-mini/`        | — (rlib)       | Shared web-sys WebGPU boilerplate: surface init, bind-group/pipeline helpers, textures, the uniform-widget scaffold. |
| `skeleton/`        | 49             | Loading shimmer — morphing 0/1 glyph cells + per-rect storage sheet. **First-loading artifact.**                     |
| `burger/`          | 50             | Nav burger hairline bars — fluid wave + breathe.                                                                     |
| `flag/`            | 52             | Locale flag — wind warp + specular + split-flag textures.                                                            |
| `switch-slider/`   | 48             | Preference toggle — pill track + per-context waveform.                                                               |
| `theme-slider/`    | 53             | Day/system/night slider — full procedural sky scene.                                                                 |
| `close-button/`    | 47             | Modal close — liquid fill + rotating self-drawn X.                                                                   |
| `menu-background/` | 47             | Fullscreen menu — domain-warped fbm contour field.                                                                   |
| `burn/`            | 54             | Mosaic letter-burn — CPU particle sim + instanced quads.                                                             |
| `media/`           | 46             | Decoded image/video textured quad (`ImageBitmap`/`VideoFrame`).                                                      |
| `diagram/`         | 132            | Local mermaid replacement — flowchart + sequenceDiagram → SVG.                                                       |
| `scene3d/`         | (in progress)  | three.js replacement for star-field + earth-playground — math foundation landed.                                     |

## Toolchain

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

## Build

```sh
cd rust/<crate>
wasm-pack build --target web --release --out-dir ../../website/public/wasm/<crate>
```

Artifacts land in `website/public/wasm/<slug>/` (served at
`/wasm/<slug>/`, immutable-cacheable). The bridge
(`core/utils/gpu/webgpu-widget.ts`) imports each artifact lazily via
`@vite-ignore`, so missing artifacts are clean no-ops — `yarn build`/
`yarn verify` never require the Rust toolchain.

## Adding a widget

1. `rust/<slug>/` — `Cargo.toml` (cdylib + `gpu-mini` path dep),
   `src/<slug>.wgsl`, `src/lib.rs` (~120 lines on the `gpu_mini::Widget`
   scaffold; see `burger/` for the minimal template, `flag/` for
   textures, `skeleton/` for storage, `burn/` for instancing).
2. `core/tokens/motion/gpu-widgets.ts` — a `GPU_WIDGET_KINDS` id, an
   `GPU_WIDGET_ARTIFACTS` entry (`url`, `cls`, `eager`), and a
   `GPU_WIDGET_UNIFORM_F32S` count.
3. `wasm-pack build` → copy `pkg/*` into `website/public/wasm/<slug>/`.
4. The mount site calls `mountWebGpuWidget(kind, canvas)` and keeps
   its WebGL renderer behind the `null` return.
