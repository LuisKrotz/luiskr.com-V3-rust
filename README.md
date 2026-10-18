# rust/ — WebGPU + WASM surface crates

Dual-layer render architecture: a WebGPU canvas paints pixels while the
prerendered HTML keeps every semantic contract (SEO, JSON-LD, routing,
screen readers, reader mode). The DOM is always the fallback — the JS
bridge returns `null` whenever WebGPU or the wasm artifact is missing.

**Start here: [`ARCHITECTURE.md`](ARCHITECTURE.md)** — the full system
design: data-flow diagrams, the input-ring protocol, the widget
registry, per-frame math, media decode paths, every sync contract, and
the safe-tuning tables.

## Workspace

| crate      | surface                                                            |
| ---------- | ------------------------------------------------------------------ |
| `overlay/` | Single-surface widget compositor — instanced quads, per-widget WGSL pipelines (burger/flag/switch/theme/close/menu/skeleton), burn sim, media texture pool, glyphon text, zero-copy input ring. See `overlay/README.md` |
| `diagram/` | Local mermaid replacement — flowchart + sequenceDiagram parse/layout/SVG in ~130KB wasm (was ~1MB JS). See `diagram/README.md` |

## Toolchain

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

## Build

```sh
cd rust/overlay
wasm-pack build --target web --release --out-dir ../../website/public/wasm/overlay
```

Output lands in `website/public/wasm/overlay/` (served at
`/wasm/overlay/`, shipped through the public dir — immutable-cacheable).
The bridge (`core/utils/gpu/webgpu-overlay.ts`) imports
`/wasm/overlay/luiskr_overlay.js` lazily via `@vite-ignore`, so a missing
artifact is a clean no-op — `yarn build`/`yarn verify` never require the
Rust toolchain.

## Contracts

- **Frame clock**: JS owns rAF (`Overlay::frame(dt)`), honoring the
  120Hz/240Hz budget — wasm never spins its own loop.
- **Input**: JS writes packed f32 records into wasm linear memory at
  `input_buffer_ptr()` (ring meta at `input_meta_ptr()`); zero boundary
  copies. Kind codes must match `INPUT_TYPES` in the bridge.
- **Text parity**: `set_text` mirrors a hidden-DOM node 1:1 — glyphon
  wraps on the DOM-measured width the caller passes in, so rasterized
  breaks cannot diverge from the accessible text.
- **Fonts**: `install_font(family, bytes)` is pushed the route font —
  Fira Sans bytes only on `ru`, Raleway elsewhere (mirrors the DOM gate).
- **Version lockstep**: wgpu and glyphon release in pairs — bump both
  together (`wgpu 25` ⇄ `glyphon 0.9`).
