# WebGPU/Rust Render Architecture

How the site's GPU layer works, why it is shaped this way, and how to
change any number in it safely. Read top-to-bottom once; every section
states the math, the intent, and the performance consequence.

> **Audience note.** This document assumes no graphics background.
> Wherever a GPU/Rust term appears it is defined at first use. Every
> tunable constant is named, with its safe range and what breaks if you
> exceed it.

---

## 1. The problem this solves

The site's complex animations (flag waves, burger morph, tile burn,
switches, theme slider, menu background, skeleton loaders, media)
each run on their own canvas. The traditional way to do that — N
independent WebGL contexts driven from JS — imposes real costs:

| Traditional structure        | Cost it imposes                                                                                                                                                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| N WebGL contexts             | Each reserves a command queue + swapchain; drivers cap contexts (~16) and evict the least-recently-used — pages with many canvases silently lose contexts.                                                        |
| JS-side shader plumbing      | Uniform math, resize, DPR, and compile/link boilerplate re-implemented per surface in garbage-collected code.                                                                                                     |
| One monolithic wasm artifact | The first iteration put every widget in ONE crate — it compiled to ~4 MB because the `wgpu` framework statically links ~3 MB per artifact. Splitting that across N crates would multiply the cost, not divide it. |

The architecture here keeps the per-widget-canvas model (contexts are
cheap and isolated) but replaces BOTH problems: **each animation is
its own ~50 KB wasm artifact** built on raw `web-sys` WebGPU bindings —
no framework, no monolith, fetched only when its surface mounts.

---

## 2. The per-artifact model — why the monolith died

```text
┌─ rust/overlay (RETIRED) ─────────────────────────────────────────┐
│ every widget in one crate → links wgpu + glyphon + image          │
│ artifact: ~4 MB — unacceptable on any load path                   │
└──────────────────────────────────────────────────────────────────┘
                          ▼ replaced by
┌─ rust/gpu-mini (rlib, shared boilerplate) ───────────────────────┐
│ raw web-sys WebGPU: init_surface, bind-group/pipeline helpers,    │
│ textures, the uniform-widget scaffold — ~30 API calls total.      │
│ Linked into each widget crate; dead code stripped per artifact.   │
└──────────────────────────────────────────────────────────────────┘
        │ linked into
        ▼
┌─ one cdylib per animation ───────────────────────────────────────┐
│ skeleton 49KB · burger 50KB · flag 52KB · switch 48KB ·           │
│ theme 53KB · close 47KB · menu-bg 47KB · burn 54KB · media 46KB   │
│                                                                 │
│ each = own GPUDevice + own canvas + own pipeline                 │
└──────────────────────────────────────────────────────────────────┘
```

Why this wins:

- **Artifact size ∝ what the surface needs** — a widget pays ~50 KB
  once, when its DOM canvas mounts. The skeleton (the loading
  placeholder itself) is the only artifact allowed on the critical
  path — it loads FIRST, at 49 KB.
- **Isolated failures** — a shader compile error or device loss in
  one widget can never stall another; each falls back to its own
  WebGL/CSS twin independently.
- **No vertex-buffer/sync glue across widgets** — each crate owns a
  trivial pipeline (fullscreen triangle, one uniform block, maybe a
  texture or storage buffer). The sync surface is per-widget: one
  WGSL struct + one JS token entry.
- **Device isolation is cheap** — WebGPU devices on separate canvases
  coexist fine; the "N contexts" problem was a WebGL context-pool
  limit, not a WebGPU constraint.

---

## 3. The two-layer rule

```text
┌────────────────────────────────────────────────────────────────┐
│ Layer 2 — DOM (always mounted, always complete)                 │
│                                                                │
│  <nav>, <h1>-<h6>, <a href>, JSON-LD, <img>, visually-hidden    │
│  mirror text, real focus order, real semantics.                 │
│  Source of truth for: crawlers, reader mode, screen readers,    │
│  no-JS clients, and ANY browser where Layer 1 can't boot.       │
└──────────────┬─────────────────────────────────────────────────┘
               │  bridge mounts only when every capability passes
               ▼
┌────────────────────────────────────────────────────────────────┐
│ Layer 1 — per-widget WebGPU canvases (paint only)               │
│                                                                │
│  Each widget's own <canvas> is driven by its own wasm artifact:  │
│  Rust repacks a uniform block per frame and issues one draw.    │
│  Paints pixels — owns no semantics.                             │
└────────────────────────────────────────────────────────────────┘
```

**The zero-regression rule:** Layer 1 can never hide content. If
WebGPU is absent (Safari < 26, old Chrome/Firefox, IE, some Android
WebViews), if the `.wasm` artifact fails to load, or if `init` throws —
the bridge returns `null` and the existing WebGL/CSS widget renders
exactly as it always did. Acceleration is strictly additive.

---

## 4. Full data flow

```text
DOM surface mounts (nav opens, flag switches, menu expands…)
        │
        ▼
JS bridge (core/utils/gpu/webgpu-widget.ts)
  kind → GPU_WIDGET_ARTIFACTS[kind] → {url, cls, eager}
        │  dynamic-imports /wasm/<slug>/<pkg>.js  (~50 KB, cached)
        ▼
Widget::new() → widget.init(canvas)      (async: adapter→device→configure)
        │  rejection → null → WebGL/CSS fallback unchanged
        ▼
per rAF tick (the bridge owns the clock — single-clock rule):
  widget.frame(dt_seconds)
    1. repack the uniform f32 block (≤64 B, widget-local state)
    2. queue.write_buffer — one upload
    3. encoder → pass → pipeline → bind → draw(3|6|instanced)
       → submit — the canvas presents on task end
        │
        ▼
DOM pointer events → widget setters (set_state/tap/click/hover)
   — each canvas hosts exactly one widget, so events map to setters
     directly; no shared hit-test ring is needed.
```

**Why JS owns the clock** (not a wasm-side loop): the project enforces
a single 120Hz mobile / 240Hz desktop frame budget (`AGENTS.md`
rule 27). One rAF per mounted widget means measurable ticks and one
place to pause on `visibilitychange`. The wasm side never waits,
never schedules — it computes what a frame needs when asked.

**Why `dt` in seconds, never frame counts:** all motion integrates as
`state += rate × dt`. At 60Hz dt≈0.0167s; at 240Hz dt≈0.0042s — the
visual result is identical because the math compensates per-tick.
Any constant measured in units-per-second (`px/s`, `rad/s`, `flips/s`)
is dt-safe by construction; a per-frame step constant is a bug waiting
for a refresh-rate change. Every shader comment tags its units.

---

## 5. gpu-mini — the shared layer that makes 50 KB possible

`rust/gpu-mini` is a plain `rlib` (never built to wasm itself) with
the boilerplate every widget needs:

| Helper                                                          | What it wraps                                                                                     |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `init_surface(canvas)`                                          | `navigator.gpu` → adapter → device → canvas context → configure (async — the only async step)     |
| `configure`/`resize`                                            | context re-bind after a canvas-bitmap resize                                                      |
| `uniform_entry`/`storage_entry`/`texture_entry`/`sampler_entry` | bind-group layout entries                                                                         |
| `new_buffer`/`write_f32s`/`f32_bytes`                           | buffer alloc + f32 upload (Pod reinterpret, no copy)                                              |
| `new_layout`/`new_bind_group` + `bind_*_entry`                  | resource binding                                                                                  |
| `fullscreen_pipeline`                                           | the widget-family pipeline: fullscreen triangle, no vertex buffers, premultiplied-alpha blend     |
| `new_texture`/`upload_rgba`/`bitmap_texture`/`view`             | RGBA8 texture creation + uploads (flag halves, media)                                             |
| `linear_sampler`                                                | shared bilinear clamp-to-edge sampler                                                             |
| `draw`                                                          | get_current_texture → pass → pipeline → bind → scissor → draw(3) → submit; false on stale surface |
| `init_widget`/`widget_frame`/`Widget`                           | the uniform-only scaffold — 90% of widgets are ~120 lines total because of it                     |

Why raw `web-sys` and not `wgpu`: `wgpu` compiles the full Rust WebGPU
stack (~3 MB) into every artifact; `web-sys` emits thin JS bindings
where unused functions vanish at link time. The widget code itself is
a few KB — the framework was the megabytes.

**Stability note:** the web-sys GPU bindings are upstream-unstable —
`.cargo/config.toml` sets `--cfg=web_sys_unstable_apis` for the wasm32
target only. If a web-sys bump renames an API (it has happened —
`GpuImageCopy*` → `GpuTexelCopy*`), the fixes land here once, not in
every crate.

---

## 6. The widget system

Each animation is a crate + a canvas + a registry entry:

```text
 mountWebGpuWidget(kind, canvas)
   → GPU_WIDGET_ARTIFACTS[kind] = {url, cls, eager}
   → import(url) → new mod[cls]() → init(canvas) → rAF loop
   → handle.widget — kind-specific setters called directly:
       burger.set_dark(v)           flag.set_state(hover,anim,split,split_x)
       switch.set_state(p,knob,ctx) theme.set_state(p,knob) + theme.tap(t,x)
       close.set_state(l,d,r)+click(t)  menu.set_state(reveal,colors…)
       skeleton.write_sheet(f32s)+set_state(r,a,count)
       burn.begin(dir,ox,oy,seed)+set_letters/ink/colors/dir/stage/origin
       media.upload_bitmap(bmp)+push_frame(vf)+set_uv_rect(…)
```

### Why fullscreen-triangle instead of real geometry

Every SDF widget draws a fullscreen triangle (3 verts synthesized in
the vertex shader from `@builtin(vertex_index)` — no vertex buffer).
On a dedicated canvas the fragment shader computes the widget-local
`uv` itself. Compared to issuing geometry:

- no vertex buffer, no index buffer, no vertex math;
- the vertex stage is trivially free (3 fixed positions);
- the fragment shader still does ALL the work — SDF edges, animation
  waves, procedural fields — which is what the GPU is for.

(Media and burn differ: media draws a 6-vertex unit quad because its
uv→texture mapping is vertex-natural; burn draws `draw(6, instances)`
with particle records in a storage buffer.)

### The kind registry (sync contract)

`GPU_WIDGET_KINDS` ⇄ `GPU_WIDGET_ARTIFACTS` in
`core/tokens/motion/gpu-widgets.ts` ⇄ `rust/<slug>/` crates:

| kind | crate/artifact     | WGSL                   | Anim                                                               |
| ---- | ------------------ | ---------------------- | ------------------------------------------------------------------ |
| 1    | `burger/`          | `burger.wgsl`          | 3-bar hamburger: staggered wave, breath                            |
| 2    | `flag/`            | `flag.wgsl`            | Locale flag: wind, split-texture, personality anim                 |
| 3    | `switch-slider/`   | `switch_slider.wgsl`   | Context switch (ECG / blueprint / streaks)                         |
| 4    | `theme-slider/`    | `theme_slider.wgsl`    | 3-stop night/system/day scene + ripple                             |
| 5    | `close-button/`    | `close_button.wgsl`    | Liquid fill + bubbles + rotating X + shockwave                     |
| 6    | `menu-background/` | `menu_background.wgsl` | Domain-warped fBM ink isolines                                     |
| 7    | `skeleton/`        | `skeleton.wgsl`        | Storage-buffer rect sheet + glyph resolve — **eager: loads first** |
| 8    | `burn/`            | `burn.wgsl`            | CPU particle sim → storage-buffer instanced quads                  |
| 9    | `media/`           | `media.wgsl`           | Decoded image/video textured quad                                  |

### Uniform-block pattern

Every widget's uniform ends with a rect field (`(0,0,w,h)` — the whole
canvas, since each canvas hosts one widget) and carries `res` = the
canvas CSS px. Uniform buffers are 256 B per widget — sized once at
init, `write_buffer`-updated per frame, never re-created.

### Adding a new widget — checklist

1. `rust/<slug>/Cargo.toml` — cdylib + `gpu-mini` path dep +
   `wasm-opt = ["-Oz", "-all"]` metadata (copy `burger/`'s).
2. `src/<slug>.wgsl` — fullscreen `vs_main`, `fs_main` reading the
   Uniforms block.
3. `src/lib.rs` — ~120 lines on `gpu_mini::init_widget`/`widget_frame`
   (see `burger/`); use the primitives directly for textures/storage
   (see `flag/`, `skeleton/`, `burn/`).
4. `core/tokens/motion/gpu-widgets.ts` — `GPU_WIDGET_KINDS` id,
   `GPU_WIDGET_ARTIFACTS` entry, `GPU_WIDGET_UNIFORM_F32S` count.
5. `wasm-pack build` → `website/public/wasm/<slug>/`.
6. Mount via `mountWebGpuWidget(kind, canvas)`; keep the WebGL path
   behind the `null` return.

---

## 7. Per-frame math, annotated

```rust
// Per widget crate, per rAF tick:
self.clock.set(self.clock.get() + dt);       // absorbed jitter

// Stage 1 — repack the uniform block (≤64 B, widget-local state).
// The layout is field-for-field the WGSL `Uniforms` struct — see each
// crate's module doc for the slot table.
let uniforms: [f32; UNIFORM_F32S] = [res.x, res.y, clock, state…];

// Stage 2 — one upload. The buffer was created once at init; this is
// a ~64 B memcpy into an already-mapped queue write, not an alloc.
gpu_mini::write_f32s(&queue, &ubuf, &uniforms);

// Stage 3 — draw. get_current_texture → pass → pipeline → bind →
// draw(0..3) → submit. The browser presents the canvas texture when
// the current task ends (no explicit present in WebGPU JS).
gpu_mini::draw(&surface, &pipeline, &bind, None)

// Stage 4 — stale-surface recovery: draw() returns false when
// get_current_texture fails (resize mid-frame, occluded tab);
// the caller re-configures so the next rAF recovers — a missed
// frame beats a panic.
gpu_mini::resize(&surface);
```

**Zero-allocation frame rule**: the uniform array is a fixed
`[f32; N]` on the stack; the buffer/pipeline/bind group were created
at init; `frame` allocates nothing. The only frame-time allocation
site anywhere is burn's instance growth — and that doubles
geometrically, so steady frames never re-allocate.

---

## 8. Media: decode and paint, off the main thread

```text
 still image ──▶ JS createImageBitmap(blob)   (browser decodes
        │        off-main-thread — hardware path)
        ▼
 media.upload_bitmap(bitmap, w, h)
        │  copyExternalImageToTexture — GPU→GPU, no CPU pixels
        ▼
 frame draws textured quad, cover-fit UVs (JS computes uv_rect)

 video frame ──▶ JS WebCodecs VideoDecoder
        │        requestVideoFrameCallback (NOT per rAF)
        ▼
 media.push_frame(VideoFrame, w, h) — same copy path, texture swap
```

Design decisions, each deliberate:

- **No `image` crate** — the browser's ImageBitmap decode is
  hardware-backed and off-main-thread already; a Rust decoder would
  cost ~800 KB of artifact AND run slower than the browser's JPEG path.
- **`Rgba8Unorm`, not sRGB**: the surface is sRGB; marking the texture
  sRGB too would double-convert and wash colors out.
- **Single mip level**: CDN derivatives are already widget-sized —
  mips would cost 33% more VRAM for aliasing that never happens.
- **Usage = `TEXTURE_BINDING | COPY_DST` only**: we never read pixels
  back (readbacks stall the queue).
- **Rebind on swap**: bind groups are immutable, so each upload
  creates a fresh texture + bind group — still cheap (µs), and it
  keeps the steady-state draw path branch-free.

---

## 9. Burn sim — CPU particles, GPU instancing

`burn/` reimplements `mosaic/burn/` in Rust (`sim.rs` — the same code
the retired overlay crate used, emitting flat f32 records now):

```text
 per letter: own progress p_l = clamp(stage·v − delay, 0..1)
 front_y(x) = eased field height + jag_amp × edge_noise(x, t)
   edge_noise = 3 detuned sines — non-harmonic frequencies so the
                edge never visibly repeats within a tile width
 per particle: pos += vel·dt; life -= dt; alpha = life_curve(life)
 spawn rate ∝ front velocity × col_ink[x] (ink occupancy per column)
 RNG: xorshift32 — deterministic, allocation-free, replay-identical
 emit: ≤110 particles × 16 f32s → storage buffer
```

The emitted records land in a read-only storage buffer indexed by
`@builtin(instance_index)` — one `draw(6, n)` call paints every ember.
Storage-over-vertex-buffer was chosen because it needs no vertex-
layout descriptors: one `write_buffer` per frame, one record shape to
keep in sync (burn.wgsl's 4×vec4 doc).

`set_dir(+1/−1)` restarts the particle stream (burn ↔ rebuild on
un-hover) while keeping letters/ink/colors — matching the JS session's
hover-out replay exactly.

---

## 10. Surface configuration choices

| Choice       | Value                        | Why                                                                                                                                              |
| ------------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Present mode | browser default              | The WebGPU canvas context presents on task end; explicit mailbox/latency knobs aren't exposed to JS — the compositor owns scheduling either way. |
| Format       | `getPreferredCanvasFormat()` | The browser's native swapchain format (usually bgra8unorm) — zero conversion cost and guaranteed compatible.                                     |
| Alpha mode   | `premultiplied`              | Every shader emits premultiplied rgb; the blend state and canvas mode agree so the canvas composites over the DOM correctly.                     |
| Blend        | premultiplied-over           | `src·1 + dst·(1−src.a)` — required for widgets to composite over the page.                                                                       |
| Backends     | WebGPU only                  | No wgpu `webgl` feature anywhere — pre-WebGPU browsers take the existing WebGL/2D fallback widgets instead of an emulated path.                  |

---

## 11. Capability gating & the compat lane

```text
 mountWebGpuWidget(kind, canvas)
   spec missing?   ──yes──▶ return null → WebGL/CSS path (unchanged)
   navigator.gpu?  ──no───▶ return null → WebGL/CSS path
   artifact 404?   ──yes──▶ return null → WebGL/CSS path
   init() throws?  ──yes──▶ return null → WebGL/CSS path
        │
        ▼
 Widget live: listeners attach (passive), rAF starts —
   hidden-tab pauses the loop; destroy() reaps everything.
```

Browsers below the ES2022 floor (pre-iOS 16 engines, legacy Edge, IE)
never reach the wasm path — the inline loader's probes fail every tier
and they get the fallback notice plus the DOM/canvas-2D/WebGL fallbacks
that already exist. The wasm artifacts never ship to them, so they pay
nothing; supported browsers get the full GPU path.

---

## 12. Differences from a traditional app structure — summary

| Traditional                           | This architecture                              | What it buys                                                       |
| ------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------ |
| One ~4MB wasm monolith                | ~50KB artifact per animation                   | Widgets pay only for what mounts; skeleton rides the critical path |
| `wgpu` Rust framework                 | `web-sys` bindings via `gpu-mini`              | ~60× smaller artifacts; dead code stripped per widget              |
| JS-side sim + draw per particle       | Rust sim → one storage-buffer instanced draw   | Particles cost one draw call regardless of count                   |
| Rust `image` decode                   | `ImageBitmap`/`VideoFrame` → copyExternalImage | Hardware decode, GPU→GPU copy, ~800KB less code                    |
| Per-frame Vec/alloc                   | Fixed-size uniform arrays + geometric growth   | Steady frames allocate nothing                                     |
| Per-widget vertex buffers/descriptors | Fullscreen triangle / storage-buffer instances | One record shape per widget; no layout descriptors to drift        |
| A shared canvas + hit-test ring       | Per-widget canvases, events → setters          | No ring protocol; each canvas hosts one widget                     |

---

## 13. Sync contracts — the two-sided tables

These MUST stay identical across the language boundary; a mismatch
fails silently (wrong animation, never an error):

| Contract            | Rust side                             | TS side                                                                           |
| ------------------- | ------------------------------------- | --------------------------------------------------------------------------------- |
| Widget kinds        | `rust/<slug>/` dir ↔ registry         | `GPU_WIDGET_KINDS` in `gpu-widgets.ts`                                            |
| Artifact url+class  | crate name + `#[wasm_bindgen]` struct | `GPU_WIDGET_ARTIFACTS[kind]`                                                      |
| Uniform sizes       | `UNIFORM_F32S` per `lib.rs`           | `GPU_WIDGET_UNIFORM_F32S`                                                         |
| Uniform slot layout | doc comment in each `lib.rs`          | the bridge/component's write order                                                |
| Flag anim codes     | `flag/` anim doc                      | `GPU_FLAG_ANIMS` token                                                            |
| Skeleton sheet      | `SHEET_VEC4S`/`MAX_RECTS`             | sheet write size (24×16 f32)                                                      |
| Burn letter stride  | 6 f32s/letter                         | flatten order x0,y0,x1,y1,v,delay                                                 |
| Skeleton WGSL copy  | `skeleton/src/skeleton.wgsl`          | `overlay/`'s retired copy documents the lineage — edit both if the shader changes |

---

## 14. Safety & tuning quick reference

- **Numbers in `*.wgsl`**: visual constants — each is commented with
  units and safe ranges. UV-space numbers (0..1) are resolution-safe;
  px-space numbers (switch slider, close button) assume the design
  size stated at the top of the file — resizing the control means
  re-tuning those constants.
- **Numbers in `*.rs`**: capacity/protocol constants (`MAX_RECTS`,
  `UNIFORM_F32S`, `START_CAP`) — documented per constant; raising
  capacity is always safe, lowering below the stated minimum breaks
  validation or truncates.
- **web-sys calls**: most GPU constructors return `Result`; init-time
  failures become `expect` traps (a JS-visible abort the bridge's
  try/catch turns into a fallback). Frame-time calls return bools —
  never panic in `frame`.
- **Clamp before the GPU**: state setters clamp progress/hover/alpha
  to documented ranges — keep those clamps when editing; an
  out-of-range uniform produces wrong pixels, not an error.
- **`RefCell` everywhere**: wasm is single-threaded here (see §15) —
  interior mutability is the idiom, never `static mut`.

---

## 15. Threading model — why no SharedArrayBuffer / wasm-threads

The crates are deliberately **single-threaded inside linear memory**.
True shared-memory wasm threading (`Atomics`, `SharedArrayBuffer`,
`-C target-feature=+atomics`) was evaluated and rejected; this section
records the reasoning so nobody re-litigates it.

### The hard requirement

`SharedArrayBuffer` is only exposed when the document is
**cross-origin isolated**, i.e. the server sends BOTH:

    Cross-Origin-Opener-Policy: same-origin
    Cross-Origin-Embedder-Policy: require-corp

This site cannot send that pair:

- `COOP: same-origin` breaks OAuth/popup flows and any
  `window.open` hand-off; the current `same-origin-allow-popups`
  value is deliberate.
- `COEP: require-corp` makes the browser REFUSE every cross-origin
  subresource that does not answer `Cross-Origin-Resource-Policy`.
  All media is served from `storage.googleapis.com` (GCS does not
  emit CORP on this bucket), so COEP would blank every cover,
  video, and star-field texture on the site.

### What we do instead — worker-pool threading

Multi-threaded CPU work happens across **Web Workers** owned by
`core/utils/wasm/wasm-pool.ts`:

    main thread ──postMessage──▶ worker[0..N] each with own wasm instance
                                   │
                                   └─ zero-copy: ArrayBuffer is
                                      TRANSFERRED (not copied);
                                      ImageBitmap moves by reference

- Each worker instantiates its own copy of the module — no shared
  linear memory, so no COOP/COEP requirement.
- CPU-heavy batch work (media conversion, layout, hashing) is
  dispatched there; the main thread keeps input + paint.
- Workers communicate results via `postMessage` with transfer
  lists — zero-copy semantics without shared memory.

### Performance comparison

| Approach               | Isolation header needed | Media from GCS  | Sync cost            |
| ---------------------- | ----------------------- | --------------- | -------------------- |
| SAB + atomics          | `crossOriginIsolated`   | BLOCKED by COEP | Shared-mem ops (~ns) |
| Worker pool + transfer | none                    | works           | postMessage (~µs)    |

For frame-budget work the difference is negligible: dispatches are
batched per frame, not per pixel — µs-level message overhead is
invisible against a 4ms/8ms frame budget, while COEP would be a
hard site-wide media outage.

### If the constraint ever changes

Should every asset host gain `Cross-Origin-Resource-Policy: cross-origin`
AND popups get redesigned, shared-memory wasm becomes reachable:
compile with `-C target-feature=+atomics -C link-arg=--shared-memory`,
gate on `globalThis.crossOriginIsolated === true`, and keep the
worker-pool path as the fallback (the code must not REQUIRE SAB —
Safari shipped isolation later and embedded webviews may never).
