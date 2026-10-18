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

The site's complex animations (flag waves, burger menu morph, tile
burn, switches, theme slider, menu background, skeleton loaders) used
to run as **N independent canvases** — each with its own WebGL context,
its own `requestAnimationFrame` loop, its own clear pass, and its own
shader plumbing. The costs of that structure:

| Traditional structure | Cost it imposes |
| --------------------- | --------------- |
| N WebGL contexts | Each context reserves a command queue + swapchain; drivers cap contexts (~16) and evict the least-recently-used — pages with many canvases silently lose contexts. |
| N rAF loops | Every loop wakes the main thread separately; wakeups can't batch, jitter multiplies, and frame-budget governance can't measure one clock. |
| N clears + N composites | The compositor blends every canvas over the page every frame — full-screen overdraw per canvas even when each paints a 40px widget. |
| State duplicated per canvas | Uniform math, resize handling, and device-pixel-ratio logic reimplemented per surface — every surface drifts. |

The architecture here replaces all of that with **one canvas, one
surface, one clock, one render pass**, with every widget scissored to
its own rectangle inside it.

---

## 2. The two-layer rule

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
│ Layer 1 — WebGPU canvas (paint only)                            │
│                                                                │
│  One fullscreen <canvas> behind/above the DOM. Rust+wasm drives  │
│  wgpu; every widget is a scissored sub-rect with its own WGSL    │
│  pipeline; text renders through glyphon; media through pooled    │
│  textures. Paints pixels — owns no semantics.                   │
└────────────────────────────────────────────────────────────────┘
```

**The zero-regression rule:** Layer 1 can never hide content. If
WebGPU is absent (Safari < 26, old Chrome/Firefox, IE, some Android
WebViews), if the `.wasm` artifact fails to load, or if `init` throws —
the bridge returns `null`, no canvas mounts, and the DOM renders
exactly as it always did. Acceleration is strictly additive.

This is why the DOM mirror discipline exists: every pixel Layer 1
paints has a DOM counterpart a crawler or screen reader can reach.
SEO, accessibility, and reader mode are unaffected by the GPU path.

---

## 3. Full data flow

```text
 pointer/wheel events (passive, coalesced upstream)
        │
        ▼
 JS bridge (core/utils/gpu/webgpu-overlay.ts)
        │  writes packed f32 records DIRECTLY into wasm linear memory
        │  at input_buffer_ptr(); bumps head at input_meta_ptr()
        │  ── zero calls, zero copies, zero allocations per event ──
        ▼
┌─ wasm linear memory ────────────────────────────────────────────┐
│ INPUT_RING: 1024 records × [kind, x, y, payload]                 │
│ INPUT_META: [head, tail]                                         │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼  once per rAF tick (JS owns the clock)
 Overlay::frame(dt_seconds)
   1. drain ring → update hover (move/down only — see §6)
   2. tick burn sims → particles emit quads into ONE shared buffer
   3. write_buffer: entire quad array, one memcpy
   4. encode ONE render pass:
        clear → instanced quad draw → per-widget scissored draws
        (sorted by rect.y, DOM paint order) → glyphon text
   5. submit + present  (Mailbox mode — newest frame wins)
        │
        ▼
 hover_group() → bridge routes click intents to the DOM router
```

**Why JS owns the clock** (not a wasm-side loop): the project enforces
a single 120Hz mobile / 240Hz desktop frame budget
(`AGENTS.md` rule 27). One rAF means one measurable tick, one place to
pause on `visibilitychange`, and no competing clocks. The wasm side
never waits, never schedules, never sleeps — it computes what a frame
needs when asked.

**Why `dt` in seconds, never frame counts:** all motion integrates as
`state += rate × dt`. At 60Hz dt≈0.0167s; at 240Hz dt≈0.0042s — the
visual result is identical because the math compensates per-tick.
Any constant measured in units-per-second (`px/s`, `rad/s`, `flips/s`)
is dt-safe by construction; a per-frame step constant is a bug waiting
for a refresh-rate change. Every shader comment tags its units.

---

## 4. The input ring — zero-copy event transport

Two regions inside wasm linear memory, exposed to JS as raw pointers:

```text
 INPUT_META  (i32[2])       INPUT_RING (f32[1024 × 4])
 ┌──────┬──────┐            ┌──────────────────────────────┐
 │ head │ tail │            │ [kind, x, y, payload] × 1024  │
 └──────┴──────┘            └──────────────────────────────┘
        ▲ writer: JS event pump          ▲ reader: Rust drain()
        (bumps head per record)          (bumps tail to head)
```

- **Record layout**: `[kind, x, y, payload]` — kind is the
  `INPUT_TYPES`/`input::kinds` discriminant (move=0, down=1, up=2,
  wheel=3); x/y in canvas CSS pixels; payload is event-specific
  (wheel deltaY already converted to px by the bridge).
- **The writer never calls into wasm** — it writes 4 floats through a
  `Float32Array` view over `memory.buffer`, then bumps `head` in an
  `Int32Array` view. Per-event cost: 5 typed-array writes. No FFI call,
  no serialization, no allocation.
- **Overflow clamp**: when `head - tail > CAPACITY`, drain only
  consumes the newest `CAPACITY - 1` records and skips the rest. Ring
  wrap makes the newest records contiguous at the tail end — one
  duplicated read (the in-flight newest record) is accepted rather
  than risking a torn read of a record the JS writer might be
  mid-write on. 1024 records ≈ 60Hz of events for 17 seconds of a
  paused tab — overflow in practice means the tab slept, and skipping
  stale input is the desired behavior anyway.
- **Freshness upstream**: the bridge uses `getCoalescedEvents()` and
  pushes only the freshest sample per native event — the ring never
  carries intermediate positions the renderer would discard.

### Changing the ring safely

| Constant | Where | Safe range | Effect |
| -------- | ----- | ---------- | ------ |
| `RING_CAPACITY` (1024) | `input.rs` | 128–65536 | Larger = more sleeping-tab slack, more memory (16B/record). Must stay a power of two if wrap masking is added later. |
| `RING_F32S` (cap×4) | `input.rs` | derived | Record stride — changing to 5+ floats per record requires updating the JS writer's stride and the drain's indexing together. |
| `INPUT_TYPES` codes | `input.rs` + `webgpu-overlay.ts` | 0–255 practical | Must match EXACTLY on both sides — a mismatch silently reinterprets events. |

---

## 5. The widget system

Each animation surface registers once and receives an id:

```text
 register_widget(kind, x, y, w, h)  →  id  (0 = rejected/fallback)
 set_widget_uniform(id, f32[])      →  per-frame state pushes
 set_widget_rect(id, x, y, w, h)    →  DOM-mirror rect moved
 unregister_widget(id)              →  resources drop
```

At frame time, every registered widget draws once, scissored to its
rect:

```text
 fullscreen triangle (3 verts, no vertex buffer)
        │
        ▼  scissor = widget rect × (surface_px / css_px)
 fragment shader computes EVERYTHING procedurally:
   - widget-local uv  = (fragXY − rect.xy) / res
   - SDF shapes       (distance to bar/knob/track/…)
   - fwidth() AA      (screen-space derivative = 1px edge)
   - procedural anim  (time-driven sines/noise/eases)
```

### Why fullscreen-triangle + scissor instead of real geometry

A fullscreen triangle is 3 vertices covering the viewport; the scissor
clips it to the widget's rect so the fragment shader runs ONLY inside
the widget. Compared to issuing a proper quad per widget:

- no vertex buffer, no index buffer, no vertex math per widget;
- the vertex stage is trivially free (3 fixed positions);
- scissor is a hardware rasterizer feature — the clipped pixels never
  invoke the fragment shader at all. A 40×40 burger costs ~1,600
  fragment invocations, not a full screen's.

### The kind registry (sync contract)

`widgets/mod.rs::kinds` ⇄ `GPU_WIDGET_KINDS` in
`core/tokens/motion/gpu-widgets.ts` — same numbers, same order:

| kind | Rust | WGSL | Anim |
| ---- | ---- | ---- | ---- |
| 1 | `burger.rs` | `burger.wgsl` | 3-bar hamburger: staggered wave, breath, hover squeeze |
| 2 | `flag.rs` | `flag.wgsl` | Locale flag: wind, flutter, split-texture, personality anim |
| 3 | `switch_slider.rs` | `switch_slider.wgsl` | Context switch (ECG / blueprint / streaks) |
| 4 | `theme_slider.rs` | `theme_slider.wgsl` | 3-stop night/system/day scene + ripple |
| 5 | `close_button.rs` | `close_button.wgsl` | Liquid fill + bubbles + rotating X + shockwave |
| 6 | `menu_background.rs` | `menu_background.wgsl` | Domain-warped fBM ink isolines |
| 7 | `skeleton.rs` | `skeleton.wgsl` | Storage-buffer rect sheet + glyph resolve |
| 8 | `burn.rs` | *(quad pass)* | CPU particle sim → shared instanced quads |

`spec_of(kind)` maps each to `{wgsl, uniform_f32s}`; `UNIFORM_F32S` in
each `*.rs` mirrors `GPU_WIDGET_UNIFORM_F32S` on the TS side and is
NOW enforced at write time: `set_widget_uniform` clamps the incoming
slice to the declared size, so an oversized JS write truncates instead
of overflowing the uniform layout.

### Uniform-block pattern

Every widget's uniform starts with the same prologue
(`[0..1]` resolution, then time/progress fields) and ends with the
mirror rect (`[w-4..w-1]` x,y,w,h) — written once per state change by
the bridge. Uniform buffers are 256B per widget (≥ the largest block,
16 f32s = 64B) — sized once at registration, `write_buffer`-updated
forever, never re-created.

### Adding a new widget — checklist

1. Write `widgets/<name>.wgsl` — fullscreen `vs_main`, `fs_main` that
   reads `uv`/`res`/`rect` from the Uniforms block.
2. Write `widgets/<name>.rs` — `UNIFORM_F32S` + `WGSL` include + doc of
   the slot layout (every slot's unit + safe range).
3. Add `kinds::<NAME>` to `widgets/mod.rs` and a `spec_of` arm.
4. Add the same discriminant + uniform size to
   `core/tokens/motion/gpu-widgets.ts`.
5. Register from the bridge where the DOM surface mounts; feed
   `set_widget_uniform` from the component's existing state.
6. If it's a burn-type sim instead of a shader, model on `burn.rs`.

---

## 6. Per-frame math, annotated

```rust
// Stage 1 — input. Hit-test only on MOVE/DOWN: pointer-up leaves the
// same widget under the cursor, wheel doesn't move it — skipping both
// saves a scene traversal per event.
drain(meta, ring, |kind, x, y, _| { … });

// Stage 2 — sims emit quads. Burn particles integrate in dt-space;
// all widgets' particles land in ONE Vec<Quad>.
for sim in burns { sim.tick(dt, &mut scene.quads); }

// Stage 3 — upload. Geometric buffer growth (next_power_of_two):
// O(log n) reallocations ever, then zero per frame — Vec::push logic.
if quads.len() > cap { cap = quads.len().next_power_of_two(); … }
queue.write_buffer(&quad_buf, 0, cast_slice(&quads));   // one memcpy

// Stage 4 — CSS→device scissor. sx = surface_px / css_px ≈ DPR.
// .max(0.0) keeps partially-offscreen rects legal;
// .min(width - x) clamps inside the surface (wgpu validates scissor).
let x  = (rect.x * sx).max(0.0) as u32;
let ww = (rect.w * sx).max(0.0) as u32;
if ww == 0 || x >= surface_w { continue; }              // free cull
pass.set_scissor_rect(x, y, ww.min(surface_w - x), hh.min(surface_h - y));
pass.draw(0..3, 0..1);                                  // SDF widgets
pass.draw(0..6, 0..1);                                  // media quads
```

**The sort**: widgets draw sorted by `rect.y` — mirroring DOM paint
order so stacked widgets composite the way their DOM twins would.
The sort uses a persistent scratch `Vec<u32>` (`draw_order`) that is
cleared and refilled each frame — the allocation happens once per
session, not once per frame (zero-allocation frame rule, `AGENTS.md`
rule 27). Widget ids sort through the `widgets` map, so the scratch
holds plain `u32`s, not references — no borrow fights.

---

## 7. Media: decode and paint, off the main thread

```text
 image bytes (jpeg/png/webp fetched same-origin)
   → upload_media_image(id, bytes)
       image::load_from_memory → RGBA8            (pure Rust, in wasm)
       make_slot → Texture + TextureView + BindGroup
   → frame draws textured quad, cover-fit UVs

 video (mp4/webm)
   → JS WebCodecs VideoDecoder → RGBA frame bytes
   → push_video_frame(id, rgba, w, h)
       same size: queue.write_texture into existing texture  ~µs
       new size:  rebuild slot (texture + bind group)        ~ms, rare
```

Design decisions, each deliberate:

- **`Rgba8Unorm`, not sRGB**: the surface is sRGB; marking the texture
  sRGB too would double-convert and wash colors out.
- **`mip_level_count: 1`**: CDN derivatives are already widget-sized —
  mips would cost 33% more VRAM for aliasing that never happens.
- **Usage = `TEXTURE_BINDING | COPY_DST` only**: no `RENDER_ATTACHMENT`,
  no `COPY_SRC` — we never read pixels back (readbacks stall the queue).
- **WebCodecs, not a Rust software decoder**: hardware video decode is
  faster than any wasm codec and already ships in the browser; Rust
  handles images because `image` is small, correct, and decode moves
  off the main thread into the module.
- **Texture pooling via `MediaSlot`**: the bind group wraps the same
  texture forever — frame N costs one `write_texture`, never a bind
  rebuild. Only a resolution change (adaptive streams) rebuilds.

---

## 8. Burn sim — CPU particles, GPU quads

`widgets/burn.rs` reimplements `mosaic/burn/front.ts` in Rust:

```text
 per letter: own progress p_l = clamp(stage·v − delay, 0..1)
 front_y(x) = eased field height + jag_amp × edge_noise(x, t)
   edge_noise = 3 detuned sines (0.09x·sin + 0.023x·sin + 0.17x·sin)
                — frequencies picked non-harmonic so the edge never
                  visibly repeats within a tile width
 per particle: pos += vel·dt; life -= dt; alpha = life_curve(life)
 spawn rate ∝ front velocity × col_ink[x] (ink occupancy per column)
 RNG: xorshift32 — deterministic, allocation-free, replay-identical
 emitted: ≤110 particles → Quad{pos,size,uv,tint,z,group} each
```

All particles across ALL burn widgets land in the shared `scene.quads`
buffer → ONE instanced draw call paints every ember on the page. The
110-particle cap (`P_MAX`) bounds worst-case work; the
`Vec::with_capacity(P_MAX)` at sim creation means tick-time pushes
never allocate.

`set_dir(+1/−1)` restarts the particle stream (burn ↔ rebuild on
un-hover) while keeping letters/ink/colors — matching the JS session's
hover-out replay exactly.

---

## 9. Text — glyphon mirror

DOM text nodes mirror into `TextNode` slots (`set_text`) only on
content change — never per frame. Each frame rebuilds `TextJob`s and
glyphon prepares + renders them:

- **Why rebuild each frame is cheap**: glyphon caches shaped runs;
  static strings re-shape nothing — prepare is a lookup.
- **Why `max_width_px` is caller-supplied**: the DOM measured the wrap
  width already; passing it in makes glyphon break lines at the same
  point the accessible text breaks — rasterized output cannot diverge
  from what a screen reader announces.
- **Font gating mirrors the DOM gate**: `install_font` receives Fira
  Sans bytes only for `ru` locales, Raleway elsewhere — same rule as
  the stylesheet loader (`core/locale/cyrillic-font.ts`), so glyph
  coverage and font payload match the DOM path exactly.
- **Device-px rasterization**: `resize` forwards the surface size;
  glyphon rasterizes at physical resolution so text stays crisp at
  120Hz+ and on retina.

---

## 10. Surface configuration choices

| Choice | Value | Why |
| ------ | ----- | --- |
| Present mode | `Mailbox` | Three modes exist: `Fifo` queues frames behind vsync (input lag grows at high Hz); `FifoRelaxed` tears; `Mailbox` presents the NEWEST finished frame and drops the rest. At 120/240Hz a queued frame is stale — mailbox minimizes pointer→pixel latency. |
| Frame latency | 2 | The compositor already holds one frame; deeper queues add lag without throughput benefit — the layer is latency-bound, not bandwidth-bound. |
| Format | sRGB preferred | DOM colors are sRGB; a non-sRGB target shifts every painted color a gamma step away from the DOM beneath it. `unwrap_or(formats[0])` keeps exotic displays alive. |
| Power preference | `HighPerformance` | Requests the discrete GPU on hybrid laptops — the whole point of the layer. |
| Blend | `ALPHA_BLENDING` | Standard `src·α + dst·(1−α)` — required for widgets/text/embers to composite over the page. |
| Backends | `BROWSER_WEBGPU | GL` | WebGPU primary; the `webgl` wgpu feature allows a WebGL2-emulated surface on pre-WebGPU browsers that still support wasm (a middle rung above pure DOM fallback). |

---

## 11. Capability gating & the compat lane

```text
 mountWebGpuOverlay()
   navigator.gpu? ──no──▶ return null → DOM path (unchanged)
   wasm artifact? ─no──▶ return null → DOM path
   init() throws? ──yes─▶ return null → DOM path
        │
        ▼
 Overlay live: canvas mounts, listeners attach (passive),
 widgets register, rAF starts — hidden-tab pauses the loop.
```

Legacy browsers (Safari < 26, Chrome < 113, Firefox < 141, IE10/11,
old iOS) never reach the wasm path — they get the DOM/canvas-2D/WebGL
fallbacks that already exist, plus the legacy tier bundles the build
emits (`dist/v/es2016`/`es2017`). Layer 1 never ships to them, so they
pay nothing for it; modern browsers get the full GPU path.

---

## 12. Differences from a traditional app structure — summary

| Traditional | This architecture | What it buys |
| ----------- | ----------------- | ------------ |
| N canvases, N contexts | 1 canvas, 1 surface | No context exhaustion, one clear, one composite |
| N rAF loops | 1 rAF (JS-owned) | Single measurable clock; one visibility pause point |
| JS-side sim + draw | Rust sim → shared instanced draw | Particles cost one draw call regardless of count |
| Per-event FFI calls | Direct linear-memory writes | ~0 cost per event |
| Per-frame Vec/alloc | Persistent scratch + geometric buffer growth | Steady frames allocate nothing |
| DOM mirrors rebuilt ad hoc | `TextNode` slots updated on change only | Text cost ∝ changes, not frames |
| One texture rebuild per video frame | `write_texture` into pooled slot | Video costs a texel memcpy, not resource churn |
| Uniform buffer per draw call | Per-widget 256B buffer, updated in place | Bind groups never rebuild mid-stream |

---

## 13. Sync contracts — the two-sided tables

These MUST stay identical across the language boundary; a mismatch
fails silently (wrong animation, never an error):

| Contract | Rust side | TS side |
| -------- | --------- | ------- |
| Input kind codes | `input::kinds` | `INPUT_TYPES` in `webgpu-overlay.ts` |
| Widget kinds | `widgets::kinds` | `GPU_WIDGET_KINDS` in `gpu-widgets.ts` |
| Uniform sizes | `UNIFORM_F32S` per `*.rs` | `GPU_WIDGET_UNIFORM_F32S` |
| Uniform slot layout | doc comment in each `*.rs` | the bridge's write order |
| Flag anim codes | `flag::anims` | `FLAG_ANIMS` token |
| Skeleton sheet | `RECT_FIELDS`/`STORAGE_VEC4S` | sheet write size (24×16 f32) |
| Burn letter stride | 6 f32s/letter | flatten order x0,y0,x1,y1,v,delay |

`set_widget_uniform` clamps writes to `spec.uniform_f32s`, so the size
contract fails safe (truncation) rather than overflowing.

---

## 14. Safety & tuning quick reference

- **Numbers in `*.wgsl`**: visual constants — each is commented with
  units and safe ranges. UV-space numbers (0..1) are resolution-safe;
  px-space numbers (switch slider, close button) assume the design
  size stated at the top of the file — resizing the control means
  re-tuning those constants.
- **Numbers in `*.rs`**: capacity/protocol constants. `RING_CAPACITY`,
  `P_MAX`, `WIDGET_UBUF_SIZE`, `STORAGE_VEC4S` — documented per
  constant; raising capacity is always safe, lowering below the
  stated minimum breaks validation or truncates.
- **wgpu calls**: validation errors panic on wasm (uncatchable). The
  code pre-clamps everything wgpu validates (scissor bounds, texture
  dims ≥1, write sizes) — keep those clamps when editing.
- **`static` memory**: only the two `SyncCell` rings; single-thread
  contract documented at declaration. Don't add more shared statics —
  per-instance `RefCell` fields instead.
