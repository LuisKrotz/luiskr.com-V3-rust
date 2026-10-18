# burn — mosaic title paper-combustion sim

**~54 KB.** Lazy artifact — mounts on mosaic-tile interaction. The
existing canvas burn is the fallback. This is the only widget whose
simulation runs on the CPU side of wasm (`sim.rs`) — particles are
too stateful for a stateless fragment shader.

```text
 tile interaction → mountWebGpuWidget(BURN, canvas)
        │
        ▼
 begin(dir, ox, oy, seed) → set_letters(f32[6×n])
   set_ink(f32[w]) → set_colors(f32[12]) → sim ready
        │
        ▼
 per frame: tick(dt) integrates ≤110 particles (xorshift32 RNG —
   deterministic replay), emits 16-f32 instance records →
   storage buffer → ONE draw(6, n) instanced call
```

## Why instancing via storage buffer

The sim emits a variable count of quad records per frame. A storage
buffer indexed by `@builtin(instance_index)` needs no vertex-layout
descriptors — one `write_buffer` per frame, one record shape (4 vec4s)
to keep in sync with `burn.wgsl`.

- `set_dir(+1/−1)` — burn ↔ rebuild on un-hover (replay, not reset).
- `set_stage(v)` / `set_origin(x,y)` — front progress + tile origin.
- Letter stride is 6 f32s (x0,y0,x1,y1,v,delay) — mirrors the JS
  flatten order; `P_MAX`-equivalents documented in `sim.rs`.
