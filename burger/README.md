# burger — nav hamburger hairline bars

**~50 KB.** Lazy artifact — mounts only when the nav renders; the
DOM/CSS burger is the fallback.

```text
 nav mounts → mountWebGpuWidget(BURGER, canvas)
        │
        ▼
 Burger::init → gpu-mini scaffold (uniforms only, no textures)
        │
        ▼
 per frame: uniform block [res.xy, time, dark, rect(4)]
   → three rounded bars, staggered traveling wave + breathing,
     theme-aware ink (dark uniform flips light/dark)
```

- `set_dark(v)` — theme flip (0=light icon on dark, 1=dark on light).
- `set_size(w,h)` — canvas CSS size feeds `res`; the shader works in
  px, so a resized canvas needs the new size pushed.
- Uniform layout: `UNIFORM_F32S = 8` — mirrors
  `GPU_WIDGET_UNIFORM_F32S[BURGER]`; bump both together.
