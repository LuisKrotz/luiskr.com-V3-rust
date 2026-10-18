# menu-background — fullscreen menu contour field

**~47 KB.** Lazy artifact — the most expensive shader of the family
(5 fbm evaluations × 4 octaves of value noise ≈ 80 hash evaluations
per pixel, ~660M ops on a 4K viewport), which is exactly why it only
loads while the menu is open. The CSS/WebGL field is the fallback.

```text
 menu opens → mountWebGpuWidget(MENU_BACKGROUND, canvas)
        │
        ▼
 MenuBackground::init → gpu-mini scaffold (uniforms only)
        │
        ▼
 per frame: double domain-warped fbm → contour isolines
   (fwidth-normalized so lines stay ~1px at any zoom) +
   vignette + reveal gate sweeping with menu open progress
```

- `set_state(reveal, r,g,b, alpha, r2,g2,b2)` — `reveal` 0..1 menu
  open progress; the two rgb triples are the ink/field tints pulled
  from live theme tokens.
- `UNIFORM_F32S = 16` — the largest uniform block (two colors + pad).
- **Perf lever**: dropping the shader from 4 octaves to 3 halves the
  noise cost — the WGSL doc marks the exact constant.
