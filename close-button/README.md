# close-button — modal close control

**~47 KB.** Lazy artifact — mounts when a modal opens; the CSS/WebGL
close button is the fallback.

```text
 mountWebGpuWidget(CLOSE_BUTTON, canvas)
        │
        ▼
 CloseButton::init → gpu-mini scaffold (uniforms only)
        │
        ▼
 per frame: circular button — liquid fill + foam crest rising on
   hover, self-drawing slowly-rotating X strokes, rising bubbles,
   click shockwave ring
```

- `set_state(liquid, draw, rot)` — hover fill height, X draw progress,
  X rotation (all driven by the DOM hover/press state).
- `click(time)` — records the click instant; the shockwave expands
  from `time` forward.
- `UNIFORM_F32S = 12` — mirror table in `lib.rs` doc.
