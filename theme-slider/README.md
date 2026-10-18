# theme-slider — day/system/night preference slider

**~53 KB.** Lazy artifact — the largest widget shader (a full
procedural scene), lazy because its GPU cost only makes sense while
the preferences UI is open. WebGL/CSS slider is the fallback.

```text
 mountWebGpuWidget(THEME_SLIDER, canvas)
        │
        ▼
 ThemeSlider::init → gpu-mini scaffold (uniforms only)
        │
        ▼
 per frame: sky gradient + deterministic stars + sun/corona +
   aurora + crescent moon + two parallax mesas + 3D knob;
   `progress` 0..2 morphs night → system → day
```

- `set_state(progress, knob_x)` — `progress` is the 3-stop position.
- `tap(time, x)` — starts the ripple: `click_time` + `click_x` feed a
  ring that expands from the tap point.
- `UNIFORM_F32S = 12` — mirror table in `lib.rs` doc.
