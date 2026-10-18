# switch-slider — preference toggle (pill track + knob)

**~48 KB.** Lazy artifact — mounts when a preference toggle renders;
the CSS/WebGL toggle is the fallback.

```text
 mountWebGpuWidget(SWITCH_SLIDER, canvas)
        │
        ▼
 SwitchSlider::init → gpu-mini scaffold (uniforms only)
        │
        ▼
 per frame: pill track SDF + knob sphere + per-context pattern:
   ECG waveform / blueprint grid / motion streaks
```

**PIXEL-SPACE SHADER** — the original canvas is a fixed 28px-tall
control; all constants are raw px and do NOT scale. `res` must be the
canvas's real CSS size and the canvas must match the design size, or
the control renders wrong-sized.

- `set_state(progress, knob_x, context)` — `progress` 0..1 toggle
  travel; `context` picks the pattern family.
- `UNIFORM_F32S = 12` — mirror table in `lib.rs` doc.
