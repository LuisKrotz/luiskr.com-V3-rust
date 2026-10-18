# skeleton — the first-loading artifact

**~49 KB.** The loading-shimmer surface — the ONLY animation allowed
on the critical path, because it _is_ the loading placeholder. Every
other widget lazy-loads on surface mount; this one loads first and
stays.

```text
 DOM skeleton layer mounts (host gets .has-skeleton-layer)
        │
        ▼
 mountWebGpuWidget(SKELETON, canvas)
        │  artifact is `eager: true` in GPU_WIDGET_ARTIFACTS —
        │  the fetch may start before the canvas exists
        ▼
 Skeleton::init → gpu-mini pipeline + a STORAGE buffer
        │  (24 rects × 4 vec4 rows — the "sheet")
        ▼
 per frame: widget.frame(dt) → the sheet's rects morph
   glyph-zero ↔ glyph-one per cell, per-cell cadence staggered,
   density gates + row raggedness + resolve collapse
```

## The sheet contract — why storage, not uniform

The sheet holds up to `MAX_RECTS = 24` rects, 4 `vec4` rows each:
`rect` (x,y,w,h) · `meta` (radius, cell size, row size, unused) ·
`sbase` · `sink`. A uniform block caps at a few KB and is meant for
per-draw constants — a per-rect data table belongs in a **storage
buffer**, which the shader indexes by `rect_index`. The JS/Rust sync
contract (`SHEET_VEC4S = MAX_RECTS * 4`) is in `lib.rs`; `count` must
never exceed `MAX_RECTS` or the shader reads garbage rows.

## Safe tuning

- `MAX_RECTS` (24): max simultaneous skeleton placeholders. Raise it
  and raise `SHEET_VEC4S` + the WGSL `MAX_RECTS` in the same commit —
  the three are one contract.
- `STORAGE_VEC4S`: derived — never edit directly.
- The shader's morph cadence / density gate constants live in
  `skeleton.wgsl` — each is commented with units + safe range.
