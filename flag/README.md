# flag — locale flag (wind warp + split textures)

**~52 KB.** Lazy artifact — mounts when the flag widget renders;
the WebGL flag is the fallback.

```text
 mountWebGpuWidget(FLAG, canvas)
        │
        ▼
 Flag::init → TWO rgba textures + linear sampler
   (split flags = two halves; single flags upload the same
    texture twice)
        │
        ▼
 per frame: wind sine envelope (pole-fixed → free-fly edge),
   horizontal foreshortening, specular sheen, seam darkening,
   per-locale personalities (GPU_FLAG_ANIMS, 12 variants)
```

- `upload_textures(bmp_a, bmp_b)` — flag halves as `ImageBitmap`s;
  `copyExternalImageToTexture` keeps it GPU→GPU.
- `set_state(hover, anim_type, split, split_x)` — `anim_type` indexes
  `GPU_FLAG_ANIMS`; `split`>0.5 draws two halves meeting at `split_x`.
- `UNIFORM_F32S = 12` — mirror table in `lib.rs` doc.
