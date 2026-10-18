# media — decoded image/video textured quad

**~46 KB.** Lazy artifact — mounts when a GPU-paintable media surface
appears. The `<img>`/`<video>` DOM stays the semantic + fallback layer.

```text
 mountWebGpuWidget(MEDIA, canvas)
        │
        ▼
 Media::init → one rgba texture + sampler + unit-quad pipeline
        │
        ▼
 uploads:
   upload_bitmap(ImageBitmap)  — still image, hardware-decoded
   push_frame(VideoFrame)      — WebCodecs frame, same copy path
        │
        ▼
 per frame: cover-fit textured quad (uv_rect crops letterbox)
```

## Why no `image` crate

The browser decodes JPEG/PNG/WebP off-main-thread already — a Rust
decoder would add ~800 KB AND run slower. JS hands us an
`ImageBitmap` (still) or `VideoFrame` (WebCodecs) and
`copyExternalImageToTexture` moves pixels GPU→GPU — no CPU readback
ever.

- `upload_bitmap(bmp)` — still frames; rebinds the texture.
- `push_frame(vf)` — video frames via `requestVideoFrameCallback`
  (NOT per rAF — decoder-driven).
- `set_uv_rect(x,y,w,h)` — cover-fit crop in uv space, JS-computed.
