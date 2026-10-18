# gpu-mini — the shared WebGPU layer every widget links

A plain `rlib` (never built to wasm itself) with the ~30 `web-sys`
WebGPU calls every widget crate needs. This is why the artifacts are
~50 KB: the alternative — `wgpu` — statically links ~3 MB of Rust
WebGPU stack into _every_ artifact. web-sys emits thin JS bindings;
dead code is stripped per artifact at link time.

```text
 widget crate (lib.rs)                    gpu-mini (this rlib)
 ─────────────────────                    ────────────────────────
 `init_surface(canvas)`     ──async──▶    navigator.gpu → adapter →
                                          device → context →
                                          configure (premultiplied
                                          alpha, preferred format)
 `init_widget(...)`         ─────────▶    uniform bind-group layout
                                          + layout + buffer +
                                          fullscreen pipeline
 `widget_frame(...)`        ──rAF────▶    repack uniforms →
                                          write_buffer → draw →
                                          submit → stale-surface
                                          recovery
 ─── only needed by non-scaffold widgets ──────────────────────────
 textures / samplers / storage / instanced pipelines: `new_texture`,
 `upload_rgba`, `bitmap_texture`, `linear_sampler`, `new_layout`,
 `new_bind_group`, `new_buffer`, `f32_bytes`, `resize`, `configure`
```

## Stability note

The web-sys GPU bindings are upstream-unstable — `rust/.cargo/config.toml`
sets `--cfg=web_sys_unstable_apis` for the wasm32 target only. If a
web-sys bump renames an API (it happened: `GpuImageCopy*` →
`GpuTexelCopy*`), the fixes land **here once**, not in every widget
crate. That isolation is the entire point of the rlib.

## The widget scaffold

`Widget` + `init_widget` + `widget_frame` cover the 90% case — a
uniform-only widget is ~120 lines of crate code:

```rust
pub async fn init(&mut self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
    let (w, p) = gpu_mini::init_widget(
        canvas, "<name>", WGSL, UNIFORM_F32S as u32,
        self.uniforms()          // the initial f32 block
    ).await?;
    self.w = Some(w); self.pipe = Some(p);
    Ok(())
}

pub fn frame(&mut self, dt: f32) {
    self.clock.set(self.clock.get() + dt);
    if let (Some(w), Some(p)) = (&self.w, &self.pipe) {
        gpu_mini::widget_frame(w, p, self.uniforms());
    }
}
```

See `ARCHITECTURE.md` §5 for the full helper table and §6 for the
per-frame math walkthrough.
