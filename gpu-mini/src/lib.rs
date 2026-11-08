//! gpu-mini — the shared WebGPU boilerplate every widget crate links.
//!
//! # Why this exists (and why it's NOT wgpu)
//!
//! `wgpu` on wasm32 links the whole Rust WebGPU stack — naga, the hal,
//! validation layers — ~3 MB into every artifact. The widgets here
//! need maybe 30 WebGPU calls total; `web-sys` emits thin JS bindings
//! where unused functions vanish at link time. A widget crate built on
//! gpu-mini lands at tens of KB of wasm — small enough that the
//! skeleton rides the critical path and every other animation lazy-
//! loads only when its DOM surface mounts.
//!
//! # How it differs from a traditional structure
//!
//! A traditional renderer owns one context and multiplexes surfaces —
//! here EVERY widget is its own wasm module with its own GPUDevice on
//! its own canvas (the DOM architecture already gave each widget a
//! dedicated canvas; GPUDevices are cheap, and isolated contexts mean
//! a shader compile or device loss in one widget can never stall
//! another). What IS shared is this crate: context init, pipeline
//! assembly, uniform/sheet buffers, the clear→draw→submit frame —
//! written once, audited once, linked into each artifact with dead
//! code stripped.
//!
//! # The frame contract every widget implements
//!
//! ```text
//! JS bridge (rAF owner, single clock)
//!   │  Widget::frame(dt)
//!   ▼
//! rust: repack uniform f32s → write_buffer (~64 B)
//!       encoder → begin_pass → set_pipeline → set_bind_group →
//!       draw(3) → end → submit → the canvas presents the texture
//!       when the current task ends (no explicit present in WebGPU JS).
//! ```

// web_sys's WebGPU surface (GpuCanvasContext/GpuDevice/…) is
// wasm32-only, so the API lives behind `imp` and is re-exported only
// on wasm builds — while `wgsl` (pure string constants) stays
// host-visible so `cargo test` can parse-gate every widget shader.
pub mod wgsl;

#[cfg(target_arch = "wasm32")]
mod imp;

#[cfg(target_arch = "wasm32")]
pub use imp::*;
