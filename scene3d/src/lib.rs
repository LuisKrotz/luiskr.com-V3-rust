//! scene3d — the WebGPU 3D renderer replacing three.js for the
//! experiments (star-field bodies-scene, earth-playground globe).
//!
//! Status: foundation crate — `math` (vec/mat/ray) and `geometry`
//! (procedural generators) are landed; the scene graph + wgpu surface
//! + per-scene bridges follow the same gpu-mini pattern as the widget
//! crates.

pub mod camera;
pub mod geometry;
pub mod material;
pub mod math;
pub mod scene;

// The GPU half is wasm32-only — gpu-mini's web-sys WebGPU types don't
// exist on the host target, so renderer/wasm are cfg-gated; host
// `cargo test` still covers math/geometry/scene/material/camera.
#[cfg(target_arch = "wasm32")]
pub mod renderer;
#[cfg(target_arch = "wasm32")]
mod wasm;
