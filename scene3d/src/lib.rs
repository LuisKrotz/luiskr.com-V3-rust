//! scene3d — the WebGPU 3D renderer replacing three.js for the
//! experiments (star-field bodies-scene, earth-playground globe).
//!
//! Status: foundation crate — `math` (vec/mat/ray) and `geometry`
//! (procedural generators) are landed; the wgpu surface + per-scene
//! bridges follow the same gpu-mini pattern as the widget crates.

pub mod math;
