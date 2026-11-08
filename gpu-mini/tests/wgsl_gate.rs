//! wgsl_gate — parse + validate every widget shader exactly as the
//! production loader composes it.
//!
//! Why this gate exists: a malformed WGSL module does NOT fail a build
//! — it fails silently at runtime, where `init_widget` rejects and the
//! widget falls back to its WebGL/CSS twin. This suite compiles each
//! widget's composed source (shared gpu-mini chunks + the crate's own
//! .wgsl) through naga's front-end parser and full validator on the
//! host, so a shader regression fails `cargo test -p gpu-mini` instead
//! of degrading a production widget.
//!
//! Composition mirrors `gpu_mini::wgsl`:
//!   * sources WITHOUT their own vertex stage get `fullscreen_source`
//!     (helpers + shared vs_main), exactly like `init_widget` builds;
//!   * sources WITH `fn vs_*` keep their own vertex stage and get
//!     `helpers_source` — the same concat custom-pipeline widgets make.

use gpu_mini::wgsl;

/// One WGSL source file in the workspace and how it should be composed.
struct ShaderCase {
    /// Crate-relative name used in failure messages.
    name: &'static str,
    /// The file's own WGSL (uniforms/bindings/fragment; maybe its own
    /// vertex stage — detected per source).
    source: &'static str,
}

/// Every .wgsl file in the workspace, paired to its crate path.
const SHADERS: &[ShaderCase] = &[
    ShaderCase { name: "burger", source: include_str!("../../burger/src/burger.wgsl") },
    ShaderCase { name: "burn", source: include_str!("../../burn/src/burn.wgsl") },
    ShaderCase { name: "close-button", source: include_str!("../../close-button/src/close_button.wgsl") },
    ShaderCase { name: "corner-brackets", source: include_str!("../../corner-brackets/src/corner_brackets.wgsl") },
    ShaderCase { name: "divider-sheen", source: include_str!("../../divider-sheen/src/divider_sheen.wgsl") },
    ShaderCase { name: "flag", source: include_str!("../../flag/src/flag.wgsl") },
    ShaderCase { name: "genie", source: include_str!("../../genie/src/genie.wgsl") },
    ShaderCase { name: "glitch", source: include_str!("../../glitch/src/glitch.wgsl") },
    ShaderCase { name: "media", source: include_str!("../../media/src/media.wgsl") },
    ShaderCase { name: "menu-background", source: include_str!("../../menu-background/src/menu_background.wgsl") },
    ShaderCase { name: "ring-fill", source: include_str!("../../ring-fill/src/ring_fill.wgsl") },
    ShaderCase { name: "ring-sheen", source: include_str!("../../ring-sheen/src/ring_sheen.wgsl") },
    ShaderCase { name: "scene3d/basic", source: concat!(include_str!("../../scene3d/src/frame_uniform.wgsl"), include_str!("../../scene3d/src/shader_basic.wgsl")) },
    ShaderCase { name: "scene3d/instanced", source: concat!(include_str!("../../scene3d/src/frame_uniform.wgsl"), include_str!("../../scene3d/src/shader_instanced.wgsl")) },
    ShaderCase { name: "scene3d/points", source: concat!(include_str!("../../scene3d/src/frame_uniform.wgsl"), include_str!("../../scene3d/src/shader_points.wgsl")) },
    ShaderCase { name: "scene3d/sprite", source: concat!(include_str!("../../scene3d/src/frame_uniform.wgsl"), include_str!("../../scene3d/src/shader_sprite.wgsl")) },
    ShaderCase { name: "scene3d/standard", source: concat!(include_str!("../../scene3d/src/frame_uniform.wgsl"), include_str!("../../scene3d/src/shader_standard.wgsl")) },
    ShaderCase { name: "skeleton", source: include_str!("../../skeleton/src/skeleton.wgsl") },
    ShaderCase { name: "switch-slider", source: include_str!("../../switch-slider/src/switch_slider.wgsl") },
    ShaderCase { name: "text-draw", source: include_str!("../../text-draw/src/text_draw.wgsl") },
    ShaderCase { name: "theme-slider", source: include_str!("../../theme-slider/src/theme_slider.wgsl") },
];

/// `owns_vertex_stage` — true when the widget source defines its own
/// `vs_*` entry point (mesh, instanced-quad, 6-vert-quad widgets);
/// those get the helpers-only prelude, never the triangle vs_main.
fn owns_vertex_stage(source: &str) -> bool {
    // Any of the three known vertex entry-point names means the widget
    // ships its own vertex stage (the names are the workspace's full
    // vertex-stage vocabulary — a new one must be added here AND in
    // the composition docs).
    source.contains("fn vs_main") || source.contains("fn vs_mesh") || source.contains("fn vs_quad")
}

/// `compose` — the same concat the widget does at init: helpers-only
/// for vertex-owning sources, helpers+triangle-vs for fragment-only.
fn compose(case: &ShaderCase) -> String {
    // Vertex-owning sources prepend helpers only; fragment-only
    // sources also get the shared triangle vs_main — identical to
    // what init_widget builds at runtime.
    if owns_vertex_stage(case.source) {
        wgsl::helpers_source(case.source)
    } else {
        wgsl::fullscreen_source(case.source)
    }
}

#[test]
fn every_widget_shader_parses() {
    // Parse each composed source — a failure names the crate in the
    // panic so the broken shader is identifiable from the test log.
    for case in SHADERS {
        let source = compose(case);

        naga::front::wgsl::parse_str(&source).unwrap_or_else(|err| {
            panic!("{}: WGSL parse failure\n{}", case.name, err.emit_to_string(&source))
        });
    }
}

#[test]
fn every_widget_shader_validates() {
    // One validator with ALL flags + capabilities — the strictest
    // gate a browser could apply; reuse it across every case.
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    // Parse then validate each composed source — parse errors and
    // validation errors both panic with the crate name attached.
    for case in SHADERS {
        let source = compose(case);

        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|err| {
            panic!("{}: WGSL parse failure\n{}", case.name, err.emit_to_string(&source))
        });

        validator.validate(&module).unwrap_or_else(|err| {
            panic!("{}: WGSL validation failure\n{:?}", case.name, err)
        });
    }
}

#[test]
fn shared_chunks_are_unique_names() {
    // The helpers prelude must never collide with itself when prepended
    // — parse it alone and again doubled-side... actually the strongest
    // check: every shader above already composes with it, so a helper
    // name that a widget ALSO defines would have failed validation
    // there. This test guards the prelude's internal uniqueness.
    // Compose the prelude + an empty widget stub — a name collision
    // inside the shared chunks would fail this parse outright.
    let source = wgsl::fullscreen_source("// widget chunk\n");

    naga::front::wgsl::parse_str(&source).expect("shared prelude must parse standalone");
}
