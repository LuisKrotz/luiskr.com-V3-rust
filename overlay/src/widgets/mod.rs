//! Widget domains — every animation surface that used to be an isolated
//! WebGL canvas becomes a rectangular region of the single overlay
//! surface, painted by a Rust-side state machine + a WGSL shader ported
//! from the widget's original GLSL.
//!
//! Two render families:
//!
//! * **SDF/procedural** — burger, flag, switch/theme sliders, close
//!   button, menu background, skeleton. Each owns a fragment pipeline
//!   and a slice of the uniform arena; `frame` issues one scissored
//!   draw per live widget. All math stays in the shader — the Rust side
//!   only forwards time + state uniforms, so the hot path is pure GPU.
//! * **Simulated** — burn embers/flecks. Rust advances the particle sim
//!   (dt-space, deterministic) and emits instances into the shared
//!   quad pass, so thousands of motes still cost one draw call.
//!
//! Adding a surface = a `kind` variant, a state struct, a `.wgsl`,
//! and a match arm in [`WIDGET_SPECS`]. The JS bridge mirrors
//! `WIDGET_KINDS` in `core/tokens/motion/gpu-widgets.ts`.

pub mod burger;
pub mod burn;
pub mod close_button;
pub mod flag;
pub mod menu_background;
pub mod skeleton;
pub mod switch_slider;
pub mod theme_slider;

/// Widget discriminants — mirrored by `GPU_WIDGET_KINDS` in
/// `core/tokens/motion/gpu-widgets.ts`; bump both together.
pub mod kinds {
    /// Nav burger hairline bars (fluid wave + breathing).
    pub const BURGER: u32 = 1;
    /// Locale flag — wind warp + specular + split-flag transition.
    pub const FLAG: u32 = 2;
    /// Preference toggle — pill track + themed context waveform.
    pub const SWITCH_SLIDER: u32 = 3;
    /// Day/system/night slider — full procedural sky scene.
    pub const THEME_SLIDER: u32 = 4;
    /// Modal close — liquid fill, foam crest, rotating X strokes.
    pub const CLOSE_BUTTON: u32 = 5;
    /// Fullscreen menu backdrop — domain-warped fbm contour field.
    pub const MENU_BACKGROUND: u32 = 6;
    /// Loading skeleton — rounded cells with morphing 0/1 glyphs.
    pub const SKELETON: u32 = 7;
    /// Mosaic letter-burn — ember/char particle sim (quad emitters).
    pub const BURN: u32 = 8;
}

/// Static description of one widget domain: which WGSL to compile and
/// how many f32 slots its uniform block occupies (16-byte aligned).
pub struct WidgetSpec {
    /// The WGSL source compiled into this widget's pipeline.
    pub wgsl: &'static str,
    /// Uniform f32 count per widget instance (padded to 4).
    pub uniform_f32s: usize,
}

/// Kind → spec lookup. Single source for pipeline creation — the frame
/// loop never branches on kind, only on this table.
pub fn spec_of(kind: u32) -> Option<WidgetSpec> {
    Some(match kind {
        kinds::BURGER => WidgetSpec {
            wgsl: burger::WGSL,
            uniform_f32s: burger::UNIFORM_F32S,
        },
        kinds::FLAG => WidgetSpec {
            wgsl: flag::WGSL,
            uniform_f32s: flag::UNIFORM_F32S,
        },
        kinds::SWITCH_SLIDER => WidgetSpec {
            wgsl: switch_slider::WGSL,
            uniform_f32s: switch_slider::UNIFORM_F32S,
        },
        kinds::THEME_SLIDER => WidgetSpec {
            wgsl: theme_slider::WGSL,
            uniform_f32s: theme_slider::UNIFORM_F32S,
        },
        kinds::CLOSE_BUTTON => WidgetSpec {
            wgsl: close_button::WGSL,
            uniform_f32s: close_button::UNIFORM_F32S,
        },
        kinds::MENU_BACKGROUND => WidgetSpec {
            wgsl: menu_background::WGSL,
            uniform_f32s: menu_background::UNIFORM_F32S,
        },
        kinds::SKELETON => WidgetSpec {
            wgsl: skeleton::WGSL,
            uniform_f32s: skeleton::UNIFORM_F32S,
        },
        _ => return None, // BURN renders through the quad pass, not a pipeline.
    })
}
