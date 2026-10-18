//! diagram — mermaid-subset → SVG, in wasm.
//!
//! # What this replaces
//!
//! The docs viewer rendered ```mermaid fences through the `mermaid` npm
//! package (~1MB chunk, 20+ diagram families, runtime DOM hacking). The
//! corpus uses exactly two families — `flowchart` and `sequenceDiagram`
//! — so this crate implements those two, plus the handful of syntax
//! features they exercise (subgraphs, `[]`/`{}`/`()`/`>]` shapes,
//! `-->`/`-.->`/labelled edges, `participant as`, `->>`/`-->>`
//! messages, `<br/>` breaks).
//!
//! # Why SVG and not canvas
//!
//! Diagram text must stay real text — selectable, screen-reader
//! reachable, indexable. The GPU buys nothing on static diagrams;
//! mermaid's cost was parse+layout, which is CPU work — wasm's spot.
//! Returning an `<svg>` string also preserves the caller's contract:
//! `block.innerHTML = svg`, same as `mermaid.render()` produced.
//!
//! # Layout model
//!
//! - **flowchart**: longest-path layering (rank = longest path from a
//!   source), barycenter ordering inside ranks (one sweep — enough for
//!   the corpus's chain-heavy graphs), center-to-center edges with
//!   arrowheads, `<br/>` split into `<tspan>` lines.
//! - **sequence**: participants → columns in declaration order,
//!   messages → rows in order, self-messages as loop stubs.
//!
//! All dimensions derive from estimated text metrics — `svg.rs` owns
//! the width model so a font change tunes one constant.

mod flowchart;
mod model;
mod sequence;
mod svg;

use wasm_bindgen::prelude::*;

/// Renders a mermaid diagram source to an SVG string.
///
/// `source` — the raw fence content (first word selects the family).
/// `theme_json` — `{"font": "...", "fontSize": 14, "fg": "#…", "bg":
/// "#…", "line": "#…", "accent": "#…"}` — the bridge maps live CSS
/// custom properties into this so diagrams follow the active theme.
///
/// Returns `<svg …>…</svg>` or a `JsValue` error the caller shows as a
/// devWarn while restoring the source block (fail-soft contract).
#[wasm_bindgen]
pub fn render_diagram(source: &str, theme_json: &str) -> Result<String, JsValue> {
    let theme = model::Theme::from_json(theme_json);
    let head = source
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();

    match head.as_str() {
        // `graph` is mermaid's legacy alias for `flowchart` — same
        // grammar, kept so old fences keep rendering.
        "flowchart" | "graph" => flowchart::render(source, &theme)
            .map_err(|e| JsValue::from_str(&format!("flowchart: {e}"))),
        "sequencediagram" => sequence::render(source, &theme)
            .map_err(|e| JsValue::from_str(&format!("sequence: {e}"))),
        other => Err(JsValue::from_str(&format!(
            "unsupported diagram type '{other}' — the corpus uses flowchart + sequenceDiagram"
        ))),
    }
}
