//! Shared IR — parsed nodes/edges before layout, plus the theme the
//! bridge passes in. Nothing here knows SVG; emitters live in `svg.rs`
//! and each family's `render` fn.

/// The palette + type scale the diagram paints with. All values come
/// from the bridge's live-CSS-token mapping — NOTHING here hardcodes a
/// color, so a theme swap needs no recompile.
#[derive(Clone, Debug)]
pub struct Theme {
    /// Font family for all text — the bridge passes the DOM's stack.
    pub font: String,
    /// Base font size in px — label/measure scale factor.
    pub font_size: f32,
    /// Foreground/ink (node labels, message text).
    pub fg: String,
    /// Node background fill.
    pub bg: String,
    /// Node border + edge/lifeline stroke.
    pub line: String,
    /// Accent — arrowheads, diamond edges, actor boxes.
    pub accent: String,
    /// Page background — used for edge-label chips so text stays
    /// readable over the lines it crosses.
    pub surface: String,
}

impl Theme {
    /// Parses the JSON theme map; any missing key falls back to a
    /// neutral value (the bridge always sends all keys — the fallbacks
    /// exist so a hand-rolled caller still gets a readable diagram).
    pub fn from_json(json: &str) -> Theme {
        let v: serde_json::Value = serde_json::from_str(json).unwrap_or(serde_json::Value::Null);

        let s = |k: &str, d: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or(d).to_string();
        let n = |k: &str, d: f32| v.get(k).and_then(|x| x.as_f64()).unwrap_or(d as f64) as f32;

        Theme {
            font: s("font", "sans-serif"),
            font_size: n("fontSize", 14.0),
            fg: s("fg", "currentColor"),
            bg: s("bg", "transparent"),
            line: s("line", "currentColor"),
            accent: s("accent", "currentColor"),
            surface: s("surface", "transparent"),
        }
    }
}

/// Node outline — which closed shape the label sits inside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// `id["…"]` / bare `id` — plain rectangle.
    Rect,
    /// `id("…")` — rounded rectangle (stadium ends).
    Round,
    /// `id{"…"}` — diamond/rhombus (decision).
    Diamond,
    /// `id>"…"]` — asymmetric flag (input/output style).
    Asym,
    /// `id[["…"]]` — subroutine (rect with side rails).
    Sub,
}

/// A parsed flowchart node.
#[derive(Clone, Debug)]
pub struct Node {
    /// Stable id from the source (`A` in `A --> B`).
    pub id: String,
    /// Display text — `<br/>` already split into lines.
    pub lines: Vec<String>,
    /// Outline shape.
    pub shape: Shape,
    /// Subgraph id this node sits inside ("" = top level).
    pub group: String,
    // ── layout output (filled by the layouter) ──
    /// Center-x in diagram px.
    pub x: f32,
    /// Center-y in diagram px.
    pub y: f32,
    /// Box width px.
    pub w: f32,
    /// Box height px.
    pub h: f32,
    /// Longest-path rank (layer index).
    pub rank: usize,
}

/// Edge style — mermaid's arrow vocabulary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EdgeKind {
    /// `-->` solid with arrowhead.
    Solid,
    /// `-.->` dashed with arrowhead.
    Dashed,
    /// `---` plain line, no head.
    Open,
    /// `==>` thick line with arrowhead.
    Thick,
}

/// A parsed flowchart edge.
#[derive(Clone, Debug)]
pub struct Edge {
    /// Source node id.
    pub from: String,
    /// Target node id.
    pub to: String,
    /// Optional `|"label"|` text (may hold `<br/>` lines).
    pub label: Vec<String>,
    /// Arrow style.
    pub kind: EdgeKind,
}

/// A subgraph (cluster box).
#[derive(Clone, Debug)]
pub struct Group {
    /// id — used for membership + dedupe.
    pub id: String,
    /// Display label lines.
    pub lines: Vec<String>,
    /// Layout output — bounding rect in diagram px.
    pub x: f32,
    /// Bounding rect.
    pub y: f32,
    /// Bounding rect.
    pub w: f32,
    /// Bounding rect.
    pub h: f32,
}

/// Splits a mermaid label on `<br/>`/`<br>` variants and trims each
/// line — the corpus writes multi-line labels with `<br/>` inside
/// `["…"]` / `{"…"}` / `participant as` text.
pub fn label_lines(raw: &str) -> Vec<String> {
    raw.replace("<br />", "<br/>")
        .replace("<br>", "<br/>")
        .split("<br/>")
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

/// Escapes text for XML content — SVG is XML; `&<>"'` must not pass
/// through raw or a label containing them produces malformed markup.
pub fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
