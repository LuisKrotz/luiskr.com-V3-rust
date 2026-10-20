//! SVG emission helpers + the text-width model.
//!
//! # The width estimate
//!
//! mermaid measures labels in the DOM; we can't (no DOM in wasm). The
//! model: `width = longest_line_chars × font_size × CHAR_W + padding`.
//! `CHAR_W = 0.62` is the mean advance ratio for proportional sans at
//! normal letter-spacing — ±8% per glyph, which the 1.5em horizontal
//! padding absorbs. If labels ever clip, raise `CHAR_W` before touching
//! anything else — it's the single source of horizontal truth.
//!
//! All lengths are diagram px == output px (viewBox is 1:1).

use crate::model::{esc, Theme};

/// Mean em-width of a sans glyph — see the module note.
pub const CHAR_W: f32 = 0.62;

/// Line height multiplier — 1.25em keeps `<br/>` lines readable.
pub const LINE_H: f32 = 1.25;

/// Horizontal padding each side of a label inside its shape.
pub const PAD_X: f32 = 16.0;

/// Vertical padding above/below the text block.
pub const PAD_Y: f32 = 10.0;

/// Arrowhead length along the edge direction, px.
pub const ARROW: f32 = 9.0;

/// Subgraph label strip height (title bar inside the cluster box).
pub const GROUP_LABEL_H: f32 = 26.0;

/// Measured box size for a label: (w, h) in px.
pub fn label_size(lines: &[String], theme: &Theme) -> (f32, f32) {
    let chars = lines.iter().map(|l| l.chars().count()).max().unwrap_or(1) as f32;
    let w = chars * theme.font_size * CHAR_W + PAD_X * 2.0;
    let h = lines.len().max(1) as f32 * theme.font_size * LINE_H + PAD_Y * 2.0;

    (w, h)
}

/// `<text>` block for a possibly multi-line label, centered at (cx, cy).
/// Each line is a `<tspan>` stacked `LINE_H` em apart — the same shape
/// mermaid emits, so the existing docs CSS keeps styling it.
pub fn centered_text(out: &mut String, cx: f32, cy: f32, lines: &[String], theme: &Theme) {
    let lh = theme.font_size * LINE_H;
    // First baseline: center the block — total height (n-1)·lh, so the
    // first line sits half the stack above center plus one baseline's
    // optical compensation (~0.35em cap-height centering).
    let y0 = cy - (lines.len() as f32 - 1.0) * lh / 2.0 + theme.font_size * 0.35;

    out.push_str(&format!(
        r##"<text x="{cx:.1}" y="{y0:.1}" text-anchor="middle" font-family="{}" font-size="{}" fill="{}">"##,
        esc(&theme.font),
        theme.font_size,
        esc(&theme.fg),
    ));

    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            out.push_str(&esc(line));
        } else {
            out.push_str(&format!(
                r##"<tspan x="{cx:.1}" dy="{lh:.1}">{}</tspan>"##,
                esc(line)
            ));
        }
    }

    out.push_str("</text>");
}

/// An edge line from (x1,y1) to (x2,y2) — straight segment; arrowhead
/// appended by `arrow_head` when the kind calls for one. `dashed`
/// toggles `stroke-dasharray` (the `-.->` style).
pub fn edge_line(
    out: &mut String,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    dashed: bool,
    thick: bool,
    theme: &Theme,
) {
    let dash = if dashed {
        r#" stroke-dasharray="5,4""#
    } else {
        ""
    };
    let w = if thick { 2.4 } else { 1.4 };

    out.push_str(&format!(
        r##"<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}" stroke="{}" stroke-width="{w}"{dash}/>"##,
        esc(&theme.line),
    ));
}

/// A filled arrowhead triangle at (x2,y2) pointing along (dx,dy) — the
/// tip lands ON the target's border so the head doesn't overshoot.
pub fn arrow_head(out: &mut String, x2: f32, y2: f32, dx: f32, dy: f32, theme: &Theme) {
    let len = (dx * dx + dy * dy).sqrt().max(0.001);
    // Unit direction + 90° perpendicular for the wing offset.
    let (ux, uy) = (dx / len, dy / len);
    let (px, py) = (-uy, ux);
    // Base sits ARROW px back from the tip; wings ±0.45·ARROW wide —
    // a slightly narrower-than-long head reads better at 14px text.
    let bx = x2 - ux * ARROW;
    let by = y2 - uy * ARROW;
    let wing = ARROW * 0.45;

    out.push_str(&format!(
        r##"<polygon points="{x2:.1},{y2:.1} {:.1},{:.1} {:.1},{:.1}" fill="{}"/>"##,
        bx + px * wing,
        by + py * wing,
        bx - px * wing,
        by - py * wing,
        esc(&theme.accent),
    ));
}

/// A small rounded chip behind an edge label — keeps text readable
/// where the line passes beneath it (the surface color masks the line).
pub fn label_chip(out: &mut String, cx: f32, cy: f32, lines: &[String], theme: &Theme) {
    let (w, h) = label_size(lines, theme);

    out.push_str(&format!(
        r##"<rect x="{:.1}" y="{:.1}" width="{w:.1}" height="{h:.1}" rx="4" fill="{}"/>"##,
        cx - w / 2.0,
        cy - h / 2.0,
        esc(&theme.surface),
    ));

    centered_text(out, cx, cy, lines, theme);
}

/// Opens the `<svg>` element with a 1:1 viewBox that may start at a
/// negative origin — callers pass the true painted bounds (min corner +
/// extents) so edge labels, self-loops and note bands that overshoot
/// the node grid are not clipped by the viewBox.
pub fn open(out: &mut String, min_x: f32, min_y: f32, w: f32, h: f32) {
    out.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{min_x:.0} {min_y:.0} {w:.0} {h:.0}" width="{w:.0}" role="img">"##,
    ));
}
