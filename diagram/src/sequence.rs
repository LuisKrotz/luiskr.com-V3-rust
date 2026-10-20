//! sequenceDiagram — 6 diagrams in the corpus.
//!
//! # Supported grammar
//!
//! ```text
//! sequenceDiagram
//!   participant R as Router      actor box, label after `as`
//!   R->>V: mount                 solid arrow
//!   DB-->>V: snapshot later      dashed return arrow
//!   V->>V: self-message          loop stub on the lifeline
//!   Note over A,B: text          (parsed, drawn as a band note)
//! ```
//!
//! # Layout
//!
//! Columns: participants in declaration order (first `participant`
//! wins; an undeclared `A->>B` endpoint auto-declares at that spot).
//! Rows: one slot per message — `ROW_H` apart — self-messages cost an
//! extra half-row for their loop stub. Lifelines are dashed verticals
//! under each actor box, from below the head row to the last message.
//!
//! Every vertical offset is a multiple of `ROW_H` — change it and the
//! whole diagram rescales uniformly.

use crate::model::{esc, label_lines, Theme};
use crate::svg;

/// Horizontal distance between lifeline columns, px. Must exceed the
/// longest message text or labels collide with the next lifeline —
/// the estimate is checked and widened per-message below.
const COL_GAP: f32 = 210.0;

/// Vertical pitch between message rows, px.
const ROW_H: f32 = 46.0;

/// Actor box height, px.
const ACTOR_H: f32 = 42.0;

/// Extra height a self-message loop adds, px.
const SELF_H: f32 = 26.0;

/// Page margin around the whole diagram, px.
const MARGIN: f32 = 16.0;

/// A participant (lifeline column).
struct Actor {
    /// Raw id — the `R` in `participant R as Router`.
    id: String,
    /// Display label lines (`as` text, `<br/>`-split).
    lines: Vec<String>,
    /// Column center-x after layout.
    x: f32,
    /// Box width after measuring the label.
    w: f32,
}

/// One arrow between lifelines.
struct Msg {
    /// Source column index.
    from: usize,
    /// Target column index (== from for self-messages).
    to: usize,
    /// Message text lines.
    label: Vec<String>,
    /// `->>` solid vs `-->>` dashed.
    dashed: bool,
    /// Row center-y after layout.
    y: f32,
}

/// Parses one `A->>B: text` message line.
///
/// Arrow vocabulary supported: `->>` solid, `-->>` dashed, `->`/`-->`
/// thin variants (same visuals, no arrowhead tail decoration), and
/// `<<--`/`<<-` reversed-arrow spellings (normalized to the pair's
/// direction). Anything else on the line is ignored — mermaid extras
/// like `activate`/`deactivate`/`loop` fall through harmlessly.
fn parse_msg(actors: &mut Vec<Actor>, line: &str) -> Option<Msg> {
    // Find the arrow — longest spellings first.
    let (kind, _len) = ["-->>", "->>", "-->", "->", "<<--", "<<-", "-x", "--x"]
        .iter()
        .find_map(|op| line.find(op).map(|_p| (*op, op.len())))
        .unwrap_or(("", 0));

    if kind.is_empty() {
        return None;
    }

    let pos = line.find(kind).unwrap();
    let lhs = line[..pos].trim();
    let rest = &line[pos + kind.len()..];
    // RHS: participant id up to `:` then the message text.
    let (rhs, text) = match rest.split_once(':') {
        Some((r, t)) => (r.trim(), t.trim()),
        None => (rest.trim(), ""),
    };

    if lhs.is_empty() || rhs.is_empty() {
        return None;
    }

    // Reversed spellings swap endpoints so `A<<--B` means B→A.
    let (from_name, to_name) = if kind.starts_with("<<") {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    };
    let find_or_push = |actors: &mut Vec<Actor>, name: &str| -> usize {
        if let Some(i) = actors.iter().position(|a| a.id == name) {
            i
        } else {
            actors.push(Actor {
                id: name.to_string(),
                lines: vec![name.to_string()],
                x: 0.0,
                w: 0.0,
            });
            actors.len() - 1
        }
    };

    let from = find_or_push(actors, from_name);
    let to = find_or_push(actors, to_name);
    let label = label_lines(text);
    let label = if label.is_empty() {
        vec![String::new()]
    } else {
        label
    };

    Some(Msg {
        from,
        to,
        label,
        dashed: kind.starts_with("--") || kind.starts_with("<<"),
        y: 0.0,
    })
}

/// Parses + lays out + emits a sequence diagram.
pub fn render(source: &str, theme: &Theme) -> Result<String, String> {
    let mut actors: Vec<Actor> = Vec::new();
    let mut msgs: Vec<Msg> = Vec::new();
    let mut notes: Vec<(f32, f32, Vec<String>)> = Vec::new(); // (x_center, y, lines)

    for raw in source.lines().skip(1) {
        let line = raw.trim();

        if line.is_empty() || line.starts_with("%%") {
            continue;
        }

        if let Some(rest) = line.strip_prefix("participant") {
            let spec = rest.trim();
            // `participant R as Router` — `as` splits id from label;
            // bare `participant R` labels itself.
            let (id, label) = match spec.split_once(" as ") {
                Some((id, l)) => (id.trim().to_string(), label_lines(l.trim())),
                None => (spec.to_string(), vec![spec.to_string()]),
            };

            actors.push(Actor {
                id,
                lines: if label.is_empty() {
                    vec![spec.to_string()]
                } else {
                    label
                },
                x: 0.0,
                w: 0.0,
            });
            continue;
        }

        if let Some(rest) = line.strip_prefix("Note over") {
            // `Note over A,B: text` — a band across those lifelines.
            if let Some((who, text)) = rest.split_once(':') {
                let cols: Vec<usize> = who
                    .split(',')
                    .filter_map(|n| actors.iter().position(|a| a.id == n.trim()))
                    .collect();

                if !cols.is_empty() {
                    notes.push((
                        cols.iter().map(|&c| c as f32).sum::<f32>() / cols.len() as f32,
                        msgs.len() as f32, // attach at the current row
                        label_lines(text.trim()),
                    ));
                }
            }
            continue;
        }

        // Delimiter keywords mermaid emits — consume silently so they
        // don't produce bogus messages: `end`, `loop`, `alt`, `else`,
        // `opt`, `par`, `activate`, `deactivate`, `autonumber`.
        if [
            "end", "loop", "alt", "else", "opt", "par", "critical", "break", "rect",
        ]
        .iter()
        .any(|k| line == *k || line.starts_with(&format!("{k} ")))
            || line.starts_with("activate")
            || line.starts_with("deactivate")
            || line.starts_with("autonumber")
            || line.starts_with("actor ")
        {
            continue;
        }

        if let Some(m) = parse_msg(&mut actors, line) {
            msgs.push(m);
        }
    }

    if actors.is_empty() || msgs.is_empty() {
        return Err("no participants/messages parsed".into());
    }

    // ── Column layout ───────────────────────────────────────────────
    // Column pitch = max(COL_GAP, widest message/actor + slack). A
    // uniform pitch keeps the grid legible — per-adjacent-pair spacing
    // would stretch the diagram for one long label.
    let mut widest: f32 = 0.0;

    for a in actors.iter_mut() {
        let (w, _) = svg::label_size(&a.lines, theme);

        a.w = w.max(80.0);
        widest = widest.max(a.w);
    }
    for m in &msgs {
        let (w, _) = svg::label_size(&m.label, theme);

        widest = widest.max(w);
    }

    let pitch = COL_GAP.max(widest + 60.0);

    for (i, a) in actors.iter_mut().enumerate() {
        a.x = MARGIN + pitch * (i as f32 + 0.5);
    }

    // ── Row layout ──────────────────────────────────────────────────
    // Head row at top; each message takes ROW_H, self-messages add
    // SELF_H so the loop stub has room.
    let head_y = MARGIN + ACTOR_H / 2.0;
    let mut y = MARGIN + ACTOR_H + ROW_H * 0.6;

    for m in msgs.iter_mut() {
        m.y = y;
        y += ROW_H + if m.from == m.to { SELF_H } else { 0.0 };
    }

    let bottom = y + MARGIN;
    let mut total_w = MARGIN * 2.0 + pitch * actors.len() as f32;

    // Painted horizontal bounds — a self-message label hangs off the
    // loop stub to the RIGHT of its lifeline and a `Note over` band is
    // centered on the lifeline midpoint, so both can overshoot the
    // column grid on either side. Folding them into the width keeps the
    // viewBox from clipping them.
    let mut min_x = 0.0f32;
    let stub = COL_GAP.min(80.0) * 0.5;

    for m in &msgs {
        if m.from != m.to {
            continue;
        }

        let (lw, _) = svg::label_size(&m.label, theme);
        let cx = actors[m.from].x + stub / 2.0 + 8.0;

        min_x = min_x.min(cx - lw / 2.0);
        total_w = total_w.max(cx + lw / 2.0 + MARGIN);
    }
    for (cx_col, _row, lines) in &notes {
        let (nw, _) = svg::label_size(lines, theme);
        let cx = MARGIN + pitch * (cx_col + 0.5);

        min_x = min_x.min(cx - nw / 2.0 - MARGIN);
        total_w = total_w.max(cx + nw / 2.0 + MARGIN);
    }

    let origin_x = min_x.min(0.0);

    // ── Emit ────────────────────────────────────────────────────────
    let mut out = String::with_capacity(4096 + msgs.len() * 240);

    svg::open(&mut out, origin_x, 0.0, total_w - origin_x, bottom);

    // Lifelines first (under everything): dashed verticals from below
    // the head row to the last row.
    for a in &actors {
        out.push_str(&format!(
            r##"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{bottom:.1}" stroke="{}" stroke-dasharray="4,4"/>"##,
            a.x,
            MARGIN + ACTOR_H,
            a.x,
            esc(&theme.line),
        ));
    }

    // Notes — a rounded band spanning the lifelines they reference.
    for (cx_col, row, lines) in &notes {
        let cx = MARGIN + pitch * (cx_col + 0.5);
        let ny = MARGIN + ACTOR_H + row * (ROW_H + 0.0) + ROW_H * 0.1;
        let (w, h) = svg::label_size(lines, theme);

        out.push_str(&format!(
            r##"<rect x="{:.1}" y="{ny:.1}" width="{w:.1}" height="{h:.1}" rx="4" fill="{}" stroke="{}"/>"##,
            cx - w / 2.0,
            esc(&theme.surface),
            esc(&theme.line),
        ));
        svg::centered_text(&mut out, cx, ny + h / 2.0, lines, theme);
    }

    // Messages — label above the arrow, arrow at row center.
    for m in &msgs {
        let a = &actors[m.from];
        let b = &actors[m.to];

        if m.from == m.to {
            // Self-message: stub out to the right and back — a
            // horizontal line out, drop SELF_H, back into the lifeline.
            let stub = COL_GAP.min(80.0) * 0.5;

            let back = -stub;
            let line_c = esc(&theme.line);
            let dash = if m.dashed { "5,4" } else { "0" };

            out.push_str(&format!(
                r##"<path d="M {:.1} {:.1} h {stub:.1} v {SELF_H:.1} h {back:.1} " fill="none" stroke="{line_c}" stroke-dasharray="{dash}"/>"##,
                a.x,
                m.y,
            ));
            svg::arrow_head(&mut out, a.x, m.y + SELF_H, -1.0, 0.0, theme);
            svg::centered_text(
                &mut out,
                a.x + stub / 2.0 + 8.0,
                m.y - theme.font_size * 0.5,
                &m.label,
                theme,
            );
            continue;
        }

        let (x1, x2) = (a.x, b.x);
        let dir = (x2 - x1).signum();
        // Label above the line, centered between the lifelines.
        svg::centered_text(
            &mut out,
            (x1 + x2) / 2.0,
            m.y - theme.font_size * 0.7,
            &m.label,
            theme,
        );
        svg::edge_line(
            &mut out,
            x1,
            m.y,
            x2 - dir * svg::ARROW,
            m.y,
            m.dashed,
            false,
            theme,
        );
        svg::arrow_head(&mut out, x2, m.y, dir, 0.0, theme);
    }

    // Actor boxes last — they cap the lifelines visually.
    for a in &actors {
        let x = a.x - a.w / 2.0;

        out.push_str(&format!(
            r##"<rect x="{x:.1}" y="{MARGIN:.1}" width="{:.1}" height="{ACTOR_H:.1}" rx="4" fill="{}" stroke="{}"/>"##,
            a.w,
            esc(&theme.bg),
            esc(&theme.accent),
        ));
        svg::centered_text(&mut out, a.x, head_y, &a.lines, theme);
    }

    out.push_str("</svg>");

    Ok(out)
}
