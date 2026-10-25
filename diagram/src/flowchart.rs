//! flowchart / graph — the largest family in the corpus (22 diagrams).
//!
//! # Supported grammar (bounded to what the corpus writes)
//!
//! ```text
//! flowchart LR|TD            direction (LR/TB/TD/RL/BT)
//! subgraph id["label"] … end  cluster box (nestable)
//! A["rect"]  A("round")  A{"diamond"}  A[["sub"]]  A>"asym"]  A
//! A --> B   A -.-> B   A --- B   A ==> B   A -->|"lbl"| B
//! A --> B --> C           chains desugar to A→B, B→C
//! ```
//!
//! # The layout, step by step
//!
//! 1. **Rank** — longest-path layering: `rank(n) = 0` for sources,
//!    else `1 + max(rank of predecessors)`. Cyclic edges (back-edges
//!    found by DFS coloring) are excluded from ranking so a cycle can
//!    never loop the layouter.
//! 2. **Order** — inside each rank, nodes sort by the barycenter (mean
//!    position) of their predecessors. One sweep: the corpus's
//!    chain-heavy graphs converge in a single pass — crossing
//!    minimization beyond that is diminishing returns.
//! 3. **Place** — the rank axis follows `direction`: LR puts ranks on
//!    x, TD/TB on y; RL/BT negate the axis. Within a rank, nodes stack
//!    centered around the rank's midpoint.
//! 4. **Edges** — center→center, clipped to the target's rect border
//!    so arrowheads land on the outline, not inside the box.
//!
//! The math lives in `layout()` — every constant is annotated with its
//! unit and what it distorts when changed.

use crate::model::{esc, label_lines, Edge, EdgeKind, Group, Node, Shape, Theme};
use crate::svg;
use std::collections::HashMap;

/// Gap between rank columns (LR) / rows (TD), px. Raising it spreads
/// the diagram along the flow axis; too low and edge labels collide.
/// 96px keeps a labelled chip + clearance between adjacent rank boxes —
/// the 72px setting packed label chips against their endpoint borders.
const RANK_GAP: f32 = 96.0;

/// Gap between siblings inside a rank, px. Vertical breathing room in
/// LR mode; raising it makes fan-outs taller, not wider. 64px keeps
/// stacked boxes + passing edge labels visually distinct.
const SIB_GAP: f32 = 64.0;

/// Padding inside a subgraph cluster box around its children, px.
const GROUP_PAD: f32 = 22.0;

/// Outer diagram margin, px.
const MARGIN: f32 = 12.0;

/// Parsed diagram IR.
struct Graph {
    /// Direction: 'h' horizontal (LR/RL) vs 'v' (TD/TB/BT).
    horizontal: bool,
    /// +1 forward / −1 reverse along the rank axis.
    sign: f32,
    /// Insertion-ordered nodes — HashMap order is nondeterministic, so
    /// we keep ids in push order and index them for lookup.
    ids: Vec<String>,
    /// id → node.
    nodes: HashMap<String, Node>,
    /// Edges in declaration order.
    edges: Vec<Edge>,
    /// Subgraphs in declaration order.
    groups: Vec<Group>,
}

/// Shape delimiters: opener char → (closer, Shape).
fn shape_after(id_end: &str) -> Option<(Shape, usize)> {
    let b = id_end.as_bytes();

    match b.first()? {
        b'[' if b.get(1) == Some(&b'[') => Some((Shape::Sub, 2)),
        b'[' => Some((Shape::Rect, 1)),
        b'{' => Some((Shape::Diamond, 1)),
        b'(' if b.get(1) == Some(&b'(') => Some((Shape::Round, 2)),
        b'(' => Some((Shape::Round, 1)),
        b'>' => Some((Shape::Asym, 1)),
        _ => None,
    }
}

/// Closer string per shape — where the label ends.
fn shape_close(shape: Shape) -> &'static str {
    match shape {
        Shape::Rect => "]",
        Shape::Round => ")",
        Shape::Diamond => "}",
        Shape::Asym => "]",
        Shape::Sub => "]]",
    }
}

/// Parses one node spec `id[shape text]` → (id, lines, shape, consumed).
/// Bare `id` gives label=id, Shape::Rect. `consumed` is byte length of
/// the whole spec, so the edge splitter can resume after it.
fn parse_node(spec: &str) -> Option<(String, Vec<String>, Shape, usize)> {
    let s = spec.trim_start();
    let lead = spec.len() - s.len();
    // The id ends at the first shape opener or whitespace.
    let id_len = s
        .find(|c: char| matches!(c, '[' | '{' | '(' | '>' | ' ' | '\t'))
        .unwrap_or(s.len());
    let id = s[..id_len].to_string();

    if id.is_empty() {
        return None;
    }

    let rest = &s[id_len..];

    if let Some((shape, open_len)) = shape_after(rest) {
        // Labels may be "quoted" or bare — strip one quote pair.
        let body = &rest[open_len..];
        let close = shape_close(shape);
        let Some(end) = body.find(close) else {
            // Unterminated shape — treat the whole thing as a bare id.
            return Some((id.clone(), vec![id], Shape::Rect, lead + id_len));
        };

        let mut label = &body[..end];

        if label.starts_with('"') {
            label = &label[1..];
        }
        let label = label.strip_suffix('"').unwrap_or(label);
        let lines = label_lines(label);
        let lines = if lines.is_empty() {
            vec![id.clone()]
        } else {
            lines
        };

        Some((
            id,
            lines,
            shape,
            lead + id_len + open_len + end + close.len(),
        ))
    } else {
        Some((id.clone(), vec![id], Shape::Rect, lead + id_len))
    }
}

/// Edge operator at position — (kind, bytes consumed of the arrow part).
/// Checks longest first so `-.->` isn't read as `---` + `.` leftovers.
fn edge_op(s: &str) -> Option<(EdgeKind, usize)> {
    for (op, kind) in [
        ("-.->", EdgeKind::Dashed),
        ("-->", EdgeKind::Solid),
        ("==>", EdgeKind::Thick),
        ("---", EdgeKind::Open),
    ] {
        if s.starts_with(op) {
            return Some((kind, op.len()));
        }
    }

    None
}

/// Parses an edge statement line into (edge, chain). Returns the list
/// of (from_id, to_id, label, kind) pairs — chains emit consecutive
/// pairs sharing the middle node.
fn parse_edges(g: &mut Graph, line: &str, group: &str) {
    let mut rest = line;
    let mut prev_id: Option<String> = None;
    let mut pending: Option<(String, Vec<String>, EdgeKind)> = None;

    loop {
        rest = rest.trim_start();

        if rest.is_empty() {
            break;
        }

        if let Some((kind, op_len)) = edge_op(rest) {
            // After the arrow an optional |"label"| or |label| follows.
            let mut after = &rest[op_len..];
            let mut label: Vec<String> = Vec::new();

            if let Some(stripped) = after.trim_start().strip_prefix('|') {
                if let Some(end) = stripped.find('|') {
                    let mut raw = &stripped[..end];

                    if raw.starts_with('"') {
                        raw = &raw[1..];
                    }
                    let raw = raw.strip_suffix('"').unwrap_or(raw);
                    label = label_lines(raw);
                    after = &stripped[end + 1..];
                }
            }

            if let Some(src) = prev_id.take() {
                pending = Some((src, label, kind));
            }

            rest = after;
            continue;
        }

        // Otherwise this token is a node spec.
        let Some((id, lines, shape, used)) = parse_node(rest) else {
            break;
        };

        // Insert-if-absent keeps `ids` in declaration order — the entry
        // API can't help here because the closure would double-borrow g.
        if !g.nodes.contains_key(&id) {
            g.ids.push(id.clone());
            g.nodes.insert(
                id.clone(),
                Node {
                    id: id.clone(),
                    lines: lines.clone(),
                    shape,
                    group: group.to_string(),
                    x: 0.0,
                    y: 0.0,
                    w: 0.0,
                    h: 0.0,
                    rank: 0,
                },
            );
        }

        let entry = g.nodes.get_mut(&id).unwrap();

        // A shaped declaration refines a bare auto-created node
        // (`A --> B` then `B["real label"]` — keep the real shape).
        // "Bare" = Rect + the label is literally the id string.
        let bare =
            entry.shape == Shape::Rect && entry.lines.len() == 1 && entry.lines[0] == entry.id;

        if shape != Shape::Rect || bare {
            entry.lines = lines;
            entry.shape = shape;
        }

        if group.is_empty() && entry.group.is_empty() {
            // top-level stays top-level
        } else if !group.is_empty() && entry.group.is_empty() {
            entry.group = group.to_string();
        }

        if let Some((src, label, kind)) = pending.take() {
            g.edges.push(Edge {
                from: src,
                to: id.clone(),
                label,
                kind,
            });
        }

        prev_id = Some(id);
        rest = &rest[used..];
    }
}

/// Parses the whole source into a `Graph` IR.
fn parse(source: &str) -> Graph {
    let mut g = Graph {
        horizontal: true,
        sign: 1.0,
        ids: Vec::new(),
        nodes: HashMap::new(),
        edges: Vec::new(),
        groups: Vec::new(),
    };
    let mut group_stack: Vec<String> = Vec::new();

    for (i, raw) in source.lines().enumerate() {
        let line = raw.trim();

        if line.is_empty() || line.starts_with("%%") {
            continue;
        }

        if i == 0 {
            // Direction word: `flowchart LR` / `graph TD` / bare word.
            let dir = line
                .split_whitespace()
                .nth(1)
                .unwrap_or("TB")
                .to_ascii_uppercase();

            match dir.as_str() {
                "LR" => {}
                "RL" => g.sign = -1.0,
                "TD" | "TB" => g.horizontal = false,
                "BT" => {
                    g.horizontal = false;
                    g.sign = -1.0;
                }
                _ => {}
            }
            continue;
        }

        if line == "end" {
            group_stack.pop();
            continue;
        }

        if let Some(rest) = line.strip_prefix("subgraph") {
            let spec = rest.trim();
            // `subgraph id["label"]` or `subgraph "label"` — id defaults
            // to the label so `subgraph "Name"` still keys correctly.
            let (id, label) = if let Some((id, lines, _s, _u)) = parse_node(spec) {
                (id, lines)
            } else {
                (
                    spec.trim_matches('"').to_string(),
                    label_lines(spec.trim_matches('"')),
                )
            };

            g.groups.push(Group {
                id: id.clone(),
                lines: if label.is_empty() {
                    vec![id.clone()]
                } else {
                    label
                },
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
            });
            group_stack.push(id);
            continue;
        }

        // `direction XX` inside a subgraph — legal mermaid; the corpus
        // doesn't nest directions so we accept-and-ignore it.
        if line.starts_with("direction")
            || line.starts_with("classDef")
            || line.starts_with("class ")
            || line.starts_with("style ")
            || line.starts_with("linkStyle")
        {
            continue;
        }

        let group = group_stack.last().cloned().unwrap_or_default();

        parse_edges(&mut g, line, &group);
    }

    g
}

/// Longest-path ranks — cyclic edges excluded via DFS back-edge marks.
fn rank(g: &mut Graph) {
    // Build adjacency on indices for speed — the corpus is tiny (<50
    // nodes) but ranks are O(V+E) either way.
    let idx: HashMap<&str, usize> = g
        .ids
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); g.ids.len()];
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); g.ids.len()];

    for e in &g.edges {
        if let (Some(&a), Some(&b)) = (idx.get(e.from.as_str()), idx.get(e.to.as_str())) {
            if a != b {
                succ[a].push(b);
                pred[b].push(a);
            }
        }
    }

    // Iterative longest-path: repeat until no rank improves. Each pass
    // either raises at least one rank or terminates; ranks are bounded
    // by V so the loop is O(V·E) worst case — fine at corpus scale and
    // correct on cycles (back-edges converge instead of looping).
    let mut ranks = vec![0usize; g.ids.len()];

    for _ in 0..g.ids.len() {
        let mut changed = false;

        for n in 0..g.ids.len() {
            let best = pred[n].iter().map(|&p| ranks[p] + 1).max().unwrap_or(0);

            if best > ranks[n] && best <= g.ids.len() {
                ranks[n] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    for (i, id) in g.ids.iter().enumerate() {
        if let Some(n) = g.nodes.get_mut(id) {
            n.rank = ranks[i].min(g.ids.len());
        }
    }
}

/// Sizes every node from its label, then places ranks + siblings.
fn layout(g: &mut Graph, theme: &Theme) {
    let idx: HashMap<&str, usize> = g
        .ids
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); g.ids.len()];

    for e in &g.edges {
        if let (Some(&a), Some(&b)) = (idx.get(e.from.as_str()), idx.get(e.to.as_str())) {
            if a != b {
                pred[b].push(a);
            }
        }
    }

    // 1. Measure — diamonds/asym get extra box because the label must
    //    fit inside the inscribed shape (diamond half-extents shrink
    //    usable width ~40%; 1.35× restores it).
    for id in &g.ids {
        let n = g.nodes.get_mut(id).unwrap();
        let (mut w, mut h) = svg::label_size(&n.lines, theme);

        match n.shape {
            Shape::Diamond => {
                w *= 1.35;
                h *= 1.25;
            }
            Shape::Asym | Shape::Sub => w *= 1.2,
            _ => {}
        }

        n.w = w;
        n.h = h;
    }

    // 2. Bucket by rank, order each bucket by predecessor barycenter.
    let max_rank = g.nodes.values().map(|n| n.rank).max().unwrap_or(0);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); max_rank + 1];

    for (i, _id) in g.ids.iter().enumerate() {
        let r = g.nodes[&g.ids[i]].rank.min(max_rank);

        buckets[r].push(i);
    }

    // Position pass — for each rank in order, sort by mean predecessor
    // position on the cross axis, then stack centered. Runs once per
    // rank; predecessor positions are already final when we reach the
    // rank because ranks ascend.
    let mut rank_offset = 0.0f32;

    for r in 0..=max_rank {
        // Cross-axis size of the rank's widest node (gap needs it).
        let major_w: f32 = buckets[r]
            .iter()
            .map(|&i| {
                let n = &g.nodes[&g.ids[i]];

                if g.horizontal {
                    n.w
                } else {
                    n.h
                }
            })
            .fold(0.0, f32::max);

        buckets[r].sort_by(|&a, &b| {
            let bc = |i: usize| {
                if pred[i].is_empty() {
                    return 0.0f32;
                }
                pred[i]
                    .iter()
                    .map(|&p| {
                        let n = &g.nodes[&g.ids[p]];

                        if g.horizontal {
                            n.y
                        } else {
                            n.x
                        }
                    })
                    .sum::<f32>()
                    / pred[i].len() as f32
            };

            bc(a)
                .partial_cmp(&bc(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Stack the rank on the cross axis, centered at 0 for now —
        // the whole band is shifted to positive space afterward.
        let total_cross: f32 = buckets[r]
            .iter()
            .map(|&i| {
                let n = &g.nodes[&g.ids[i]];

                if g.horizontal {
                    n.h
                } else {
                    n.w
                }
            })
            .sum::<f32>()
            + SIB_GAP * buckets[r].len().saturating_sub(1) as f32;

        let mut cross = -total_cross / 2.0;

        for &i in &buckets[r] {
            let n = g.nodes.get_mut(&g.ids[i]).unwrap();
            // Along-axis center = margin + this rank's start offset +
            // half the node's own extent (nodes are center-anchored).
            // Cross-axis center = running cursor + half the extent.
            // `if … else …` is an expression — without parens `/2.0` would
            // bind to the else arm only (horizontal then doubled the
            // offset). Same for `+ SIB_GAP` below.
            let cross_c = cross + (if g.horizontal { n.h } else { n.w }) / 2.0;

            if g.horizontal {
                n.x = MARGIN + rank_offset + n.w / 2.0;
                n.y = cross_c;
            } else {
                n.x = cross_c;
                n.y = MARGIN + rank_offset + n.h / 2.0;
            }

            cross += (if g.horizontal { n.h } else { n.w }) + SIB_GAP;
        }

        rank_offset += major_w + RANK_GAP;
    }

    // 3. Shift so the cross axis starts at the margin (min → MARGIN)
    //    and RL/BT flip the rank axis around the max.
    let (min_cross, max_along) = {
        let mut mc = f32::MAX;
        let mut ma = 0.0f32;

        for n in g.nodes.values() {
            let c = if g.horizontal {
                n.y - n.h / 2.0
            } else {
                n.x - n.w / 2.0
            };
            let a = if g.horizontal {
                n.x + n.w / 2.0
            } else {
                n.y + n.h / 2.0
            };

            mc = mc.min(c);
            ma = ma.max(a);
        }

        (mc, ma)
    };
    let shift = MARGIN - min_cross;

    // Sign flip mirrors centers around the content span: the rightmost
    // edge (max_along) becomes the left margin, so p' = MARGIN+max−p.
    // Widths are symmetric around the center, so no half-extent term.
    for id in &g.ids {
        let n = g.nodes.get_mut(id).unwrap();

        if g.horizontal {
            n.y += shift;

            if g.sign < 0.0 {
                n.x = MARGIN + max_along - n.x;
            }
        } else {
            n.x += shift;

            if g.sign < 0.0 {
                n.y = MARGIN + max_along - n.y;
            }
        }
    }

    // 4. Subgraph bboxes — children extents + pad + label strip.
    for grp in g.groups.iter_mut() {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        let mut any = false;

        for n in g.nodes.values().filter(|n| n.group == grp.id) {
            any = true;
            min_x = min_x.min(n.x - n.w / 2.0);
            min_y = min_y.min(n.y - n.h / 2.0);
            max_x = max_x.max(n.x + n.w / 2.0);
            max_y = max_y.max(n.y + n.h / 2.0);
        }

        if any {
            grp.x = min_x - GROUP_PAD;
            grp.y = min_y - GROUP_PAD - svg::GROUP_LABEL_H;
            grp.w = max_x - min_x + GROUP_PAD * 2.0;
            grp.h = max_y - min_y + GROUP_PAD * 2.0 + svg::GROUP_LABEL_H;
        }
    }

    // 5. Final normalize — the group label strip can push a cluster's
    //    top edge above y=0 (children start at MARGIN but the strip
    //    adds GROUP_LABEL_H + GROUP_PAD above them). Shift ALL geometry
    //    (nodes AND groups) so the global top-left is ≥ MARGIN.
    let mut off_x = f32::MAX;
    let mut off_y = f32::MAX;

    for n in g.nodes.values() {
        off_x = off_x.min(n.x - n.w / 2.0);
        off_y = off_y.min(n.y - n.h / 2.0);
    }
    for grp in g.groups.iter().filter(|g| g.w > 0.0) {
        off_x = off_x.min(grp.x);
        off_y = off_y.min(grp.y);
    }

    let dx = MARGIN - off_x;
    let dy = MARGIN - off_y;

    if dx != 0.0 || dy != 0.0 {
        for n in g.nodes.values_mut() {
            n.x += dx;
            n.y += dy;
        }
        for grp in g.groups.iter_mut() {
            grp.x += dx;
            grp.y += dy;
        }
    }
}

/// Where a center→center ray exits a node's border — the point the
/// edge endpoint lands on. Axis-aligned rect clip: scale the ray by
/// the smaller of the half-extent ratios so it stops at the first
/// face it would cross.
fn border(n: &Node, tx: f32, ty: f32) -> (f32, f32) {
    let dx = tx - n.x;
    let dy = ty - n.y;

    if dx == 0.0 && dy == 0.0 {
        return (n.x, n.y);
    }

    let sx = if dx != 0.0 {
        (n.w / 2.0) / dx.abs()
    } else {
        f32::MAX
    };
    let sy = if dy != 0.0 {
        (n.h / 2.0) / dy.abs()
    } else {
        f32::MAX
    };
    let s = sx.min(sy);

    (n.x + dx * s, n.y + dy * s)
}

/// Clearance kept between an edge-label chip and every node box, px —
/// without it a chip may hug a border hard enough to read as part of
/// the node.
const LABEL_CLEAR: f32 = 8.0;

/// Axis-aligned overlap test — true when the two rects (half-extent
/// form) intersect within `clear` px of each other.
fn rects_touch(
    ax: f32,
    ay: f32,
    aw: f32,
    ah: f32,
    bx: f32,
    by: f32,
    bw: f32,
    bh: f32,
    clear: f32,
) -> bool {
    !(ax + aw / 2.0 + clear <= bx - bw / 2.0
        || ax - aw / 2.0 - clear >= bx + bw / 2.0
        || ay + ah / 2.0 + clear <= by - bh / 2.0
        || ay - ah / 2.0 - clear >= by + bh / 2.0)
}

/// True when a chip centered at (cx,cy) sized (w,h) clears every node
/// by LABEL_CLEAR AND every already-placed label chip. Axis-aligned
/// test — diamond/subroutine shapes keep their rect bounds
/// (conservative: the chip may sit slightly further off a diamond than
/// strictly needed, never inside it). The chip-vs-chip check matters:
/// chips are emitted in edge order and paint over earlier chips —
/// without it two labels converging on one node overlap and the first
/// renders as clipped text ("prob").
fn chip_free(g: &Graph, placed: &[(f32, f32, f32, f32)], cx: f32, cy: f32, w: f32, h: f32) -> bool {
    g.nodes.values().all(|n| {
        !rects_touch(cx, cy, w, h, n.x, n.y, n.w, n.h, LABEL_CLEAR)
    }) && placed.iter().all(|&(px0, py0, px1, py1)| {
        !rects_touch(
            cx,
            cy,
            w,
            h,
            (px0 + px1) / 2.0,
            (py0 + py1) / 2.0,
            px1 - px0,
            py1 - py0,
            LABEL_CLEAR / 2.0,
        )
    })
}

/// Where an edge's label chip goes. The midpoint is the mermaid
/// default, but skip-level edges (rank n → n+2 or longer) pass through
/// intervening nodes, and a chip dropped at the midpoint lands on top
/// of a box. The search slides the chip along the clipped edge
/// (border→border) in widening steps off the midpoint — staying ON
/// the edge so the label still reads as belonging to that edge — and
/// takes the first spot whose chip clears every node. Endpoints are
/// not exempted: a chip hugging its own source/target box fails the
/// test too, which naturally pushes labels into the open rank gap.
/// Fallback is the midpoint itself when no candidate clears (dense
/// graphs) — a worst-case label-on-box beats a dropped label.
/// `placed` carries the rects of chips already claimed this render so
/// converging labels slide instead of stacking.
fn label_pos(
    g: &Graph,
    placed: &[(f32, f32, f32, f32)],
    a: &Node,
    b: &Node,
    lw: f32,
    lh: f32,
) -> (f32, f32) {
    let (x1, y1) = border(a, b.x, b.y);
    let (x2, y2) = border(b, a.x, a.y);
    // Alternating offsets around 0.5 so the nearest free spot wins —
    // 0.5 is listed first and is also the fallback.
    const T: [f32; 11] = [0.5, 0.62, 0.38, 0.74, 0.26, 0.86, 0.14, 0.94, 0.06, 0.98, 0.02];
    // Lateral sidesteps (px, perpendicular to the edge) tried at each
    // t — converging labels (two edges → one node) crowd the same
    // midpoint, and sliding ALONG a short edge runs out of room. A
    // small perpendicular step keeps the chip beside its own edge
    // while escaping the pile; 0 first keeps on-edge placement free.
    const LAT: [f32; 7] = [0.0, 18.0, -18.0, 34.0, -34.0, 48.0, -48.0];
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt().max(0.001);
    let (px, py) = (-dy / len, dx / len);
    let mut mid = (0.0f32, 0.0f32);

    for t in T {
        for lat in LAT {
            let cx = x1 + dx * t + px * lat;
            let cy = y1 + dy * t + py * lat;

            if t == 0.5 && lat == 0.0 {
                mid = (cx, cy);
            }

            if chip_free(g, placed, cx, cy, lw, lh) {
                return (cx, cy);
            }
        }
    }

    mid
}

/// Computes every labelled edge's chip center — parallel to `g.edges`,
/// None for self-edges, unlabeled edges, and dangling references.
/// Placement order matters: labels on the SHORTEST edges go first —
/// a short edge offers few candidate spots along its clipped span, so
/// its label claims room before long-edge labels (with a whole edge
/// to roam) take a spot that starves it. `placed` accumulates claimed
/// chip rects so a later label never lands on top of an earlier one
/// (chips paint in edge order — the second would cover the first's
/// text).
fn place_labels(g: &Graph, theme: &Theme) -> Vec<Option<(f32, f32)>> {
    let mut centers: Vec<Option<(f32, f32)>> = vec![None; g.edges.len()];
    let mut placed: Vec<(f32, f32, f32, f32)> = Vec::new();
    // Edge length = clipped border→border distance; unlabeled/self/
    // missing edges get +∞ so they sort last.
    let edge_len = |i: usize| -> f32 {
        let e = &g.edges[i];

        if e.from == e.to || e.label.is_empty() {
            return f32::MAX;
        }

        let (Some(a), Some(b)) = (g.nodes.get(&e.from), g.nodes.get(&e.to)) else {
            return f32::MAX;
        };
        let (x1, y1) = border(a, b.x, b.y);
        let (x2, y2) = border(b, a.x, a.y);
        let dx = x2 - x1;
        let dy = y2 - y1;

        (dx * dx + dy * dy).sqrt()
    };
    let mut order: Vec<usize> = (0..g.edges.len()).collect();

    order.sort_by(|&i, &j| edge_len(i).partial_cmp(&edge_len(j)).unwrap_or(std::cmp::Ordering::Equal));

    for &i in &order {
        let e = &g.edges[i];

        if e.from == e.to || e.label.is_empty() {
            continue;
        }

        let (Some(a), Some(b)) = (g.nodes.get(&e.from), g.nodes.get(&e.to)) else {
            continue;
        };
        let (lw, lh) = svg::label_size(&e.label, theme);
        let (cx, cy) = label_pos(g, &placed, a, b, lw, lh);

        placed.push((cx - lw / 2.0, cy - lh / 2.0, cx + lw / 2.0, cy + lh / 2.0));
        centers[i] = Some((cx, cy));
    }

    centers
}

/// Draws a node shape centered at (n.x, n.y).
fn draw_node(out: &mut String, n: &Node, theme: &Theme) {
    let x = n.x - n.w / 2.0;
    let y = n.y - n.h / 2.0;
    let stroke = esc(&theme.line);
    let fill = esc(&theme.bg);

    match n.shape {
        Shape::Rect => out.push_str(&format!(
            r##"<rect x="{x:.1}" y="{y:.1}" width="{:.1}" height="{:.1}" fill="{fill}" stroke="{stroke}"/>"##,
            n.w, n.h,
        )),
        Shape::Round => {
            // rx = half height → stadium ends; capped at ½ width for
            // labels wider than tall.
            let rx = (n.h / 2.0).min(n.w / 4.0);

            out.push_str(&format!(
                r##"<rect x="{x:.1}" y="{y:.1}" width="{:.1}" height="{:.1}" rx="{rx:.1}" fill="{fill}" stroke="{stroke}"/>"##,
                n.w, n.h,
            ));
        }
        Shape::Diamond => {
            // Rhombus: mid-left, top-center, mid-right, bottom-center.
            out.push_str(&format!(
                r##"<polygon points="{:.1},{:.1} {:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" fill="{fill}" stroke="{stroke}"/>"##,
                n.x - n.w / 2.0, n.y,
                n.x, n.y - n.h / 2.0,
                n.x + n.w / 2.0, n.y,
                n.x, n.y + n.h / 2.0,
            ));
        }
        Shape::Asym => {
            // `>]` flag shape — pointed left edge, flat right. tip =
            // 20% of height so the point scales with the node.
            let tip = n.h * 0.2;

            out.push_str(&format!(
                r##"<polygon points="{:.1},{y:.1} {:.1},{y:.1} {:.1},{:.1} {:.1},{:.1} {x:.1},{:.1}" fill="{fill}" stroke="{stroke}"/>"##,
                x + tip,
                x + n.w,
                x + n.w, y + n.h,
                x + tip, y + n.h,
                y + n.h / 2.0,
            ));
        }
        Shape::Sub => {
            // Subroutine: outer rect + inner side rails at 10% inset.
            out.push_str(&format!(
                r##"<rect x="{x:.1}" y="{y:.1}" width="{:.1}" height="{:.1}" fill="{fill}" stroke="{stroke}"/>"##,
                n.w, n.h,
            ));

            let rail = n.w * 0.08;

            out.push_str(&format!(
                r##"<line x1="{:.1}" y1="{y:.1}" x2="{:.1}" y2="{:.1}" stroke="{stroke}"/>"##,
                x + rail, x + rail, y + n.h,
            ));
            out.push_str(&format!(
                r##"<line x1="{:.1}" y1="{y:.1}" x2="{:.1}" y2="{:.1}" stroke="{stroke}"/>"##,
                x + n.w - rail, x + n.w - rail, y + n.h,
            ));
        }
    }

    svg::centered_text(out, n.x, n.y, &n.lines, theme);
}

/// Parses + lays out + emits a flowchart diagram.
pub fn render(source: &str, theme: &Theme) -> Result<String, String> {
    let mut g = parse(source);

    if g.nodes.is_empty() {
        return Err("no nodes parsed".into());
    }

    rank(&mut g);
    layout(&mut g, theme);

    // Label chip centers — computed once (they need the clipped edge
    // endpoints AND the laid-out node grid for the collision test) so
    // the painted-bounds pass and the draw pass agree on where every
    // label lands. Parallel to `g.edges`; None for self-edges and
    // unlabeled edges.
    let label_centers = place_labels(&g, theme);

    // Painted bounds — groups may exceed node bounds on the label side;
    // edge-label chips and self-edge loops can overshoot every side, so
    // the extents fold them in too. A negative min corner becomes the
    // viewBox origin instead of clipping those elements.
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (0.0f32, 0.0f32);

    let mut grow = |x0: f32, y0: f32, x1: f32, y1: f32| {
        min_x = min_x.min(x0);
        min_y = min_y.min(y0);
        max_x = max_x.max(x1);
        max_y = max_y.max(y1);
    };

    for n in g.nodes.values() {
        grow(
            n.x - n.w / 2.0,
            n.y - n.h / 2.0,
            n.x + n.w / 2.0,
            n.y + n.h / 2.0,
        );
    }
    for grp in &g.groups {
        grow(grp.x, grp.y, grp.x + grp.w, grp.y + grp.h);
    }
    for (i, e) in g.edges.iter().enumerate() {
        let (Some(a), Some(_b)) = (g.nodes.get(&e.from), g.nodes.get(&e.to)) else {
            continue;
        };

        if e.from == e.to {
            // Self-loop extends `a.w/2 + 18` px past the node's right
            // edge plus the arrowhead's reach below the loop.
            let reach = a.x + a.w / 2.0 + a.w / 2.0 + 18.0 + svg::ARROW;

            grow(a.x, a.y - 6.0 - svg::ARROW, reach, a.y + 6.0 + svg::ARROW);

            continue;
        }

        if e.label.is_empty() {
            continue;
        }

        let Some((mx, my)) = label_centers[i] else {
            continue;
        };
        let (lw, lh) = svg::label_size(&e.label, theme);

        grow(mx - lw / 2.0, my - lh / 2.0, mx + lw / 2.0, my + lh / 2.0);
    }

    let w = max_x - min_x + MARGIN * 2.0;
    let h = max_y - min_y + MARGIN * 2.0;
    let origin_x = min_x - MARGIN;
    let origin_y = min_y - MARGIN;

    let mut out = String::with_capacity(4096 + g.edges.len() * 200);

    svg::open(&mut out, origin_x, origin_y, w, h);

    // Paint order: cluster boxes UNDER edges UNDER nodes — the lines
    // must not cover node fills, clusters must sit behind everything.
    for grp in &g.groups {
        if grp.w <= 0.0 {
            continue;
        }

        out.push_str(&format!(
            r##"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" rx="6" fill="none" stroke="{}" stroke-dasharray="2,3"/>"##,
            grp.x, grp.y, grp.w, grp.h,
            esc(&theme.line),
        ));

        svg::centered_text(
            &mut out,
            grp.x + grp.w / 2.0,
            grp.y + svg::GROUP_LABEL_H / 2.0,
            &grp.lines,
            theme,
        );
    }

    for (i, e) in g.edges.iter().enumerate() {
        let (Some(a), Some(b)) = (g.nodes.get(&e.from), g.nodes.get(&e.to)) else {
            continue;
        };

        if e.from == e.to {
            // Self-edge — a small loop off the right side: quarter-arc
            // polyline out and back so it doesn't cross the node.
            let r = a.w / 2.0 + 18.0;
            let (x0, y0) = (a.x + a.w / 2.0, a.y - 6.0);
            let (x1, y1) = (a.x + a.w / 2.0, a.y + 6.0);
            let mid_x = a.x + a.w / 2.0 + r;

            out.push_str(&format!(
                r##"<path d="M {x0:.1} {y0:.1} C {mid_x:.1} {y0:.1} {mid_x:.1} {y1:.1} {x1:.1} {y1:.1}" fill="none" stroke="{}"/>"##,
                esc(&theme.line),
            ));
            svg::arrow_head(&mut out, x1, y1, -1.0, 0.4, theme);
            continue;
        }

        let (x1, y1) = border(a, b.x, b.y);
        let (x2, y2) = border(b, a.x, a.y);

        svg::edge_line(
            &mut out,
            x1,
            y1,
            x2,
            y2,
            e.kind == EdgeKind::Dashed,
            e.kind == EdgeKind::Thick,
            theme,
        );

        if e.kind != EdgeKind::Open {
            svg::arrow_head(&mut out, x2, y2, x2 - x1, y2 - y1, theme);
        }

        if let Some((mx, my)) = label_centers[i] {
            svg::label_chip(&mut out, mx, my, &e.label, theme);
        }
    }

    for n in g.nodes.values() {
        draw_node(&mut out, n, theme);
    }

    out.push_str("</svg>");

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Neutral theme for layout tests — colors are irrelevant to the
    /// geometry under test; font size drives all box measurements.
    fn theme() -> Theme {
        Theme {
            font: "sans-serif".into(),
            font_size: 14.0,
            fg: "#fff".into(),
            bg: "#111".into(),
            line: "#ccc".into(),
            accent: "#ccc".into(),
            surface: "#000".into(),
        }
    }

    /// The docs "where compute runs" diagram — TD flowchart whose two
    /// skip-level edges (B→C over W, G→C over W+rows) used to drop
    /// their labels at midpoints inside the WASM node box.
    const SKIP_LEVEL_SRC: &str = r#"flowchart TD
    Q["animation channel"] --> A{"per-frame, per-pixel/vertex?"}
    A -->|"yes"| G["GPU — WebGPU TSL nodes / GLSL shaders"]
    A -->|"no — batch math"| B{"heavy + parallelizable?"}
    B -->|"yes"| W["WASM worker pool — wasm-pool.ts"]
    B -->|"no"| C["JS on main thread — synchronous"]

    G -.->|"probe fails / energy-saver"| C
    W -.->|"no wasm / no worker"| C
"#;

    /// Runs the real `place_labels` pass — returns (center, size,
    /// source edge) per labelled edge for collision assertions.
    fn placed_labels(g: &Graph, theme: &Theme) -> Vec<((f32, f32), (f32, f32), String)> {
        let centers = place_labels(g, theme);

        g.edges
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                let (cx, cy) = centers[i]?;
                let (lw, lh) = svg::label_size(&e.label, theme);

                Some(((cx, cy), (lw, lh), format!("{} -> {}", e.from, e.to)))
            })
            .collect()
    }

    #[test]
    fn labelled_edges_place_chips_clear_of_every_node() {
        let theme = theme();
        let mut g = parse(SKIP_LEVEL_SRC);

        rank(&mut g);
        layout(&mut g, &theme);

        let labels = placed_labels(&g, &theme);

        for ((cx, cy), (lw, lh), edge) in &labels {
            assert!(
                g.nodes.values().all(|n| !rects_touch(
                    *cx, *cy, *lw, *lh, n.x, n.y, n.w, n.h, LABEL_CLEAR
                )),
                "label on {edge} overlaps a node at ({cx:.1},{cy:.1})"
            );
        }
    }

    #[test]
    fn converging_label_chips_never_overlap() {
        // Regression: G→C and W→C both aim at C — the second chip used
        // to land on the first and paint over its text ("prob" clip).
        let theme = theme();
        let mut g = parse(SKIP_LEVEL_SRC);

        rank(&mut g);
        layout(&mut g, &theme);

        let labels = placed_labels(&g, &theme);

        for (i, ((ax, ay), (aw, ah), ea)) in labels.iter().enumerate() {
            for ((bx, by), (bw, bh), eb) in labels.iter().skip(i + 1) {
                assert!(
                    !rects_touch(*ax, *ay, *aw, *ah, *bx, *by, *bw, *bh, 0.0),
                    "label chips {ea} and {eb} overlap"
                );
            }
        }
    }

    #[test]
    fn skip_level_labels_slide_off_the_midpoint() {
        let theme = theme();
        let mut g = parse(SKIP_LEVEL_SRC);

        rank(&mut g);
        layout(&mut g, &theme);

        // B → C crosses the whole W row: the naive midpoint sits inside
        // W's rect, so the picked spot must differ from it (the exact
        // step is an implementation detail — moving is the contract).
        let e = g
            .edges
            .iter()
            .find(|e| e.from == "B" && e.to == "C")
            .expect("B -> C edge exists");
        let a = &g.nodes["B"];
        let b = &g.nodes["C"];
        let (lw, lh) = svg::label_size(&e.label, &theme);
        let (cx, cy) = label_pos(&g, &[], a, b, lw, lh);
        let (x1, y1) = border(a, b.x, b.y);
        let (x2, y2) = border(b, a.x, a.y);

        assert!(
            (cx - (x1 + x2) / 2.0).abs() > 0.5 || (cy - (y1 + y2) / 2.0).abs() > 0.5,
            "B -> C label stayed on the colliding midpoint"
        );
    }

    #[test]
    fn adjacent_rank_labels_keep_the_midpoint() {
        let theme = theme();
        let mut g = parse(SKIP_LEVEL_SRC);

        rank(&mut g);
        layout(&mut g, &theme);

        // B → W is a single-rank step — nothing between the endpoints,
        // so midpoint placement survives (minimal label drift).
        let a = &g.nodes["B"];
        let b = &g.nodes["W"];
        let e = g
            .edges
            .iter()
            .find(|e| e.from == "B" && e.to == "W")
            .expect("B -> W edge exists");
        let (lw, lh) = svg::label_size(&e.label, &theme);
        let (cx, cy) = label_pos(&g, &[], a, b, lw, lh);
        let (x1, y1) = border(a, b.x, b.y);
        let (x2, y2) = border(b, a.x, a.y);

        assert!(
            (cx - (x1 + x2) / 2.0).abs() <= 0.5 && (cy - (y1 + y2) / 2.0).abs() <= 0.5,
            "B -> W label drifted off a clear midpoint"
        );
    }

    #[test]
    fn chip_overlap_uses_the_nodes_height_not_the_chips() {
        // Regression: the "chip below node" clause compared against
        // `n.y + h/2` (the CHIP's half-height) instead of `n.y + n.h/2`,
        // so a chip overlapping the bottom half of a tall node passed
        // as free — the label painted, then the node rect painted over
        // it (info hidden). Node 40x80 at (100,100) spans y 60..140; a
        // 30x10 chip at cy=145 overlaps (top edge 140 = node bottom).
        let mut nodes = HashMap::new();

        nodes.insert(
            "N".to_string(),
            Node {
                id: "N".into(),
                lines: vec!["N".into()],
                shape: Shape::Rect,
                group: String::new(),
                x: 100.0,
                y: 100.0,
                w: 40.0,
                h: 80.0,
                rank: 0,
            },
        );
        let g = Graph {
            horizontal: false,
            sign: 1.0,
            ids: vec!["N".to_string()],
            nodes,
            edges: Vec::new(),
            groups: Vec::new(),
        };

        assert!(
            !chip_free(&g, &[], 100.0, 145.0, 30.0, 10.0),
            "chip overlapping the tall node's lower half passed as free"
        );
        assert!(
            chip_free(&g, &[], 100.0, 160.0, 30.0, 10.0),
            "chip fully below the node (top 147 > bottom 140 + clear) must be free"
        );
    }

    #[test]
    fn rank_and_sibling_gaps_hold_their_floor() {
        // The spacing contract: two ranks sit >= RANK_GAP apart and two
        // siblings in a rank >= SIB_GAP — guards the readability floor.
        let theme = theme();
        let mut g = parse("flowchart LR\nA --> B\nA --> C\n");

        rank(&mut g);
        layout(&mut g, &theme);

        let (a, b, c) = (&g.nodes["A"], &g.nodes["B"], &g.nodes["C"]);

        assert!(
            (b.x - b.w / 2.0) - (a.x + a.w / 2.0) >= RANK_GAP - 0.01,
            "rank gap shrank below {RANK_GAP}px"
        );
        assert!(
            (c.y - b.y).abs() - (b.h + c.h) / 2.0 >= SIB_GAP - 0.01,
            "sibling gap shrank below {SIB_GAP}px"
        );
    }

    #[test]
    fn node_boxes_keep_their_label_inside() {
        // Longer labels must grow the box, not overflow it: the width
        // estimate is deliberately pessimistic, so a node's inner width
        // (w - 2*PAD_X) must still cover the pessimistic text width.
        let theme = theme();
        let mut g = parse("flowchart LR\nA[\"a very long node label that must fit\"] --> B\n");

        rank(&mut g);
        layout(&mut g, &theme);

        let n = &g.nodes["A"];
        let longest = n.lines.iter().map(|l| l.chars().count()).max().unwrap_or(0) as f32;

        assert!(
            n.w - svg::PAD_X * 2.0 >= longest * theme.font_size * 0.6,
            "label estimate overflows the node box (w={:.1}, label~{:.1})",
            n.w,
            longest * theme.font_size
        );
    }
}
