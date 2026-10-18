# diagram — the local mermaid replacement

~130KB wasm instead of ~1MB of mermaid JS. Parses, lays out, and emits
**SVG** for the two diagram families the docs corpus uses:

| Family | Count in corpus | Status |
| ------ | --------------- | ------ |
| `flowchart` / `graph` (LR/RL/TD/TB/RL, subgraphs, `[]`/`{}`/`()`/`[[]]`/`>]` shapes, `-->`/`-.->`/`---`/`==>`, `\|"label"\|` edges, chains) | 22 | full |
| `sequenceDiagram` (`participant … as`, `->>`/`-->>`/reversed arrows, self-messages, `Note over`) | 6 | full |
| everything else (mindmap, gantt, erDiagram…) | 0 | throws → caller restores source |

```text
 ```mermaid fence (docs markdown)
        │  escaped source in a .docs-mermaid div
        ▼
 experiments/docs/mermaid.ts
        │  themeJson(getComputedStyle(root)) — live CSS tokens
        ▼
 core/utils/wasm/wasm-diagram.ts        lazy, cached, unlatches on
        │  import('/wasm/diagram/…')    failure; jest-mockable seam
        ▼
 render_diagram(source, theme) → <svg>   (rust/diagram, wasm32)
        │
        ▼
 block.innerHTML = svg — real text, selectable, a11y, indexable
```

## Why SVG and not WebGPU

Diagram text must stay real text — selectable, screen-reader reachable,
indexable (the "all content readable" directive). The expensive part of
mermaid was parse + layout (CPU) — wasm's sweet spot. Rasterizing a
static diagram to canvas buys nothing and costs accessibility.

## Layout model

- **flowchart** — longest-path layering (`rank(n) = 1 + max(pred)`),
  back-edges excluded so cycles converge; barycenter ordering inside
  ranks (one sweep — chain-heavy graphs converge immediately); rank
  axis follows `direction`; edges clip to the target border so
  arrowheads land on the outline. Groups emit last-pass bounding
  boxes with a label strip; a final normalize pass shifts all geometry
  positive.
- **sequence** — participants → columns in declaration order
  (undeclared endpoints auto-declare); one row per message; self-
  messages draw a stub loop; lifelines are dashed verticals; actor
  boxes cap the columns.

## Text metrics

No DOM in wasm → widths are estimated: `chars × fontSize × 0.62` (mean
sans advance) + padding. If a label ever clips, raise `CHAR_W` in
`src/svg.rs` — it's the single source of horizontal truth. All tuning
constants are annotated at declaration with units + effects.

## Theme contract

`render_diagram(source, themeJson)` — theme JSON is
`{font, fontSize, fg, bg, line, accent, surface}`, mapped from live
CSS custom properties by `experiments/docs/mermaid.ts` so diagrams
follow the active palette (light/dark + accent). Every missing key
falls back to a neutral value — never a hardcoded project color.

## Build

```sh
cd rust/diagram
wasm-pack build --target web --release
cp pkg/luiskr_diagram* ../../website/public/wasm/diagram/
```

## Fail-soft contract

Any parse/layout error or unsupported family throws a `JsValue` string
— `mermaid.ts` catches it and restores the escaped source, so a bad
diagram reads as documentation instead of breaking the page.
