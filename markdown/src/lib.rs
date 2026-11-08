//! markdown — CommonMark+GFM → HTML, replacing `marked` at build time.
//!
//! # What this replaces
//!
//! `shared/build/docs/render.mjs` converted portal markdown through the
//! `marked` npm package — ~350KB of regex-driven JavaScript invoked on
//! every docs build for every `.md` file. This crate wraps
//! pulldown-cmark (a pull-parser CommonMark engine) for the same
//! contract: source string in, HTML string out.
//!
//! # Output parity with `marked.use({ gfm: true })`
//!
//! Both engines emit the GitHub-flavored subset the docs pipeline
//! asserts on:
//!
//!   - `<h1>..<h6>` headings (the JS post-pass adds `id` anchors —
//!     GitHub's anchor scheme — the parser leaves them bare, same as
//!     marked did)
//!   - fenced code → `<pre><code class="language-{info}">` (the
//!     mermaid-div post-pass matches exactly this shape)
//!   - GFM tables → `<table><thead><th>…` / `<tbody><td>…`
//!   - strikethrough → `<del>`, tasklists → `<li><input …/></li>`
//!   - `<hr>` thematic breaks — pulldown emits `<hr />`; the JS
//!     post-pass normalizes void-element slashes for marked parity
//!   - inline links `<a href>` and images `<img src alt />` (the lazy-
//!     loading post-pass keys on `<img `)
//!   - raw HTML blocks/inline pass through verbatim (marked's default;
//!     the script-strip post-pass handles the executable surface)
//!
//! # The autolink gap
//!
//! marked's `gfm` bundles a bare-URL autolinker (`https://example.com`
//! as plain text → `<a>`). pulldown's `ENABLE_GFM` flag does NOT — it
//! enables `[!NOTE]` blockquote alerts, which marked never produced,
//! so the flag stays off (parity) and the autolink runs as an
//! event-stream transform here: `Text` events outside links/code are
//! scanned for `http(s)://` and `www.` spans, trimmed of trailing
//! punctuation per the GFM autolink rules, and re-emitted as Link
//! events. Same HTML shape marked produced: `<a href="u">u</a>`.
//!
//! # Build-time only
//!
//! The artifact loads into the Node docs pipeline via `initSync` —
//! it is never bundled into the site, so artifact size doesn't matter.

use pulldown_cmark::{html, CowStr, Event, LinkType, Options, Parser, Tag, TagEnd};
use wasm_bindgen::prelude::*;

/// `Autolink` — one link span inside a `Event::Text` run.
///
/// marked's gfm autolinker matches `https?://` and `www.` at a word
/// boundary, extends the URL over non-whitespace/non-`<>"'` chars,
/// then trims trailing punctuation (`,.;:!?*` and unbalanced `)`).
/// Spans are ordered and non-overlapping — the fold emits the literal
/// gaps between them verbatim.
struct Autolink {
    /// Start of the link span inside the run.
    url_start: usize,
    /// End of the link span (exclusive).
    url_end: usize,
}

/// `scan_autolinks` — split one text run into literal/link spans.
///
/// The scan is a left-to-right char walk, not a regex: the URL head is
/// `https?://` or `www.`; the body extends while chars are URL-legal
/// (anything but whitespace or `<>"'` — GFM's loose rule, matching
/// marked's autolinker which wraps whole non-whitespace runs), then
/// the tail trims:
///
///   - trailing `, ; : . ! ? ' " *` — the GFM "trailing punctuation"
///     class marked also strips (`example.com.` links `example.com`)
///   - unbalanced `)`/`(` pairs — the GFM rule that `](…)`-style parens
///     inside a bare URL only extend as far as they balance, so
///     `(https://x)` links `https://x`, not `https://x)`
///
/// @param text One text run (a `Event::Text` payload).
/// @returns The span list — empty when the run holds no URL.
fn scan_autolinks(text: &str) -> Vec<Autolink> {
    // Byte-level walk — every matcher below is ASCII, so raw bytes
    // are safe and avoid char-boundary overhead.
    let bytes = text.as_bytes();

    // Ordered, non-overlapping link spans found so far.
    let mut spans = Vec::new();

    // Scan cursor — advances past each consumed span (never rescans).
    let mut i = 0;

    while i < bytes.len() {
        // Head match: `http://` `https://` or `www.` — must sit at the
        // run start or after a non-word char so `xhttps://` can't cut
        // `x` off a larger word (marked applies the same boundary).
        let head = if bytes[i..].starts_with(b"https://") {
            Some(8)
        } else if bytes[i..].starts_with(b"http://") {
            Some(7)
        } else if bytes[i..].starts_with(b"www.") {
            Some(4)
        } else {
            None
        };

        // Boundary check: run start, or the previous char is non-word —
        // `xhttps://` must not link `https://` off the larger word.
        let boundary = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();

        match head {
            Some(_) if boundary => {
                // Extend over URL-legal chars — stop at whitespace and
                // the quote/bracket set GFM excludes.
                let mut end = i;
                while end < bytes.len()
                    && !bytes[end].is_ascii_whitespace()
                    && !matches!(bytes[end], b'<' | b'>' | b'"' | b'\'')
                {
                    end += 1;
                }

                // Trailing-punctuation trim — `,;:.!?'"*` can't end a
                // URL in the loose autolink form.
                while end > i
                    && matches!(
                        bytes[end - 1],
                        b',' | b';' | b':' | b'.' | b'!' | b'?' | b'*'
                    )
                {
                    end -= 1;
                }

                // Unbalanced-paren trim — walk back `)` beyond the `(`s
                // they close (GFM: parens extend only while balanced).
                while end > i && bytes[end - 1] == b')' {
                    let mut depth = 0i32;
                    for &b in &bytes[i..end] {
                        if b == b'(' {
                            depth += 1;
                        } else if b == b')' {
                            depth -= 1;
                        }
                    }
                    if depth >= 0 {
                        break;
                    }
                    end -= 1;
                }

                // Anything left after the trims is a real link span.
                if end > i {
                    spans.push(Autolink {
                        url_start: i,
                        url_end: end,
                    });
                }

                // Resume past the span — even when empty (the head
                // matched but everything trimmed), don't rescan the
                // same `i` and loop forever.
                i = end.max(i + 1);
            }
            // No head match (or head without a boundary) — step one
            // byte and keep scanning.
            _ => {
                i += 1;
            }
        }
    }

    spans
}

/// `autolink_events` — expand `Text` events into Link-wrapped spans.
///
/// A stateful fold over the event stream: the scan only applies when
/// the text isn't already inside a link (`[https://x](y)` stays one
/// link) or a code context (fenced/indented blocks + inline code pass
/// URLs through as literals — marked treats them the same).
///
/// `www.` links prepend `http://` on the href only (the visible text
/// keeps the `www.` — exactly marked's `href` behavior).
fn autolink_events<'a>(events: impl Iterator<Item = Event<'a>>) -> Vec<Event<'a>> {
    // The rewritten event stream.
    let mut out = Vec::new();

    // Context latches — track whether the cursor sits inside a link
    // or code span so their text stays literal (no autolinking).
    let mut in_link = false;
    let mut in_code = false;

    for ev in events {
        // Maintain the context latches on every Start/End tag before
        // the text handling below reads them.
        match &ev {
            Event::Start(Tag::Link { .. }) => in_link = true,
            Event::Start(Tag::CodeBlock(_)) => in_code = true,
            Event::End(TagEnd::Link) => in_link = false,
            Event::End(TagEnd::CodeBlock) => in_code = false,
            _ => {}
        }

        // Non-text events pass through untouched — only Text runs can
        // carry a bare URL.
        let Event::Text(text) = ev else {
            out.push(ev);
            continue;
        };

        // Inside a link or code context the run stays literal — pass
        // it through unchanged.
        if in_link || in_code {
            out.push(Event::Text(text));
            continue;
        }

        // Scan for URL spans…
        let spans = scan_autolinks(&text);

        // …none found → the run stays a single Text event.
        if spans.is_empty() {
            out.push(Event::Text(text));
            continue;
        }

        // Segment the run: literal → link → literal → … → the
        // trailing literal after the last span.
        let mut cursor = 0;

        for span in spans {
            // Emit the literal gap before this span (skipped when the
            // span starts right at the cursor).
            if span.url_start > cursor {
                out.push(Event::Text(CowStr::Boxed(
                    text[cursor..span.url_start].into(),
                )));
            }

            // Slice the URL; `www.` spans get `http://` on the href
            // only — the visible text keeps `www.` (marked parity).
            let url = &text[span.url_start..span.url_end];
            let href = if url.starts_with("www.") {
                CowStr::Boxed(format!("http://{url}").into_boxed_str())
            } else {
                CowStr::Boxed(url.to_string().into_boxed_str())
            };

            // Emit Link(Start) → Text(url) → End — the triple pulldown
            // itself produces for a real link, so push_html renders
            // the identical `<a href="u">u</a>` shape.
            out.push(Event::Start(Tag::Link {
                link_type: LinkType::Autolink,
                dest_url: href,
                title: "".into(),
                id: "".into(),
            }));

            out.push(Event::Text(CowStr::Boxed(url.to_string().into_boxed_str())));

            out.push(Event::End(TagEnd::Link));

            // Advance past the consumed span.
            cursor = span.url_end;
        }

        // Emit the trailing literal after the last span, if any.
        if cursor < text.len() {
            out.push(Event::Text(CowStr::Boxed(text[cursor..].into())));
        }
    }

    out
}

/// `markdown_to_html` — parse a Markdown source string and return the
/// HTML, GFM-flavored (tables, strikethrough, tasklists, bare-URL
/// autolinks — the `marked.use({ gfm: true })` bundle).
///
/// # Extensions
///
/// - `ENABLE_TABLES` — pipe tables, the `| a | b |` + `|---|---|`
///   delimiter form marked renders into `<table><thead><tbody>`.
/// - `ENABLE_STRIKETHROUGH` — `~~gone~~` → `<del>`.
/// - `ENABLE_TASKLISTS` — `- [ ] item` → `<li><input disabled
///   type="checkbox">`.
///
/// Plus the event-stream bare-URL autolinker above — marked's gfm
/// autolink parity that pulldown's `ENABLE_GFM` flag doesn't cover
/// (it gates `[!NOTE]` blockquote alerts instead, which marked never
/// produced, so that flag stays OFF for parity).
///
/// Footnotes, smart punctuation, math, superscript/subscript,
/// wikilinks, and heading attributes stay OFF — marked never enabled
/// them, so enabling them here would silently change output for
/// syntax that previously rendered as literal text.
///
/// @param source The Markdown source text.
/// @returns The rendered HTML string.
#[wasm_bindgen]
pub fn markdown_to_html(source: &str) -> String {
    // Enable exactly the three GFM extensions marked's `gfm` bundle
    // produced — every other option stays off (see doc: parity means
    // no footnotes/smart-punct/math/etc.).
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TASKLISTS);

    // Build the pull parser, then wrap it in the autolink transform —
    // pulldown emits the events, the fold rewrites bare-URL text runs.
    let parser = autolink_events(Parser::new_ext(source, opts));

    // Preallocate ~1.5× source — HTML output for prose documents runs
    // 1.2–1.5× the source length (tags + escaping); overshooting avoids
    // a mid-write reallocation, undershooting just costs one.
    let mut out = String::with_capacity(source.len() + source.len() / 2);

    // Render the (possibly rewritten) event stream into the
    // preallocated HTML buffer…
    html::push_html(&mut out, parser.into_iter());

    // …and hand the whole document back to the JS pipeline.
    out
}
