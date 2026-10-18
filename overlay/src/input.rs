//! Zero-copy input ring — the single channel by which DOM input events
//! reach the Rust renderer.
//!
//! # Why a ring instead of function calls
//!
//! A traditional app calls a function per event (`onclick`, `onmove`).
//! Across the JS↔WASM boundary every call has marshaling cost (~1µs+
//! each, plus allocation for the argument list). At 120/240Hz with an
//! 8kHz mouse that is hundreds of crossings per frame. Instead we share
//! one flat `f32` buffer inside wasm linear memory: JS writes records
//! directly through a raw pointer, Rust reads them once per frame. The
//! only "call" is `Overlay::frame(dt)` — one boundary crossing per
//! animation tick total, matching the project's frame-budget rule that
//! per-frame work must cost ~0 on the JS side.
//!
//! # How it differs from a traditional structure
//!
//! In a normal Rust program you'd use `VecDeque` + `Mutex`; in a normal
//! JS program, an array of event objects. Neither works here: the buffer
//! must be a fixed region of `WebAssembly.Memory` so JS can wrap it in a
//! `Float32Array` without copying. The ring is a `static mut` — the old-
//! fashioned global — because there is exactly one producer (the JS
//! thread) and one consumer (the WASM thread, which is the same OS
//! thread: wasm32-unknown-unknown has no threads unless built with
//! atomics). The `static mut` warnings are silenced by construction:
//! JS only writes `meta[0]` (head) and ring slots; Rust only writes
//! `meta[1]` (tail) and reads slots. Disjoint ownership → no race.
//!
//! # Record layout — 4 × f32 = 16 bytes per record
//!
//! | index | field   | meaning                                              |
//! |-------|---------|------------------------------------------------------|
//! | [0]   | type    | 0=pointer-move 1=pointer-down 2=pointer-up 3=wheel   |
//! | [1]   | x       | CSS px, canvas space (JS sends clientX)              |
//! | [2]   | y       | CSS px, canvas space                                 |
//! | [3]   | payload | wheel deltaY when type=3, otherwise 0                |
//!
//! 16 bytes is deliberate: it is one `memory.fill`-sized stride, keeps
//! every field inside a single 64-byte cacheline per record, and lets
//! the JS side view the whole ring as ONE flat `Float32Array` with no
//! stride arithmetic beyond `index * 4`.
//!
//! # Power-of-two capacity
//!
//! `head & (CAP-1)` replaces `head % CAP`. A bitwise AND is a single
//! ALU op on every CPU; integer modulo is a division (~20-40 cycles on
//! older cores, a microcoded op on WASM's abstract machine that the JIT
//! may or may not strength-reduce). The trick only works when CAP is a
//! power of two — if you change `RING_CAP`, keep it 2^n or every index
//! computation here silently corrupts.
//!
//! # When the writer outruns the reader
//!
//! If JS bursts more than CAP records between frames (an 8kHz mouse on
//! a stalled frame), `head - tail > CAP` and the ring has wrapped —
//! `tail` positions now hold records the writer already overwrote. We
//! clamp `tail = head - CAP - 1`… see [`drain`]; the oldest records are
//! dropped wholesale because fresh pointer positions beat stale ones
//! (the "last sample wins" rule from perf-animation-frame — nobody wants
//! a hover state trailing 500ms behind the cursor).

/// f32 records per ring entry — [type, x, y, payload]. See module docs
/// for the byte layout. SAFE RANGE: keep at 4; JS's `RECORD_F32S` token
/// must match or every record after the first is misparsed.
pub const RECORD_F32S: usize = 4;

/// Ring capacity in records. Sizing math: an 8kHz mouse produces
/// ~8000 move events/s; a worst-case hitchy frame is ~130ms → ~1040
/// records. 1024 covers that burst with ~1% headroom; the deterministic
/// rng makes dropped-tail behavior identical on replay. SAFE RANGE:
/// any power of two ≥ 256. Halving to 512 only drops stale moves
/// sooner — harmless. Raising to 4096 quadruples the shared-memory
/// footprint (4096 × 16B = 64KiB) and lets the renderer fall a full
/// burst behind — do NOT raise above 4096; latency beats history here.
pub const RING_CAP: usize = 1024;

/// Total f32 slots the JS side wraps as `Float32Array`. Derived —
/// never edit directly; change RING_CAP or RECORD_F32S instead.
/// Byte footprint = RING_F32S × 4 (16KiB at current settings).
pub const RING_F32S: usize = RING_CAP * RECORD_F32S;

/// Head/tail live in a separate i32 cell (8 bytes) rather than in-band
/// ring slots so JS publishes the head with a single aligned i32 store —
/// the closest thing wasm1.0 has to an atomic write on non-atomics
/// builds. `meta[0]` = head (written only by JS), `meta[1]` = tail
/// (written only by Rust, mirrored for JS-side fullness telemetry).
/// SAFE RANGE: must stay 2; the JS meta view is `Int32Array(_, _, 2)`.
pub const RING_META_I32S: usize = 2;

/// Type discriminants written into `record[0]` by the JS bridge —
/// `INPUT_TYPES` in `core/utils/gpu/webgpu-overlay.ts` mirrors this
/// table field-for-field. If you renumber here, renumber there or every
/// event is misclassified (a wheel would read as a move).
pub mod kinds {
    /// pointermove sample. JS sends only the LAST coalesced event per
    /// native event (freshest wins) — intermediate positions are dropped
    /// upstream, so this stream is already decimated.
    pub const POINTER_MOVE: f32 = 0.0;
    /// pointerdown — opens a press session; hit_test resolves the target.
    pub const POINTER_DOWN: f32 = 1.0;
    /// pointerup — closes the press; on the same group = a click.
    /// (Mirror-contract constant — the JS bridge writes this; the Rust
    /// click-resolve consumer lands with the navigation-intent pass.)
    #[allow(dead_code)]
    pub const POINTER_UP: f32 = 2.0;
    /// wheel — `record[3]` carries deltaY already normalized to px units
    /// (the JS bridge converts deltaMode before writing).
    /// (Same mirror contract — consumed when scroll-capture lands.)
    #[allow(dead_code)]
    pub const WHEEL: f32 = 3.0;
}

/// Drains the ring from `tail` to `head`, invoking `f(kind, x, y, payload)`
/// per record in write order. `tail` is advanced so the JS writer can
/// reuse the freed slots.
///
/// # The overflow clamp, explained
///
/// `head` and `tail` are monotonically increasing cursors — they never
/// wrap (the *index* wraps via `& (CAP-1)`; the cursor itself grows
/// forever until i32 overflow at ~2^31 events, ~250 days of 8kHz input,
/// then `wrapping_sub` still computes the correct delta modulo 2^32).
///
/// `head.wrapping_sub(tail)` is the backlog. If backlog > CAP, records
/// were overwritten before we read them; we jump tail to the oldest
/// definitely-valid cursor: `head - CAP - 1`.
///
/// Why `- CAP - 1` and not `- CAP`: the writer fills slot `head & (CAP-1)`
/// NEXT, so cursor `head - CAP` maps to exactly the slot it may be
/// writing right now — reading it races the in-flight store. `head - 1`
/// is the last *completed* write, so `head - CAP - 1` maps to slot
/// `(head-1) & (CAP-1)` — completed. The trade: positions `head-CAP-1`
/// and `head-1` share that slot, so the newest record is delivered twice
/// and one genuinely-old record is skipped. One duplicated event is
/// harmless (pointer-move is idempotent; a click needs a down+up pair);
/// reading a half-written record is not. This is a deliberate
/// conservative fence, not off-by-one sloppiness.
///
/// # Performance note
///
/// The loop body is 4 array reads + a callback per record — at UI
/// scale (≤ ~1k records/frame worst case) this is microseconds. The
/// callback is `impl FnMut` so the compiler inlines the drain site in
/// `lib.rs::frame` — zero indirect calls in the hot loop.
pub fn drain(
    meta: &mut [i32; RING_META_I32S],
    ring: &[f32; RING_F32S],
    mut f: impl FnMut(f32, f32, f32, f32),
) {
    let head = meta[0] as usize;
    let mut tail = meta[1] as usize;

    // Backlog guard — see the function doc for why it's `CAP + 1` here:
    // `head - tail > CAP` means every slot was potentially overwritten;
    // resync tail to the first slot that is definitely not being written.
    if head.wrapping_sub(tail) > RING_CAP {
        tail = head.wrapping_sub(RING_CAP + 1);
    }

    while tail != head {
        // Cursor → slot index. `& (CAP-1)` is modulo-2^n; `* RECORD_F32S`
        // converts records to f32 lanes. Both are single ALU ops — the
        // cheapest possible address math for a fixed-stride buffer.
        let off = (tail & (RING_CAP - 1)) * RECORD_F32S;

        f(ring[off], ring[off + 1], ring[off + 2], ring[off + 3]);

        tail = tail.wrapping_add(1);
    }

    // Publish the drained tail so the JS writer (and any fullness check
    // on that side) sees the freed window on the very next event.
    meta[1] = tail as i32;
}
