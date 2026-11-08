//! nav-glyph — hover motion for the nav chip glyphs.
//!
//! Two emitters share one fixed particle pool:
//! - `KIND_STREAK` (search prompt): speed streaks fired left→right across the
//!   prompt rows, like the reference's light leaving the terminal.
//! - `KIND_SWOOSH` (preferences brush): sparks riding the paint swoosh's cubic
//!   curve, fading as they travel.
//!
//! `step(dt)` eases an activity energy toward its target (smooth wake and
//! settle, no punch), emits particles in proportion to that energy, and
//! writes `POOL` records of `STRIDE` floats — x, y, alpha, size, hue — into a
//! buffer that lives for the instance's lifetime. JS reads it through
//! `data_ptr()` / `data_len()` as a Float32Array view, so the frame loop
//! allocates nothing.

use wasm_bindgen::prelude::*;

/// Fixed particle capacity — the per-frame cost is bounded by this.
const POOL: usize = 96;
/// Floats per particle record: x, y, alpha, size, hue.
const STRIDE: usize = 5;
/// Emitter kind selectors (mirrored in core/tokens/data/nav-glyph.ts).
pub const KIND_STREAK: u8 = 0;
pub const KIND_SWOOSH: u8 = 1;
/// Frame-time clamp — a background tab resuming can't teleport particles.
const MAX_DT: f32 = 0.05;
/// Energy relaxation rate (per second) — higher wakes/settles faster.
const EASE_RATE: f32 = 6.0;
/// Streak rows in viewBox units — the prompt chevron's vertical band.
const STREAK_ROWS: [f32; 3] = [15.0, 24.0, 33.0];
/// Paint swoosh cubic in the brush's 48-unit viewBox (first arc of the stroke).
const SWOOSH: [[f32; 2]; 4] = [[4.0, 35.0], [10.0, 47.0], [31.0, 47.0], [39.0, 34.0]];

/// One pooled particle — Copy + Default so `vec![default; POOL]`
/// fills the pool without a loop.
#[derive(Clone, Copy, Default)]
struct Particle {
    /// Position x in viewBox units (streak: slides right; swoosh: set
    /// from the curve each frame).
    x: f32,
    /// Position y in viewBox units (streak: row + jitter; swoosh:
    /// curve y).
    y: f32,
    /// Velocity/step — streak: px-per-second x speed; swoosh: the
    /// curve parameter's per-second advance rate.
    vx: f32,
    /// Remaining life 0..1 — drains each frame; ≤0 frees the slot.
    life: f32,
    /// Swoosh-only cubic parameter 0..1 along the stroke.
    t: f32,
    /// Render size in viewBox units (written to the out buffer).
    size: f32,
    /// Per-particle hue seed 0..1 (JS maps it onto the theme palette).
    hue: f32,
}

/// xorshift32 — cheap, deterministic, allocation-free.
fn rand01(state: &mut u32) -> f32 {
    // Marsaglia's three-shift step — maximal period 2³²−1.
    *state ^= *state << 13;

    *state ^= *state >> 17;

    *state ^= *state << 5;

    // Keep the high 24 bits (exactly representable in f32) and scale
    // into 0..1 by 2²⁴.
    (*state >> 8) as f32 / 16_777_216.0
}

/// Cubic Bézier point at parameter `t` for the four control points `p`.
fn bezier(t: f32, p: &[[f32; 2]; 4]) -> (f32, f32) {
    // Complement of t — the basis weights run (1−t)³→t³.
    let u = 1.0 - t;

    // Bernstein basis terms for a cubic: (1−t)³, 3(1−t)²t,
    // 3(1−t)t², t³.
    let a = u * u * u;

    let b = 3.0 * u * u * t;

    let c = 3.0 * u * t * t;

    let d = t * t * t;

    // Weighted sum of the four control points — x and y components.
    (
        a * p[0][0] + b * p[1][0] + c * p[2][0] + d * p[3][0],
        a * p[0][1] + b * p[1][1] + c * p[2][1] + d * p[3][1],
    )
}

/// One glyph's particle system — construct once per icon, `step` per frame.
#[wasm_bindgen]
pub struct GlyphFx {
    /// Emitter selector — KIND_STREAK or KIND_SWOOSH, fixed at
    /// construction (one GlyphFx per icon).
    kind: u8,
    /// The fixed particle pool (POOL records — dead slots have
    /// life ≤ 0 and are reclaimed by `emit`).
    parts: Vec<Particle>,
    /// The JS-visible output buffer — POOL×STRIDE f32s, rewritten
    /// every `step` (x, y, alpha, size, hue per record).
    out: Vec<f32>,
    /// xorshift32 state for `rand01` — seeded per instance.
    rng: u32,
    /// Current activity energy 0..1 — eased toward `target`.
    energy: f32,
    /// The energy set-point — 1 while active, 0 at rest.
    target: f32,
    /// Fractional spawn accumulator — each integer part emits one
    /// particle (keeps emission proportional to energy·dt).
    spawn: f32,
    /// Live particle count after the last `step` — JS's stop signal.
    alive: usize,
}

#[wasm_bindgen]
impl GlyphFx {
    /// Creates an idle emitter of `kind` (KIND_STREAK / KIND_SWOOSH) seeded for
    /// deterministic jitter.
    #[wasm_bindgen(constructor)]
    pub fn new(kind: u8, seed: u32) -> GlyphFx {
        // Fully idle system: dead pool, zeroed buffer, energy at rest.
        // The seed is clamped ≥1 — a 0 state freezes xorshift forever.
        GlyphFx {
            kind,
            parts: vec![Particle::default(); POOL],
            out: vec![0.0; POOL * STRIDE],
            rng: seed.max(1),
            energy: 0.0,
            target: 0.0,
            spawn: 0.0,
            alive: 0,
        }
    }

    /// Sets the wake target: true eases energy up (emission on), false eases it
    /// back to rest while live particles finish their travel.
    pub fn set_active(&mut self, on: bool) {
        // Latch the energy set-point — `step` relaxes `energy` toward
        // it, so emission ramps up/dies down smoothly.
        self.target = if on { 1.0 } else { 0.0 };
    }

    /// Current activity energy in [0, 1].
    pub fn energy(&self) -> f32 {
        self.energy
    }

    /// Live particle count — JS stops the frame loop once this and energy are 0.
    pub fn alive(&self) -> usize {
        self.alive
    }

    /// Advances the simulation by `dt` seconds and rewrites the output buffer.
    pub fn step(&mut self, dt: f32) {
        // Clamp dt — a resumed background tab can't teleport particles
        // across the glyph (and dt < 0 is meaningless).
        let dt = dt.clamp(0.0, MAX_DT);

        // Copy the discriminator once — the hot loop below reads it
        // per particle, off the field not the struct.
        let kind = self.kind;

        // Exponential relax toward the target: energy += gap ×
        // (1 − e^(−dt·rate)) — frame-rate independent easing.
        self.energy += (self.target - self.energy) * (1.0 - (-dt * EASE_RATE).exp());

        // Accrue fractional spawns: rate × energy × dt — streaks emit
        // faster than swoosh sparks (46 vs 26 particles/s at full energy).
        let rate = if kind == KIND_STREAK { 46.0 } else { 26.0 };

        self.spawn += dt * rate * self.energy;

        // Emit one particle per whole accumulated unit — the
        // fractional remainder carries into the next frame.
        while self.spawn >= 1.0 {
            self.spawn -= 1.0;
            self.emit();
        }

        // Recount live particles while rewriting the output buffer.
        let mut alive = 0;
        let parts = &mut self.parts;
        let out = &mut self.out;

        for (i, p) in parts.iter_mut().enumerate() {
            // This record's slot in the f32 buffer.
            let base = i * STRIDE;

            // Dead slot → write alpha 0 (JS skips transparent
            // records) and move on.
            if p.life <= 0.0 {
                out[base + 2] = 0.0;
                continue;
            }

            // Advance the particle per its emitter kind…
            if kind == KIND_STREAK {
                // Streak: linear x travel at vx, moderate fade.
                p.x += p.vx * dt;
                p.life -= dt * 1.5;
            } else {
                // Swoosh: advance along the cubic, re-sample the
                // curve position, slower fade.
                p.t = (p.t + p.vx * dt).min(1.0);
                let (x, y) = bezier(p.t, &SWOOSH);
                p.x = x;
                p.y = y;
                p.life -= dt * 0.8;
            }

            // …then write the record: x, y, alpha (clamped life),
            // size, hue.
            alive += 1;
            out[base] = p.x;
            out[base + 1] = p.y;
            out[base + 2] = p.life.clamp(0.0, 1.0);
            out[base + 3] = p.size;
            out[base + 4] = p.hue;
        }

        // Publish the live count — JS stops stepping once this and
        // energy both read 0.
        self.alive = alive;
    }

    /// Address of the output buffer inside wasm linear memory.
    pub fn data_ptr(&self) -> *const f32 {
        // Raw pointer — JS views it as Float32Array(ptr, len); the Vec
        // never reallocates after construction so the pointer is stable.
        self.out.as_ptr()
    }

    /// Float count of the output buffer (POOL * STRIDE).
    pub fn data_len(&self) -> usize {
        // Fixed length — the view spans the whole buffer every frame.
        self.out.len()
    }
}

impl GlyphFx {
    /// Claims the first dead slot and seeds it for the active emitter.
    fn emit(&mut self) {
        // Find a free slot — a full pool drops the spawn (the cap is
        // what bounds the per-frame cost).
        let Some(slot) = self.parts.iter().position(|p| p.life <= 0.0) else {
            return;
        };

        // Two deterministic randoms: r drives row/size/hue, r2 drives
        // position jitter and speed.
        let r = rand01(&mut self.rng);

        let r2 = rand01(&mut self.rng);

        let p = &mut self.parts[slot];

        // Every particle starts at full life; the hue seeds from r.
        p.life = 1.0;
        p.hue = r;

        if self.kind == KIND_STREAK {
            // Streak: spawn at the left edge on one of the three
            // chevron rows, jittered ±1 unit vertically.
            let row = STREAK_ROWS[(r * STREAK_ROWS.len() as f32) as usize % STREAK_ROWS.len()];
            p.x = -2.0;
            p.y = row + (r2 - 0.5) * 2.0;

            // Speed 46–86 units/s — the spread makes streaks feel
            // organic rather than conveyor-belted.
            p.vx = 46.0 + r2 * 40.0;
            p.size = 5.0 + r * 6.0;
        } else {
            // Swoosh: start at the curve's head (t=0), small sparks.
            p.t = 0.0;

            // Curve-advance rate 0.7–1.3 /s — most sparks reach the
            // end of the stroke before their life drains.
            p.vx = 0.7 + r2 * 0.6;
            p.size = 0.8 + r * 1.2;
        }
    }
}
