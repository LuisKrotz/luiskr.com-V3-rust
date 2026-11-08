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

#[derive(Clone, Copy, Default)]
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    life: f32,
    t: f32,
    size: f32,
    hue: f32,
}

/// xorshift32 — cheap, deterministic, allocation-free.
fn rand01(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    (*state >> 8) as f32 / 16_777_216.0
}

/// Cubic Bézier point at parameter `t` for the four control points `p`.
fn bezier(t: f32, p: &[[f32; 2]; 4]) -> (f32, f32) {
    let u = 1.0 - t;
    let a = u * u * u;
    let b = 3.0 * u * u * t;
    let c = 3.0 * u * t * t;
    let d = t * t * t;
    (
        a * p[0][0] + b * p[1][0] + c * p[2][0] + d * p[3][0],
        a * p[0][1] + b * p[1][1] + c * p[2][1] + d * p[3][1],
    )
}

/// One glyph's particle system — construct once per icon, `step` per frame.
#[wasm_bindgen]
pub struct GlyphFx {
    kind: u8,
    parts: Vec<Particle>,
    out: Vec<f32>,
    rng: u32,
    energy: f32,
    target: f32,
    spawn: f32,
    alive: usize,
}

#[wasm_bindgen]
impl GlyphFx {
    /// Creates an idle emitter of `kind` (KIND_STREAK / KIND_SWOOSH) seeded for
    /// deterministic jitter.
    #[wasm_bindgen(constructor)]
    pub fn new(kind: u8, seed: u32) -> GlyphFx {
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
        let dt = dt.clamp(0.0, MAX_DT);
        let kind = self.kind;

        self.energy += (self.target - self.energy) * (1.0 - (-dt * EASE_RATE).exp());

        let rate = if kind == KIND_STREAK { 46.0 } else { 26.0 };
        self.spawn += dt * rate * self.energy;

        while self.spawn >= 1.0 {
            self.spawn -= 1.0;
            self.emit();
        }

        let mut alive = 0;
        let parts = &mut self.parts;
        let out = &mut self.out;

        for (i, p) in parts.iter_mut().enumerate() {
            let base = i * STRIDE;

            if p.life <= 0.0 {
                out[base + 2] = 0.0;
                continue;
            }

            if kind == KIND_STREAK {
                p.x += p.vx * dt;
                p.life -= dt * 1.5;
            } else {
                p.t = (p.t + p.vx * dt).min(1.0);
                let (x, y) = bezier(p.t, &SWOOSH);
                p.x = x;
                p.y = y;
                p.life -= dt * 0.8;
            }

            alive += 1;
            out[base] = p.x;
            out[base + 1] = p.y;
            out[base + 2] = p.life.clamp(0.0, 1.0);
            out[base + 3] = p.size;
            out[base + 4] = p.hue;
        }

        self.alive = alive;
    }

    /// Address of the output buffer inside wasm linear memory.
    pub fn data_ptr(&self) -> *const f32 {
        self.out.as_ptr()
    }

    /// Float count of the output buffer (POOL * STRIDE).
    pub fn data_len(&self) -> usize {
        self.out.len()
    }
}

impl GlyphFx {
    /// Claims the first dead slot and seeds it for the active emitter.
    fn emit(&mut self) {
        let Some(slot) = self.parts.iter().position(|p| p.life <= 0.0) else {
            return;
        };

        let r = rand01(&mut self.rng);
        let r2 = rand01(&mut self.rng);
        let p = &mut self.parts[slot];

        p.life = 1.0;
        p.hue = r;
        if self.kind == KIND_STREAK {
            let row = STREAK_ROWS[(r * STREAK_ROWS.len() as f32) as usize % STREAK_ROWS.len()];
            p.x = -2.0;
            p.y = row + (r2 - 0.5) * 2.0;
            p.vx = 46.0 + r2 * 40.0;
            p.size = 5.0 + r * 6.0;
        } else {
            p.t = 0.0;
            p.vx = 0.7 + r2 * 0.6;
            p.size = 0.8 + r * 1.2;
        }
    }
}
