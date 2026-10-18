//! Burn widget — the mosaic title's paper-combustion particle sim.
//!
//! Rust port of `website/components/home/mosaic/burn/` (particles.ts +
//! front.ts): embers lift off the live burn front on buoyancy, char
//! flecks tumble dark, rebuild-mode deposits rain onto the climbing
//! edge so glyphs appear to re-ink themselves.
//!
//! Unlike the shader widgets, burn is a CPU sim: `tick` advances
//! particles in dt-space and pushes quads into the shared instanced
//! pass — thousands of motes still cost one draw call. Deterministic
//! rng (xorshift) keeps motion identical at 60/120/240Hz and removes
//! the per-frame Math.random calls the JS version made.
//!
//! Letter/ink data arrives from JS once per session mount:
//!   `burn_set_letters(id, f32x6[])`  — x0,y0,x1,y1,v,delay per letter
//!   `burn_set_ink(id, w, bytes)`     — 1B/column alpha-ink occupancy
//!   `burn_set_colors(id, 4×rgba)`    — core/mid/edge/scorch tints

// Instance record emitted by `tick` — 16 f32s (4×vec4, 64 B) per
// particle, matching burn.wgsl's storage layout field-for-field:
//   [0,1]   pos.xy      CSS px (top-left origin, +y down)
//   [2,3]   size.xy     CSS px
//   [4,5]   uv.xy       reserved (0 — flat-tinted particles today)
//   [6,7]   uv_size.xy  reserved (0)
//   [8..11] tint.rgba   straight-alpha multiplier
//   [12]    z           painter's z (kept for record parity)
//   [13..15] pad        vec4 alignment
/// f32s emitted per particle — see the record doc above.
pub const QUAD_F32S: usize = 16;

// Tunables — mirror `burn/tuning.ts`; the two tables stay in lockstep.
/// Hard cap on live particles.
const P_MAX: usize = 110;
/// Spawns per second along the front while it burns.
const SPAWN_PER_S: f32 = 90.0;
/// Fraction of spawns that are dark char flecks.
const CHAR_FRACTION: f32 = 0.3;
const EMBER_LIFE_MIN: f32 = 0.35;
const EMBER_LIFE_MAX: f32 = 0.9;
const FLECK_LIFE_MIN: f32 = 0.5;
const FLECK_LIFE_MAX: f32 = 1.1;
const EMBER_VY_MIN: f32 = 26.0;
const EMBER_VY_MAX: f32 = 74.0;
const EMBER_VX: f32 = 18.0;
const FLECK_VY: f32 = 46.0;
const FLECK_VX: f32 = 22.0;
const FLECK_G: f32 = 60.0;
const DEPOSIT_VY: f32 = 18.0;
const EMBER_SIZE: f32 = 2.2;
const FLECK_SIZE: f32 = 3.4;

/// One letter's burn unit — rect in paper px + speed + ignition delay.
#[derive(Clone, Copy, Default)]
struct Letter {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    v: f32,
    delay: f32,
}

/// A live particle — the JS version stored 12 fields; same set here.
#[derive(Clone, Copy)]
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    life: f32,
    max_life: f32,
    size: f32,
    /// 0 = ember, 1 = char fleck.
    kind: u8,
    rot: f32,
    spin: f32,
    phase: f32,
    /// Owning letter index (rebuild deposits only).
    letter: u32,
}

/// Per-widget burn session — one per mosaic title hover.
pub struct BurnSim {
    /// Particle pool — retained capacity across frames.
    particles: Vec<Particle>,
    /// Letter rects/speeds pushed at mount.
    letters: Vec<Letter>,
    /// Paper-space ink occupancy — 1 byte per column.
    col_ink: Vec<u8>,
    /// Ember ramp: core → mid → edge by life fraction.
    ink_core: [f32; 4],
    ink_mid: [f32; 4],
    ink_edge: [f32; 4],
    /// Char-fleck scorch tint.
    ink_scorch: [f32; 4],
    /// Widget rect origin — quads land in canvas CSS-px space.
    origin: [f32; 2],
    /// Session clock (drives edge noise + wobble phases).
    t: f32,
    /// +1 burn, −1 rebuild.
    dir: f32,
    /// Global burn stage 0..1 — pushed per frame by the bridge.
    stage: f32,
    /// Jagged-edge amplitude in device px — pushed per session.
    jag_amp: f32,
    /// xorshift32 state — deterministic, allocation-free.
    rng: u32,
}

/// Deterministic edge noise — three detuned sines (front.ts `edgeNoise`).
fn edge_noise(x: f32, ph: f32) -> f32 {
    (x * 0.09 + ph * 2.1).sin() * 0.5
        + (x * 0.023 - ph * 1.3).sin() * 0.34
        + (x * 0.17 + ph * 3.7).sin() * 0.16
}

/// A letter's own burn progress (front.ts `letterProgress`).
fn letter_progress(l: &Letter, progress: f32) -> f32 {
    (progress * l.v - l.delay).clamp(0.0, 1.0)
}

impl BurnSim {
    /// Fresh session — deterministic seed; dir ∈ {+1, −1}.
    pub fn new(dir: f32, origin: [f32; 2], seed: u32) -> BurnSim {
        BurnSim {
            particles: Vec::with_capacity(P_MAX),
            letters: Vec::new(),
            col_ink: Vec::new(),
            ink_core: [0.0; 4],
            ink_mid: [0.0; 4],
            ink_edge: [0.0; 4],
            ink_scorch: [0.0; 4],
            origin,
            t: 0.0,
            dir,
            stage: 0.0,
            jag_amp: 0.0,
            rng: seed | 1,
        }
    }

    /// Per-frame stage push — the bridge forwards global progress +
    /// jagged amplitude before `frame`, so `tick` needs no parameters.
    pub fn set_stage(&mut self, stage: f32, jag_amp: f32) {
        self.stage = stage;
        self.jag_amp = jag_amp;
    }

    /// Live-particle count — the JS loop's settle check idles the
    /// session when the front is spent and the last mote died.
    pub fn live_count(&self) -> usize {
        self.particles.len()
    }

    /// xorshift32 → [0,1) — the JS sim called Math.random() per particle;
    /// here the stream is deterministic so replays are bit-identical.
    fn rnd(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        (self.rng as f32) / (u32::MAX as f32)
    }

    /// Moves the sim anchor — `set_widget_rect` calls this when the
    /// mirrored DOM rect moves, so a scrolling/resizing tile drags its
    /// flame front and letter field along instead of burning in place.
    pub fn set_origin(&mut self, origin: [f32; 2]) {
        self.origin = origin;
    }

    /// Direction flip (burn ↔ rebuild) — keeps letters/ink/colors, only
    /// restarts the particle stream, matching the JS session's hover-out.
    pub fn set_dir(&mut self, dir: f32) {
        self.dir = dir;
        self.particles.clear();
    }

    /// Replaces the letter table — called at session mount only.
    pub fn set_letters(&mut self, flat: &[f32]) {
        self.letters.clear();
        self.letters.extend(flat.chunks_exact(6).map(|c| Letter {
            x0: c[0],
            y0: c[1],
            x1: c[2],
            y1: c[3],
            v: c[4],
            delay: c[5],
        }));
    }

    /// Replaces the per-column ink bitmap.
    pub fn set_ink(&mut self, ink: &[u8]) {
        self.col_ink.clear();
        self.col_ink.extend_from_slice(ink);
    }

    /// Sets the four ink tints — [core, mid, edge, scorch] × rgba.
    pub fn set_colors(&mut self, flat: &[f32]) {
        let take = |i: usize, out: &mut [f32; 4]| {
            out.copy_from_slice(&flat[i * 4..i * 4 + 4]);
        };

        if flat.len() >= 16 {
            take(0, &mut self.ink_core);
            take(1, &mut self.ink_mid);
            take(2, &mut self.ink_edge);
            take(3, &mut self.ink_scorch);
        }
    }

    /// Paper-space y of the live front inside letter `l` at column `x`
    /// (front.ts `letterFrontY` — bottom→top travel + animated jag).
    /// Associated fn (no `&self`) so callers under a `&mut self.particles`
    /// borrow can still evaluate it — `t` arrives as a captured local.
    fn front_y(t: f32, l: &Letter, x: f32, progress: f32, jag_amp: f32) -> f32 {
        let p = letter_progress(l, progress);

        if p >= 1.0 {
            return -1.0;
        }

        let travel = l.y1 - l.y0 + jag_amp * 2.0;

        l.y1 + jag_amp - p * travel + edge_noise(x, t * 3.0) * jag_amp
    }

    /// Spawns along live fronts — probabilistic gating keeps the rate
    /// fractional (particles.ts `spawn`), deposits invert for rebuild.
    fn spawn(&mut self, progress: f32, jag_amp: f32, dt: f32) {
        let want = SPAWN_PER_S * dt;
        let rebuild = self.dir < 0.0;
        let n_letters = self.letters.len();

        if n_letters == 0 {
            return;
        }

        let mut i = 0.0;

        while i < want {
            if self.rnd() > want - i {
                i += 1.0;
                continue;
            }

            if self.particles.len() >= P_MAX {
                return;
            }

            let li = (self.rnd() * n_letters as f32) as usize;
            let l = self.letters[li.min(n_letters - 1)];
            let xp = (l.x0 + self.rnd() * (l.x1 - l.x0)).round() as usize;

            if xp >= self.col_ink.len() || self.col_ink[xp] == 0 {
                i += 1.0;
                continue;
            }

            let fy = Self::front_y(self.t, &l, xp as f32, progress, jag_amp);

            if fy < l.y0 - jag_amp || fy > l.y1 + jag_amp {
                i += 1.0;
                continue;
            }

            let zone_top = l.y0 - jag_amp;
            let x = self.origin[0] + xp as f32;
            let y = if rebuild {
                self.origin[1] + zone_top + self.rnd() * (fy - zone_top).max(0.0)
            } else {
                self.origin[1] + fy
            };

            let fleck = self.rnd() < CHAR_FRACTION;
            let max_life = if fleck {
                FLECK_LIFE_MIN + self.rnd() * (FLECK_LIFE_MAX - FLECK_LIFE_MIN)
            } else {
                EMBER_LIFE_MIN + self.rnd() * (EMBER_LIFE_MAX - EMBER_LIFE_MIN)
            };
            let vx = (self.rnd() * 2.0 - 1.0) * if fleck { FLECK_VX } else { EMBER_VX };
            let vy = if rebuild {
                self.rnd() * DEPOSIT_VY
            } else if fleck {
                -self.rnd() * FLECK_VY
            } else {
                -(EMBER_VY_MIN + self.rnd() * (EMBER_VY_MAX - EMBER_VY_MIN))
            };

            // Draw every rng sample before push — a `&mut self.particles`
            // borrow must not interleave with `&mut self.rng` calls.
            let size = if fleck {
                1.0 + self.rnd() * FLECK_SIZE
            } else {
                0.8 + self.rnd() * EMBER_SIZE
            };
            let rot = self.rnd() * std::f32::consts::TAU;
            let spin = (self.rnd() * 2.0 - 1.0) * 6.0;
            let phase = self.rnd() * std::f32::consts::TAU;

            self.particles.push(Particle {
                x,
                y,
                vx,
                vy,
                life: 0.0,
                max_life,
                size,
                kind: u8::from(fleck),
                rot,
                spin,
                phase,
                letter: if rebuild { li as u32 } else { u32::MAX },
            });

            i += 1.0;
        }
    }

    /// Advances + emits: embers flicker up, flecks tumble down, rebuild
    /// deposits land on the front (particles.ts `drawParticles`).
    /// Stage/jag state arrives via `set_stage` each frame. Emits into
    /// `quads` — one instanced draw for all live particles.
    pub fn tick(&mut self, dt: f32, quads: &mut Vec<f32>) {
        let stage = self.stage;
        let jag_amp = self.jag_amp;

        self.t += dt;
        self.spawn(1.0 - stage, jag_amp, dt);

        let rebuild = self.dir < 0.0;
        let t = self.t;
        let origin = self.origin;
        let mut i = self.particles.len();

        while i > 0 {
            i -= 1;

            // Landing test needs `self.letters`/`self.origin` immutable
            // reads — precompute outside the `&mut particles` window.
            let p = &mut self.particles[i];

            p.life += dt;

            if p.life >= p.max_life {
                self.particles.swap_remove(i);
                continue;
            }

            let k = p.life / p.max_life;

            if p.kind == 0 {
                // Ember — buoyancy up (burn) / gravity down (rebuild).
                p.vy += if rebuild { FLECK_G } else { -8.0 } * dt;
                p.x += p.vx * dt + (p.phase + t * 9.0).sin() * 14.0 * dt;
                p.y += p.vy * dt;

                let letter = p.letter;
                let px = p.x;
                let py = p.y;
                let landed = rebuild
                    && letter != u32::MAX
                    && py
                        >= origin[1]
                            + Self::front_y(
                                t,
                                &self.letters[letter as usize],
                                px - origin[0],
                                stage,
                                jag_amp,
                            );

                if landed {
                    self.particles.swap_remove(i);
                    continue;
                }

                let flick = 0.55 + 0.45 * (p.phase + t * 31.0).sin();
                let tint = if k < 0.35 {
                    self.ink_core
                } else if k < 0.7 {
                    self.ink_mid
                } else {
                    self.ink_edge
                };

                quads.extend_from_slice(&[
                    p.x,
                    p.y,
                    p.size,
                    p.size,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    tint[0],
                    tint[1],
                    tint[2],
                    (1.0 - k) * flick * tint[3],
                    9.0,
                    0.0,
                    0.0,
                    0.0,
                ]);
            } else {
                // Char fleck — heat lifts, gravity wins, tumbles dark.
                p.vy += FLECK_G * dt;
                p.x += p.vx * dt + (p.phase + t * 4.0).sin() * 10.0 * dt;
                p.y += p.vy * dt;
                p.rot += p.spin * dt;

                let letter = p.letter;
                let px = p.x;
                let py = p.y;
                let landed = rebuild
                    && letter != u32::MAX
                    && py
                        >= origin[1]
                            + Self::front_y(
                                t,
                                &self.letters[letter as usize],
                                px - origin[0],
                                stage,
                                jag_amp,
                            );

                if landed {
                    self.particles.swap_remove(i);
                    continue;
                }

                // Flecks render axis-aligned at this scale; rotation is
                // kept for parity with the 2D-canvas original.
                quads.extend_from_slice(&[
                    p.x - p.size * 0.5,
                    p.y - p.size * 0.35,
                    p.size,
                    p.size * 0.7,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    self.ink_scorch[0],
                    self.ink_scorch[1],
                    self.ink_scorch[2],
                    (1.0 - k) * self.ink_scorch[3],
                    8.0,
                    0.0,
                    0.0,
                    0.0,
                ]);
            }
        }
    }
}
