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
const FLECK_VY: f32 = 46.0;
const FLECK_VX: f32 = 22.0;
const FLECK_G: f32 = 60.0;
const DEPOSIT_VY: f32 = 18.0;
const FLECK_SIZE: f32 = 3.4;

// ── Real-fire ember model ───────────────────────────────────────────
// Embers are ejected incandescent (~2000 K) and obey a thermal cycle
// instead of a fixed life-ramp: Newtonian convective+radiative cooling
// toward ambient, buoyant lift proportional to their excess heat
// (a mote only rises while it's hot), quadratic drag on velocity, and
// blackbody color — the ink ramp maps to TEMPERATURE (core = white-hot,
// mid = yellow-orange, edge = deep red, scorch = dying char) with a
// T^2.2 luminosity falloff (sRGB-compressed Stefan–Boltzmann). A mote
// dies when it drops below visible incandescence (~640 K) — fuel-
// starved sparks wink out rather than fading on a timer.
/// Ambient temperature the ember field cools toward (K).
const T_AMBIENT: f32 = 340.0;
/// Ejecta temperature bounds at the front — paper flames run
/// 1800–2300 K at the reaction zone.
pub(crate) const T_EJECT_MIN: f32 = 1850.0;
const T_EJECT_MAX: f32 = 2350.0;
/// Ember Newtonian cooling rate (/s) — mm-scale char loses heat to
/// convection+radiation in roughly a second.
const EMBER_COOL: f32 = 2.4;
/// Fleck cooling — thin char scraps shed heat faster than embers.
const FLECK_COOL: f32 = 3.6;
/// Temperature below which a mote no longer incandesces (~dark char).
const T_EXTINCT: f32 = 640.0;
/// Hot-ember launch speeds — the convective burst ejects sparks.
const EMBER_VY_MIN: f32 = 30.0;
const EMBER_VY_MAX: f32 = 82.0;
const EMBER_VX: f32 = 16.0;
/// Peak buoyant acceleration for a fully-hot ember (px/s² upward) —
/// scales with excess heat (T−T_amb)/(T_eject−T_amb) as it cools.
const EMBER_BUOY: f32 = 120.0;
/// Gravity on a cooled mote — dead sparks fall back as ash (px/s²).
const EMBER_ASH_G: f32 = 42.0;
/// Quadratic drag coefficient (/px·s scaled) — small motes are drag-
/// dominated: lateral launch velocity damps out in ~0.3 s.
const EMBER_DRAG: f32 = 0.045;
/// Ember spark diameter bound (device px).
const EMBER_SIZE: f32 = 2.2;

// Flame licks — the visible flame sheet riding the burn front. Real
// turbulent-diffusion flames read as a band of short-lived bright
// tongues, not a static edge: each lick is born at the front, races
// up on its own buoyancy, necks and dissipates in a few tenths of a
// second. Rendered through the soft-eroded puff path with a hot core.
/// Lick spawn rate along the live front (/s).
const FLAME_PER_S: f32 = 55.0;
const FLAME_LIFE_MIN: f32 = 0.14;
const FLAME_LIFE_MAX: f32 = 0.34;
/// Lick launch speed — flame gas rises fast near the reaction zone.
const FLAME_VY_MIN: f32 = 95.0;
const FLAME_VY_MAX: f32 = 190.0;
const FLAME_VX: f32 = 14.0;
/// Lick buoyant acceleration — flame gas is near-front hot, strong lift.
const FLAME_BUOY: f32 = 260.0;
/// Lick quad size bounds at birth (device px) — elongated tongues.
const FLAME_SIZE_MIN: f32 = 5.0;
const FLAME_SIZE_MAX: f32 = 11.0;
/// Peak lick opacity — the flame band must read through smoke.
const FLAME_PEAK_A: f32 = 0.6;
/// Hard cap on live licks.
const FLAME_MAX: usize = 90;

// Smoke — the ash column rising off the burn front. Real combustion
// smoke: buoyant while hot (fast initial rise), decelerates toward a
// slow terminal drift as the plume cools and mixes, expands roughly
// linearly as it entrains air, wanders on low-frequency turbulence,
// and lives several times longer than the embers it follows.
/// Smoke puffs per second along the live front — far sparser than
/// embers; each puff is large and long-lived.
const SMOKE_PER_S: f32 = 12.0;
const SMOKE_LIFE_MIN: f32 = 1.8;
const SMOKE_LIFE_MAX: f32 = 3.2;
/// Initial buoyant velocity (upward, px/s) at the hot front.
const SMOKE_VY_MIN: f32 = 50.0;
const SMOKE_VY_MAX: f32 = 90.0;
/// Terminal drift once the plume cools — buoyancy decays toward this.
const SMOKE_VY_TERMINAL: f32 = -18.0;
/// Initial lateral spread velocity, px/s.
const SMOKE_VX: f32 = 12.0;
/// Mean lateral shear — a gentle sideways drift like still-air plumes.
const SMOKE_DRIFT_X: f32 = 6.0;
/// Turbulent wander amplitude (px/s equivalent) — lower-frequency,
/// larger-amplitude than ember wobble: smoke billows, embers flicker.
const SMOKE_WANDER: f32 = 28.0;
/// Puff diameter at spawn, device px.
const SMOKE_SIZE0: f32 = 6.0;
/// Entrainment coefficient — self-similar turbulent plumes grow
/// linearly with height above the source (dr/dz ≈ 0.1 measured →
/// diameter ≈ 0.22 per unit rise). Growth tracks RISE, not time.
pub(crate) const SMOKE_ENTRAIN: f32 = 0.22;
/// Plume reference height (px) for the centerline decay — a pure
/// buoyant plume's axial velocity falls ≈ z^(−1/3) as momentum
/// diffuses; below this height the source's momentum dominates.
const SMOKE_Z0: f32 = 55.0;
/// Slow spin the shader's noise erosion rotates at, rad/s.
const SMOKE_SPIN: f32 = 0.4;
/// Peak puff opacity — smoke stays translucent so the glyphs and
/// embers read through it.
const SMOKE_PEAK_A: f32 = 0.32;
/// Hard cap on live smoke puffs (separate pool from sparks).
const SMOKE_MAX: usize = 40;

/// Post-burn extinction ramp (seconds) — once every letter is
/// consumed the fire is fuel-starved: remaining motes snuff on a fast
/// shared fade instead of lingering over empty paper, then the pools
/// clear so `live_count` hits 0 and the session sleeps.
const EXTINCT_S: f32 = 0.55;

/// One letter's burn unit — rect in paper px + speed + ignition delay.
#[derive(Clone, Copy, Default)]
pub(crate) struct Letter {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    v: f32,
    delay: f32,
}

/// A live particle — the JS version stored 12 fields; same set here
/// plus the thermal/plume state the real-fire model needs.
#[derive(Clone, Copy)]
pub(crate) struct Particle {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) vx: f32,
    pub(crate) vy: f32,
    pub(crate) life: f32,
    pub(crate) max_life: f32,
    pub(crate) size: f32,
    /// 0 = ember, 1 = char fleck, 2 = smoke puff, 3 = flame lick.
    pub(crate) kind: u8,
    pub(crate) rot: f32,
    pub(crate) spin: f32,
    pub(crate) phase: f32,
    /// Particle temperature (K) — embers/flecks cool toward ambient;
    /// smoke uses this slot as the puff's birth diameter instead.
    pub(crate) temp: f32,
    /// Accumulated upward travel (px) — the smoke plume's entrainment
    /// coordinate: radius grows and velocity decays with rise, not age.
    pub(crate) rise: f32,
    /// Birth upward velocity (px/s, negative = up) — the smoke plume's
    /// w(z) = w_term + (w0 − w_term)·(1+z/Z0)^(−1/3) curve needs the
    /// launch value every frame.
    pub(crate) vy0: f32,
    /// Owning letter index (rebuild deposits only).
    pub(crate) letter: u32,
}

/// Per-widget burn session — one per mosaic title hover.
pub struct BurnSim {
    /// Particle pool — retained capacity across frames.
    pub(crate) particles: Vec<Particle>,
    /// Letter rects/speeds pushed at mount.
    pub(crate) letters: Vec<Letter>,
    /// Paper-space ink occupancy — 1 byte per column.
    pub(crate) col_ink: Vec<u8>,
    /// Ember ramp: core → mid → edge by life fraction.
    ink_core: [f32; 4],
    ink_mid: [f32; 4],
    ink_edge: [f32; 4],
    /// Char-fleck scorch tint.
    ink_scorch: [f32; 4],
    /// Smoke-puff tint — the rising ash column.
    ink_smoke: [f32; 4],
    /// Smoke pool — kept separate from `particles` so the plume layer
    /// emits under the sparks with its own budget.
    pub(crate) smoke: Vec<Particle>,
    /// Flame-lick pool — the bright tongue band riding the front.
    pub(crate) flames: Vec<Particle>,
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
    /// Fuel-starvation clock — counts up once no live front remains;
    /// every emitted alpha is scaled by its ramp so the whole effect
    /// vanishes shortly after the last glyph is consumed.
    extinct: f32,
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

/// `mix4` — rgba lerp.
fn mix4(a: [f32; 4], b: [f32; 4], k: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * k,
        a[1] + (b[1] - a[1]) * k,
        a[2] + (b[2] - a[2]) * k,
        a[3] + (b[3] - a[3]) * k,
    ]
}

/// `heat_ramp` — the blackbody ramp over the session's ink palette.
/// A burning mote's color IS its temperature (Planckian locus):
/// white-hot at ejection cooling through yellow/orange to deep red,
/// then dark char. `heat` 0..1 maps ambient→ejecta onto scorch→edge
/// →mid→core so the theme ramp doubles as the thermal ramp — keying
/// color to temperature instead of age is what makes the cooling
/// read as physics rather than a fade.
fn heat_ramp(
    heat: f32,
    scorch: [f32; 4],
    edge: [f32; 4],
    mid: [f32; 4],
    core: [f32; 4],
) -> [f32; 4] {
    if heat > 0.66 {
        mix4(mid, core, (heat - 0.66) / 0.34)
    } else if heat > 0.33 {
        mix4(edge, mid, (heat - 0.33) / 0.33)
    } else {
        mix4(scorch, edge, (heat / 0.33).max(0.0))
    }
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
            ink_smoke: [0.0; 4],
            smoke: Vec::with_capacity(SMOKE_MAX),
            flames: Vec::with_capacity(FLAME_MAX),
            origin,
            t: 0.0,
            dir,
            stage: 0.0,
            jag_amp: 0.0,
            extinct: 0.0,
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
    /// session when the front is spent and the last mote died. Smoke
    /// counts too: the plume outlives the sparks by seconds, so a
    /// sparks-only check would cut the session mid-dissipation.
    pub fn live_count(&self) -> usize {
        self.particles.len() + self.smoke.len() + self.flames.len()
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
    /// Re-inking paper doesn't smoke, so the plume clears too.
    pub fn set_dir(&mut self, dir: f32) {
        self.dir = dir;
        self.particles.clear();
        self.smoke.clear();
        self.flames.clear();
        self.extinct = 0.0;
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

        if flat.len() >= 20 {
            take(4, &mut self.ink_smoke);
        } else if flat.len() >= 16 {
            // Legacy 4-tint callers — derive the plume from scorch:
            // desaturate toward mid-grey, keep it faint.
            let s = self.ink_scorch;

            self.ink_smoke = [
                (s[0] + 0.5) * 0.5,
                (s[1] + 0.5) * 0.5,
                (s[2] + 0.5) * 0.5,
                0.4,
            ];
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
            // Ejecta temperature — flecks are thin char (cool fast, dim
            // edge), embers leave the reaction zone near-white-hot.
            let temp = if fleck {
                1200.0 + self.rnd() * 400.0
            } else {
                T_EJECT_MIN + self.rnd() * (T_EJECT_MAX - T_EJECT_MIN)
            };

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
                temp,
                rise: 0.0,
                vy0: vy,
                letter: if rebuild { li as u32 } else { u32::MAX },
            });

            i += 1.0;
        }
    }

    /// Smoke spawn — same front-gating as sparks but its own rate and
    /// pool. Burn mode only: a re-inking front emits no smoke.
    fn spawn_smoke(&mut self, progress: f32, jag_amp: f32, dt: f32) {
        if self.dir < 0.0 || self.letters.is_empty() {
            return;
        }

        let want = SMOKE_PER_S * dt;
        let n_letters = self.letters.len();
        let mut i = 0.0;

        while i < want {
            if self.rnd() > want - i {
                i += 1.0;
                continue;
            }

            if self.smoke.len() >= SMOKE_MAX {
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

            let x = self.origin[0] + xp as f32;
            // Puffs condense slightly above the front — smoke lifts off
            // the charring line rather than inside the flame band.
            let y = self.origin[1] + fy - self.rnd() * 6.0;

            let vy = -(SMOKE_VY_MIN + self.rnd() * (SMOKE_VY_MAX - SMOKE_VY_MIN));
            let vx = (self.rnd() * 2.0 - 1.0) * SMOKE_VX;
            let max_life = SMOKE_LIFE_MIN + self.rnd() * (SMOKE_LIFE_MAX - SMOKE_LIFE_MIN);
            let size = SMOKE_SIZE0 * (0.7 + self.rnd() * 0.6);
            let rot = self.rnd() * std::f32::consts::TAU;
            let spin = (self.rnd() * 2.0 - 1.0) * SMOKE_SPIN;
            let phase = self.rnd() * std::f32::consts::TAU;

            self.smoke.push(Particle {
                x,
                y,
                vx,
                vy,
                life: 0.0,
                max_life,
                size,
                kind: 2,
                rot,
                spin,
                phase,
                // `temp` doubles as the puff's BIRTH diameter — the
                // entrainment growth needs it every frame.
                temp: size,
                rise: 0.0,
                vy0: vy,
                letter: u32::MAX,
            });

            i += 1.0;
        }
    }

    /// Flame-lick spawn — the visible fire sheet: short-lived bright
    /// tongues born AT the burn front, racing up on strong buoyancy.
    /// Burn mode only; the front gates identically to the sparks.
    fn spawn_flames(&mut self, progress: f32, jag_amp: f32, dt: f32) {
        if self.dir < 0.0 || self.letters.is_empty() {
            return;
        }

        let want = FLAME_PER_S * dt;
        let n_letters = self.letters.len();
        let mut i = 0.0;

        while i < want {
            if self.rnd() > want - i {
                i += 1.0;
                continue;
            }

            if self.flames.len() >= FLAME_MAX {
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

            let x = self.origin[0] + xp as f32;
            // Licks root on the front and immediately peel upward.
            let y = self.origin[1] + fy - self.rnd() * 3.0;
            let vy = -(FLAME_VY_MIN + self.rnd() * (FLAME_VY_MAX - FLAME_VY_MIN));
            let vx = (self.rnd() * 2.0 - 1.0) * FLAME_VX;
            let max_life = FLAME_LIFE_MIN + self.rnd() * (FLAME_LIFE_MAX - FLAME_LIFE_MIN);
            let size = FLAME_SIZE_MIN + self.rnd() * (FLAME_SIZE_MAX - FLAME_SIZE_MIN);
            let rot = self.rnd() * std::f32::consts::TAU;
            let spin = (self.rnd() * 2.0 - 1.0) * 3.0;
            let phase = self.rnd() * std::f32::consts::TAU;
            // Flame gas sits near reaction-zone temperature.
            let temp = 1600.0 + self.rnd() * 500.0;

            self.flames.push(Particle {
                x,
                y,
                vx,
                vy,
                life: 0.0,
                max_life,
                size,
                kind: 3,
                rot,
                spin,
                phase,
                temp,
                rise: 0.0,
                vy0: vy,
                letter: u32::MAX,
            });

            i += 1.0;
        }
    }

    /// Any letter still burning — the extinction ramp starts the frame
    /// every glyph's front has passed its ink top (fuel starvation).
    fn fronts_alive(&self) -> bool {
        self.letters
            .iter()
            .any(|l| letter_progress(l, self.stage) < 1.0)
    }

    /// Advances + emits: smoke billows on plume entrainment, flame
    /// licks race off the front, embers cool on the blackbody ramp
    /// while their buoyancy dies, flecks tumble dark, rebuild deposits
    /// land on the front (particles.ts `drawParticles`). Stage/jag
    /// state arrives via `set_stage` each frame. Emits into `quads` —
    /// one instanced draw for all live particles.
    pub fn tick(&mut self, dt: f32, quads: &mut Vec<f32>) {
        let stage = self.stage;
        let jag_amp = self.jag_amp;

        self.t += dt;
        // `stage` climbs 0→1 in both directions (JS pushes 1−prog on
        // rebuild) — the same value the landing checks use below. The
        // earlier `1.0 − stage` inverted the front: motes spawned on
        // already-consumed paper and peaked exactly when the title was
        // gone (the "burn won't vanish" report).
        self.spawn(stage, jag_amp, dt);
        self.spawn_smoke(stage, jag_amp, dt);
        self.spawn_flames(stage, jag_amp, dt);

        // Fuel starvation → global extinction. The first frame no
        // letter's front is mid-flight, the fire is out of fuel: every
        // remaining mote fades on a shared fast ramp (sparks snuff,
        // the plume thins) and the pools empty when it lands — nothing
        // is left glowing over consumed paper.
        if self.dir > 0.0 && !self.fronts_alive() {
            self.extinct += dt;

            if self.extinct >= EXTINCT_S {
                self.particles.clear();
                self.smoke.clear();
                self.flames.clear();

                return;
            }
        }

        // The extinction multiplier — 1 while fuel remains, ramping to
        // 0 over EXTINCT_S once every letter is ash.
        let die = if self.dir > 0.0 {
            1.0 - (self.extinct / EXTINCT_S).min(1.0)
        } else {
            1.0
        };

        // ── Smoke plume — emits FIRST: the translucent column sits
        // behind flame and sparks in painter order, matching how a
        // real fire reads (bright motes in front of the ash column).
        let t = self.t;
        let ink_smoke = self.ink_smoke;
        let mut i = self.smoke.len();

        while i > 0 {
            i -= 1;

            let p = &mut self.smoke[i];

            p.life += dt;

            if p.life >= p.max_life {
                self.smoke.swap_remove(i);
                continue;
            }

            let k = p.life / p.max_life;

            // Rise bookkeeping FIRST — plume laws key on height above
            // the source, not on elapsed time.
            p.rise += (-p.vy).max(0.0) * dt;

            // Plume axial velocity — a pure buoyant plume's centerline
            // speed decays ≈ w_term + (w0 − w_term)·(1 + z/Z0)^(−1/3)
            // as entrained air diffuses its momentum. The puff relaxes
            // onto that curve (rate 6/s) — stable at any frame rate.
            let w_target = SMOKE_VY_TERMINAL
                + (p.vy0 - SMOKE_VY_TERMINAL) * (1.0 + p.rise / SMOKE_Z0).powf(-1.0 / 3.0);
            p.vy += (w_target - p.vy) * 6.0 * dt;
            p.vx += (SMOKE_DRIFT_X - p.vx) * 0.6 * dt;

            // Entrainment growth — a self-similar turbulent plume
            // widens linearly with height (d = d0 + 2αz, α≈0.11):
            // the diameter tracks RISE, not age.
            p.size = p.temp + SMOKE_ENTRAIN * p.rise;

            // Eddies scale with the plume — the wander amplitude grows
            // with height so the column sways harder as it disperses.
            let wander = SMOKE_WANDER * (1.0 + p.rise / 140.0);
            p.x += p.vx * dt + (p.phase * 2.0 + t * 1.6).sin() * wander * dt;
            p.y += p.vy * dt;
            p.rot += p.spin * dt;

            // Opacity — condense-in/dissipate-out envelope × soot
            // dilution (mass conserved over a widening disc → optical
            // depth falls ~d0/d) × the session extinction ramp.
            let fade_in = (k / 0.12).min(1.0);
            let fade_out = ((1.0 - k) / 0.55).min(1.0);
            let dilute = p.temp / p.size;
            let a = fade_in * fade_out * dilute * SMOKE_PEAK_A * ink_smoke[3] * die;

            let half = p.size * 0.5;

            // uv record doubles as the shape block for soft particles:
            // x=1 flags the billow path, y=rot drives the erosion
            // rotation, z=phase seeds the density noise, w=0 smoke.
            quads.extend_from_slice(&[
                p.x - half,
                p.y - half,
                p.size,
                p.size,
                1.0,
                p.rot,
                p.phase,
                0.0,
                ink_smoke[0],
                ink_smoke[1],
                ink_smoke[2],
                a,
                6.0,
                0.0,
                0.0,
                0.0,
            ]);
        }

        // ── Flame licks — the fire's visible sheet. Short-lived tall
        // tongues riding hard buoyancy; they emit between plume and
        // sparks so the flame band glows through its own smoke.
        let ink_scorch = self.ink_scorch;
        let ink_edge = self.ink_edge;
        let ink_mid = self.ink_mid;
        let ink_core = self.ink_core;
        let mut i = self.flames.len();

        while i > 0 {
            i -= 1;

            let p = &mut self.flames[i];

            p.life += dt;

            if p.life >= p.max_life {
                self.flames.swap_remove(i);
                continue;
            }

            let k = p.life / p.max_life;

            // Continuous strong lift — flame gas stays near reaction
            // temperature its whole short life, so no cooling term.
            p.vy -= FLAME_BUOY * dt;
            p.vx *= (1.0 - 4.0 * dt).max(0.0);
            p.x += p.vx * dt + (p.phase + t * 22.0).sin() * 16.0 * dt;
            p.y += p.vy * dt;
            p.rot += p.spin * dt;

            // Hot gas expands as it climbs — modest growth; the necking
            // pinch a real tongue shows comes from shader erosion.
            let grow = 1.0 + k * 0.6;

            // Envelope — ignites fast (15%), holds, snuffs over the
            // last 60%: tongues pop off the front, burn, detach.
            let a = (k / 0.15).min(1.0) * ((1.0 - k) / 0.6).min(1.0) * FLAME_PEAK_A * die;

            // Cools visibly over its short life — core-white at the
            // root toward mid-orange at the tip's death.
            let heat = (1.0 - k) * 0.5 + 0.5;
            let tint = heat_ramp(heat, ink_scorch, ink_edge, ink_mid, ink_core);

            // Tall tongues — half-extents differ so the quad is
            // portrait, not round: width ~60% of height.
            let hw = p.size * 0.30 * grow;
            let hh = p.size * 0.5 * grow;

            // Same soft-eroded record as smoke but w=1 — the shader's
            // flame variant (tighter core, higher-frequency licking).
            quads.extend_from_slice(&[
                p.x - hw,
                p.y - hh,
                hw * 2.0,
                hh * 2.0,
                1.0,
                p.rot,
                p.phase,
                1.0,
                tint[0],
                tint[1],
                tint[2],
                a,
                7.0,
                0.0,
                0.0,
                0.0,
            ]);
        }

        let rebuild = self.dir < 0.0;
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
                if rebuild {
                    // Re-ink deposit — gravity down toward the front.
                    p.vy += FLECK_G * dt;
                    p.x += p.vx * dt + (p.phase + t * 9.0).sin() * 14.0 * dt;
                    p.y += p.vy * dt;
                } else {
                    // ── Thermal ember model ─────────────────────────
                    // Newtonian cooling — mm-scale char sheds heat to
                    // convection+radiation in ~a second; that time
                    // constant IS why real sparks die fast.
                    p.temp += (T_AMBIENT - p.temp) * EMBER_COOL * dt;

                    if p.temp < T_EXTINCT {
                        // Winked out — cooled past incandescence.
                        self.particles.swap_remove(i);
                        continue;
                    }

                    let heat = ((p.temp - T_AMBIENT) / (T_EJECT_MAX - T_AMBIENT)).clamp(0.0, 1.0);

                    // Buoyant lift ∝ excess heat — a mote only climbs
                    // while it's hot; under ~810 K gravity wins and it
                    // falls back as dead ash.
                    let grav = if p.temp < T_EXTINCT + 170.0 {
                        EMBER_ASH_G
                    } else {
                        0.0
                    };
                    p.vy += (grav - EMBER_BUOY * heat) * dt;

                    // Quadratic drag — dv/dt = −k·|v|·v, integrated as
                    // an exponential damping factor (frame-stable).
                    let sp = (p.vx * p.vx + p.vy * p.vy).sqrt();
                    let d = (EMBER_DRAG * sp * dt).min(0.92);
                    p.vx *= 1.0 - d;
                    p.vy *= 1.0 - d;

                    // Convective wobble scales with heat — cold ash
                    // drifts nearly straight, hot sparks dance.
                    p.x += p.vx * dt + (p.phase + t * 9.0).sin() * 14.0 * (0.25 + heat * 0.75) * dt;
                    p.y += p.vy * dt;

                    // Blackbody tint on temperature + Stefan–Boltzmann-
                    // shaped luminosity (T⁴ compressed to ~T^2.2 for
                    // sRGB output) + oxidation-burst flicker.
                    let tint = heat_ramp(heat, ink_scorch, ink_edge, ink_mid, ink_core);
                    let lum = 0.15 + 0.85 * heat.powf(2.2);
                    let flick = 0.55 + 0.45 * (p.phase + t * 31.0).sin();

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
                        (1.0 - k) * flick * lum * tint[3] * die,
                        9.0,
                        0.0,
                        0.0,
                        0.0,
                    ]);

                    continue;
                }

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
                    ink_core
                } else if k < 0.7 {
                    ink_mid
                } else {
                    ink_edge
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
                    (1.0 - k) * flick * tint[3] * die,
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

                // Flecks shed heat fast (thin char) — their tint dims
                // from a hot scorch edge toward dead black as they
                // cool, rather than holding a flat scorch color.
                p.temp += (T_AMBIENT - p.temp) * FLECK_COOL * dt;
                let heat = ((p.temp - T_AMBIENT) / (1600.0 - T_AMBIENT)).clamp(0.0, 1.0);
                let ftint = mix4(ink_scorch, ink_edge, heat * 0.8);

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
                    ftint[0],
                    ftint[1],
                    ftint[2],
                    (1.0 - k) * ftint[3] * die,
                    8.0,
                    0.0,
                    0.0,
                    0.0,
                ]);
            }
        }
    }
}
