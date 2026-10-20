//! Host coverage for the burn CPU sim — the physics contract the
//! user reported broken: particles must spawn on the LIVE front
//! (stage direction), everything must vanish once the text is
//! consumed (extinction ramp → `live_count` 0), and the run must be
//! deterministic (seeded xorshift, no Math.random).
//!
//! Lives in its own file so the physics module stays focused; the
//! internals it inspects are `pub(crate)`.

use crate::sim::*;

/// One letter 10×20 paper px + full ink + a full tint table — the
/// minimal live-front fixture every test builds on.
fn fixture(dir: f32) -> BurnSim {
    let mut sim = BurnSim::new(dir, [0.0, 0.0], 42);

    // x0,y0,x1,y1, v,delay — progress 0.5 → the front sits at the
    // letter's vertical middle.
    sim.set_letters(&[0.0, 10.0, 10.0, 30.0, 1.0, 0.0]);
    sim.set_ink(&[1; 64]);
    sim.set_colors(&[
        1.0, 0.9, 0.5, 1.0, // core — white-hot
        1.0, 0.5, 0.1, 1.0, // mid — orange
        0.8, 0.2, 0.05, 1.0, // edge — deep red
        0.1, 0.05, 0.02, 1.0, // scorch — char
        0.6, 0.6, 0.6, 1.0, // smoke — grey plume
    ]);

    sim
}

/// Ticks the sim `steps` frames of `dt` seconds and returns the LAST
/// frame's emitted quad records (stride `QUAD_F32S`).
fn run(sim: &mut BurnSim, dt: f32, steps: usize) -> Vec<f32> {
    let mut last = Vec::new();

    for _ in 0..steps {
        last.clear();
        sim.tick(dt, &mut last);
    }

    last
}

#[test]
/// Stage=0.15 puts the burn front near the letter BOTTOM (progress
/// climbs bottom→top): spawns must land there. The old
/// `spawn(1.0 - stage)` inversion used progress 0.85 instead — a
/// front near the letter TOP — so this bounds check fails on the
/// regression and only there (at 0.5 the two fronts coincide).
fn particles_spawn_on_the_live_front() {
    let mut sim = fixture(1.0);
    sim.set_stage(0.15, 0.0);

    let quads = run(&mut sim, 0.016, 2);

    assert!(sim.live_count() > 0, "a live front must emit particles");
    assert!(!quads.is_empty(), "a live front must emit quads");

    // True front y = y1 − 0.15·(y1−y0) = 27; inverted front = 13.
    // Pool y must sit in the lower half of the letter — smoke/flame
    // offsets lift at most ~9 px, one 16ms tick adds ≤3 px of climb.
    let pools = [&sim.particles[..], &sim.smoke[..], &sim.flames[..]].concat();

    for p in &pools {
        assert!(
            p.y >= 15.0 && p.y <= 32.0,
            "spawn y {} must hug the stage-0.15 front (~27), not the inverted one (~13)",
            p.y
        );
    }
}

#[test]
/// With the front fully consumed (stage 1.0) no new particles spawn —
/// the pre-fix inversion spawned the MOST at stage 1.
fn no_spawns_once_the_front_is_gone() {
    let mut sim = fixture(1.0);
    sim.set_stage(1.0, 0.0);

    let quads = run(&mut sim, 0.05, 2);

    assert_eq!(sim.live_count(), 0);
    assert!(quads.is_empty());
}

#[test]
/// The reported bug — the burn must fully vanish once the text is
/// consumed: burn through stage 0→1, then keep ticking past the
/// extinction ramp; live_count must reach 0 and stay there.
fn effect_extinguishes_to_zero_after_consumption() {
    let mut sim = fixture(1.0);
    sim.set_stage(0.5, 0.0);

    run(&mut sim, 0.05, 4);
    assert!(sim.live_count() > 0, "front alive → particles live");

    // Consume every letter, then run past EXTINCT_S.
    sim.set_stage(1.0, 0.0);
    let quads = run(&mut sim, 0.1, 10);

    assert_eq!(sim.live_count(), 0, "extinct sim must empty every pool");
    assert!(quads.is_empty(), "extinct sim must emit nothing");
}

#[test]
/// Rebuild mode (dir < 0) re-inks paper — it must never emit the
/// smoke plume or flame licks; those pools are burn-only.
fn rebuild_emits_neither_smoke_nor_flames() {
    let mut sim = fixture(-1.0);
    sim.set_stage(0.5, 0.0);

    run(&mut sim, 0.1, 6);

    assert!(sim.smoke.is_empty(), "rebuild must not smoke");
    assert!(sim.flames.is_empty(), "rebuild must not flame");
}

#[test]
/// Burn mode emits all three pools — smoke behind, licks in the
/// band — so `live_count` covers all three (a sparks-only settle
/// check would idle the session mid-plume).
fn burn_mode_uses_all_three_pools() {
    let mut sim = fixture(1.0);
    sim.set_stage(0.5, 0.0);

    run(&mut sim, 0.1, 6);

    assert!(!sim.particles.is_empty(), "embers must spawn");
    assert!(!sim.smoke.is_empty(), "smoke must spawn");
    assert!(!sim.flames.is_empty(), "flame licks must spawn");
    assert_eq!(
        sim.live_count(),
        sim.particles.len() + sim.smoke.len() + sim.flames.len()
    );
}

#[test]
/// Embers are thermal: launched ~2000 K they cool toward ambient on
/// Newtonian law and wink out below T_EXTINCT — never lingering past
/// their fuel.
fn embers_cool_and_wink_out() {
    let mut sim = fixture(1.0);
    sim.set_stage(0.5, 0.0);

    run(&mut sim, 0.016, 2);
    let first_gen = sim.particles.iter().filter(|p| p.kind == 0).count();

    assert!(first_gen > 0, "embers must exist to be tested");

    let mut birth_max = 0.0f32;

    for p in sim.particles.iter().filter(|p| p.kind == 0) {
        birth_max = birth_max.max(p.temp);
    }

    assert!(birth_max >= T_EJECT_MIN, "embers launch near-white-hot");

    // ~1.6 s of cooling → every first-generation ember is ash; any
    // ember still live must be a younger spawn.
    run(&mut sim, 0.016, 100);

    for p in sim.particles.iter().filter(|p| p.kind == 0) {
        assert!(p.life < 1.7, "a >1.6 s-old ember must have winked out");
    }
}

#[test]
/// Smoke plume physics — puffs keep rising, and the diameter obeys
/// the entrainment identity d = d0 + 2α·rise (size tracks the rise
/// coordinate, not age).
fn smoke_follows_the_plume_law() {
    let mut sim = fixture(1.0);
    sim.set_stage(0.5, 0.0);

    run(&mut sim, 0.05, 10);

    assert!(!sim.smoke.is_empty(), "fixture must produce smoke");

    for p in &sim.smoke {
        let want = p.temp + SMOKE_ENTRAIN * p.rise;

        assert!(
            (p.size - want).abs() < 1e-3,
            "smoke size {} must equal birth+entrained rise {}",
            p.size,
            want
        );
        assert!(p.vy < 0.0, "smoke must keep rising (+y is down)");
    }
}

#[test]
/// The seeded xorshift makes a session replay bit-identical — the
/// frame-budget rule bans per-frame Math.random.
fn same_seed_replays_identically() {
    let mut a = fixture(1.0);
    let mut b = fixture(1.0);

    a.set_stage(0.5, 0.0);
    b.set_stage(0.5, 0.0);

    let qa = run(&mut a, 0.033, 8);
    let qb = run(&mut b, 0.033, 8);

    assert_eq!(qa, qb, "same seed + same stages → identical emit");
}

#[test]
/// `set_dir` restarts the particle stream but keeps letters/ink —
/// the hover-out→hover-in flip path.
fn direction_flip_resets_pools_keeps_geometry() {
    let mut sim = fixture(1.0);
    sim.set_stage(0.5, 0.0);

    run(&mut sim, 0.05, 4);
    assert!(sim.live_count() > 0);

    sim.set_dir(-1.0);

    assert_eq!(sim.live_count(), 0, "flip must drain all pools");
    assert!(!sim.letters.is_empty(), "geometry survives the flip");
    assert!(!sim.col_ink.is_empty(), "ink survives the flip");
}
