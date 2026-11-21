"""Phase A0: openings detector, built-in safety filter, hand driver."""

import numpy as np
import pytest

from submarine.control import (DriverConfig, DriverPolicy, Odometry, SafetyConfig, SafetyFilter, detect_openings,
                               scan_points)
from submarine.env import CaveEnv
from submarine.sim import Sonar, SonarConfig, VehicleConfig, VehicleState, step_vehicle

RES = 0.2
SONAR = Sonar(SonarConfig().ideal())
V = VehicleConfig()


def grid(free_boxes, shape=(50, 100)):
    """Rock everywhere except the given (x0, x1, y0, y1) boxes in metres (map 20 m x 10 m)."""
    rock = np.ones(shape, dtype=bool)
    for x0, x1, y0, y1 in free_boxes:
        rock[int(y0 / RES):int(y1 / RES), int(x0 / RES):int(x1 / RES)] = False
    return rock


CORRIDOR = grid([(0, 20, 4, 6)])
DEAD_END = grid([(0, 8, 4, 6)])
T_JUNCTION = grid([(0, 20, 4, 6), (9, 11, 4, 10)])


def scan(rock, x, y, psi=0.0):
    return SONAR.true_ranges(rock, RES, x, y, psi)


# ---------------------------------------------------------------- openings

def test_openings_corridor_junction_dead_end():
    ops = detect_openings(scan(CORRIDOR, 5, 5), SONAR.angles, 10.0)
    front = [o for o in ops if not o.rear]
    assert len(front) == 1 and abs(front[0].bearing) < np.deg2rad(15) and sum(o.rear for o in ops) == 1

    ops = detect_openings(scan(T_JUNCTION, 10, 5), SONAR.angles, 10.0)
    front = sorted(np.degrees([o.bearing for o in ops if not o.rear]))
    assert len(front) == 2 and abs(front[0]) < 15 and abs(front[1] - 90) < 20      # straight on + left branch

    ops = detect_openings(scan(DEAD_END, 7.2, 5), SONAR.angles, 10.0)            # 0.8 m from the end wall
    assert not [o for o in ops if not o.rear] and [o for o in ops if o.rear]       # only the way back

    ops = detect_openings(scan(DEAD_END, 6.0, 5), SONAR.angles, 10.0)            # 2 m from it: not yet a
    assert [o.narrow for o in ops if not o.rear] == [True]                          # dead end, a narrow way on


# ---------------------------------------------------------------- safety filter

def test_odometry_matches_vehicle_without_current():
    s, o = VehicleState(0.0, 0.0, 0.0), Odometry()
    rng = np.random.default_rng(0)
    for _ in range(100):
        s = step_vehicle(s, rng.uniform(-1, 1, 2), V)
        o.step(s.u, s.r, V.dt)
    assert np.allclose([o.x, o.y], [s.x, s.y], atol=1e-9) and np.isclose(np.cos(o.psi - s.psi), 1.0)


def test_dropout_bridged():
    r = scan(CORRIDOR, 5, 5)
    left = int(np.argmin(np.abs(SONAR.angles - np.pi / 2)))
    holed = r.copy()
    holed[left] = 10.0                                   # a dropout in the left wall
    pts = scan_points(holed, SONAR.angles, 10.0, 0, 0, 0, SafetyConfig())
    near_left = pts[(np.abs(pts[:, 0]) < 0.15) & (pts[:, 1] > 0)]
    assert len(near_left) and np.allclose(near_left[:, 1], 1.0, atol=0.15)    # wall still there


def make_filter(rock, x, y):
    f = SafetyFilter(SafetyConfig(), V, SONAR.angles, 10.0)
    f.update_scan(scan(rock, x, y))
    return f


def test_filter_brakes_before_wall_and_allows_when_far():
    f = make_filter(DEAD_END, 6.4, 5)                    # wall 1.6 m ahead: can't stop in time from 1.5 m/s
    a, over = f.filter((1.0, 0.0), u=1.5, r=0.0, scan_age=0.0)
    assert over and a[0] < 0
    a, over = f.filter((1.0, 0.0), u=0.0, r=0.0, scan_age=0.0)
    assert not over and a[0] == 1.0                      # from rest the stopping path is short
    a, over = f.filter((0.0, 1.0), u=0.0, r=0.0, scan_age=0.0)
    assert not over                                      # turning in place is never blocked


def test_hard_turn_at_speed_is_softened_near_a_side_wall():
    """Phase B crash: full-rate turn toward a side wall while still moving. Braking alone can't stop the hull
    sweeping into the wall; wherever only a gentler turn is safe, the filter must choose one."""
    rng = np.random.default_rng(3)
    found = 0
    for _ in range(400):
        y0 = rng.uniform(4.55, 5.45)
        f = make_filter(CORRIDOR, 5, y0)
        u, r, yaw = rng.uniform(0.5, 1.5), rng.uniform(-1, 1), rng.choice([-1.0, 1.0])
        need = f.required(u, 0.0)
        thrust_only = f.clearances(u, r, np.array([(t, yaw) for t in f.cfg.thrusts]))
        softer = f.clearances(u, r, np.array([(t, yaw * k) for k in f.cfg.yaw_scales for t in f.cfg.thrusts]))
        if thrust_only.max() < need <= softer.max():
            found += 1
            a, over = f.filter((-1.0, yaw), u, r, 0.0)
            assert over and abs(a[1]) < abs(yaw)
    assert found >= 5                                    # the situation really occurs


def test_stale_scan_needs_more_room():
    f = make_filter(DEAD_END, 4.4, 5)
    fresh = f.required(u=1.2, scan_age=0.0)
    stale = f.required(u=1.2, scan_age=0.4)
    assert stale > fresh


# ---------------------------------------------------------------- driver in the environment

@pytest.mark.parametrize("ping", [0.1, 0.4])
def test_full_thrust_request_stops_short_of_the_wall(ping):
    """Policy asks for full thrust straight ahead forever: the driver's filter must stop it, not crash."""
    e = CaveEnv({"difficulty": 0.0, "sonar": {"ping_interval": ping}, "max_steps": 300})
    obs, _ = e.reset(seed=0, options={"cave_seed": 0, "difficulty": 0.0})
    pol = DriverPolicy()
    pol.reset(e)
    d, L = pol.driver, pol.L
    for _ in range(300):
        d.sense(L.ranges(obs), L.speed(obs), L.yaw_rate(obs), L.ping_age(obs))
        a, info = d.act(proposed=np.array([1.0, 0.0], np.float32))
        obs, r, te, tr, info = e.step(a)
        assert not info["collided"]
        if te or tr:
            break
    assert abs(e.sim.state.u) < 0.2 and pol.driver.n_overrides > 0


def test_driver_policy_completes_easy_tunnel():
    e = CaveEnv({"difficulty": 0.0, "sonar": {"ping_interval": 0.4}, "max_steps": 900})
    obs, _ = e.reset(seed=0, options={"cave_seed": 1000000, "difficulty": 0.0})
    pol = DriverPolicy()
    pol.reset(e)
    while True:
        obs, r, te, tr, info = e.step(pol.act(obs))
        if te or tr:
            break
    assert info["is_success"] and not info["collided"]


def test_driver_config_from_dict():
    c = DriverConfig.from_dict({"cruise_speed": 1.0, "safety": {"floor": 0.2, "thrusts": [1, 0, -1],
                                                                  "yaw_scales": [0.0]}})
    assert c.cruise_speed == 1.0 and c.safety.floor == 0.2 and c.safety.thrusts == (1, 0, -1)
    assert c.safety.yaw_scales == (0.0,)


# ---------------------------------------------------------------- fast paths == reference

def _scan_points_reference(ranges, angles, max_range, x, y, psi, cfg, full_circle=True):
    """The original loop version of scan_points (before vectorisation)."""
    r = np.asarray(ranges, dtype=float).copy()
    hit = r < max_range - 1e-6
    n = len(r)
    left, right = np.roll(r, 1), np.roll(r, -1)
    hl, hr = np.roll(hit, 1), np.roll(hit, -1)
    bridge = ~hit & hl & hr & (np.abs(left - right) < cfg.join_frac * np.maximum(left, right) + cfg.join_abs)
    r[bridge] = 0.5 * (left[bridge] + right[bridge])
    hit |= bridge
    a = psi + np.asarray(angles, dtype=float)
    px, py = x + r * np.cos(a), y + r * np.sin(a)
    pts = [np.stack([px[hit], py[hit]], axis=1)]
    for i in range(n):
        j = (i + 1) % n
        if not (hit[i] and hit[j]) or abs(r[i] - r[j]) >= cfg.join_frac * max(r[i], r[j]) + cfg.join_abs:
            continue
        k = int(np.ceil(np.hypot(px[j] - px[i], py[j] - py[i]) / cfg.spacing)) - 1
        if k > 0:
            t = np.arange(1, k + 1) / (k + 1)
            pts.append(np.stack([px[i] + t * (px[j] - px[i]), py[i] + t * (py[j] - py[i])], axis=1))
    return np.concatenate(pts)


def test_vectorised_scan_points_match_reference():
    rng = np.random.default_rng(0)
    for rock, x in ((CORRIDOR, 5), (T_JUNCTION, 10), (DEAD_END, 6.5)):
        r = scan(rock, x, 5) + rng.normal(0, 0.03, 32)
        r[rng.integers(32)] = 10.0
        a = _scan_points_reference(r, SONAR.angles, 10.0, 1.0, 2.0, 0.3, SafetyConfig())
        b = scan_points(r, SONAR.angles, 10.0, 1.0, 2.0, 0.3, SafetyConfig())
        assert a.shape == b.shape and np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))


def test_fast_filter_matches_batched_reference():
    """filter() (scalar paths, lazy candidates, unqueried lower bound) must decide exactly like the batched
    clearances() over all candidates."""
    rng = np.random.default_rng(1)
    for rock, x in ((DEAD_END, 6.4), (DEAD_END, 5.0), (T_JUNCTION, 10), (CORRIDOR, 5)):
        f = make_filter(rock, x, 5)
        for _ in range(60):
            u, r = rng.uniform(-0.7, 1.5), rng.uniform(-1, 1)
            act = rng.uniform(-1, 1, 2).astype(np.float32)
            age = rng.choice([0.0, 0.2, 0.4])
            got, over = f.filter(act, u, r, age)
            need = f.required(u, age)
            cands = f.candidates(float(act[0]), float(act[1]))
            clr = f.clearances(u, r, np.array([act] + cands))
            if clr[0] >= need:
                want = act
            else:
                safe = np.flatnonzero(clr[1:] >= need)
                want = np.array(cands[int(safe[0]) if safe.size else int(np.argmax(clr[1:]))])
            assert np.allclose(got, want, atol=1e-6) and over == (clr[0] < need)
