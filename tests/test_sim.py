from dataclasses import replace

import numpy as np
import pytest

from submarine.world import generate_cave
from submarine.sim import SonarConfig, Simulator, Sonar, VehicleConfig, VehicleState, sample_bilinear, step_vehicle

RES = 0.2


def corridor():
    """20 m x 10 m map, free only for 4 m <= y < 6 m."""
    rock = np.ones((50, 100), dtype=bool)
    rock[20:30, :] = False
    return rock


IDEAL = SonarConfig().ideal()
FWD, LEFT, BACK = 16, 24, 0   # beam indices for 32 beams over 360 degrees


# ---------------------------------------------------------------- sonar

def test_beam_layout():
    s = Sonar(IDEAL)
    assert s.angles[FWD] == 0 and np.isclose(s.angles[LEFT], np.pi / 2) and np.isclose(s.angles[BACK], -np.pi)
    f = Sonar(SonarConfig(n_beams=24, fov_deg=120))
    assert np.isclose(f.angles[0], -np.pi / 3) and np.isclose(f.angles[-1], np.pi / 3)


def test_corridor_ranges():
    r = Sonar(IDEAL).true_ranges(corridor(), RES, 5.0, 5.0, 0.0)
    assert r[FWD] == 10.0                       # nothing within max range ahead
    assert abs(r[LEFT] - 1.0) <= 0.1            # wall at y = 6
    assert abs(r[8] - 1.0) <= 0.1               # right beam, wall at y = 4
    assert abs(r[BACK] - 5.0) <= 0.1            # map edge behind counts as rock


def test_ranges_rotate_with_heading():
    r = Sonar(IDEAL).true_ranges(corridor(), RES, 5.0, 5.0, np.pi / 2)
    assert abs(r[FWD] - 1.0) <= 0.1


def test_noise_and_dropout_statistics():
    rng = np.random.default_rng(0)
    noisy = Sonar(replace(IDEAL, noise_std=0.05, noise_frac=0.02))
    left = np.array([noisy.measure(corridor(), RES, 5.0, 5.0, 0.0, rng)[LEFT] for _ in range(4000)])
    true = Sonar(IDEAL).true_ranges(corridor(), RES, 5.0, 5.0, 0.0)[LEFT]
    assert abs(left.mean() - true) < 0.01
    assert abs(left.std() - (0.05 + 0.02 * true)) < 0.01

    drop = Sonar(replace(IDEAL, dropout=0.3))
    out = np.array([drop.measure(corridor(), RES, 5.0, 5.0, 0.0, rng) for _ in range(500)])
    assert abs((out[:, LEFT] == 10.0).mean() - 0.3) < 0.05


def test_blind_zone_and_beam_width():
    r = Sonar(replace(IDEAL, min_range=2.0)).measure(corridor(), RES, 5.0, 5.0, 0.0, np.random.default_rng(0))
    assert r[LEFT] == 2.0
    narrow = Sonar(IDEAL).true_ranges(corridor(), RES, 5.0, 5.0, 0.0)
    wide = Sonar(replace(IDEAL, beam_width_deg=20)).true_ranges(corridor(), RES, 5.0, 5.0, 0.0)
    assert (wide <= narrow + 1e-9).all() and (wide < narrow).any()


# ---------------------------------------------------------------- dynamics

def run(action, n, cfg=VehicleConfig(), current=(0.0, 0.0), s=None):
    s = s or VehicleState(0.0, 0.0, 0.0)
    for _ in range(n):
        s = step_vehicle(s, action, cfg, current)
    return s


def test_rest_stays_put():
    s = run((0, 0), 100)
    assert (s.x, s.y, s.psi, s.u, s.r) == (0, 0, 0, 0, 0)


def test_speed_limits():
    assert abs(run((1, 0), 300).u - 1.5) < 1e-3            # full thrust settles at a_max / k_u
    assert abs(run((-1, 0), 300).u + 0.75) < 1e-3          # reverse is half strength
    assert run((1, 0), 300).x > 0 and run((-1, 0), 300).x < 0


def test_yaw_lag_and_direction():
    s = run((0, 1), 3)                                       # 0.3 s = one time constant
    assert 0.55 < s.r < 0.8                                  # about 1 - 1/e of r_max after one tau
    s = run((0, 1), 50)
    assert abs(s.r - 1.0) < 1e-3
    assert run((0, 1), 5).psi > 0                            # positive yaw command turns left (CCW)


def test_turning_radius():
    s = run((1, 0), 300)                                     # get up to speed
    xs, ys = [], []
    for _ in range(200):
        s = step_vehicle(s, (1, 1), VehicleConfig())
        xs.append(s.x); ys.append(s.y)
    radius = (max(xs) - min(xs)) / 2
    assert abs(radius - 1.5 / 1.0) < 0.1                     # u / r_max


def test_current_drift():
    s = run((0, 0), 100, current=(0.3, -0.1))                # 10 s
    assert np.isclose(s.x, 3.0) and np.isclose(s.y, -1.0)


# ---------------------------------------------------------------- simulator

def test_bilinear():
    yy, xx = np.mgrid[0:10, 0:10]
    grid = (2.0 * xx + 3.0 * yy).astype(float)               # linear in cell index
    x, y = 1.37, 0.91
    assert np.isclose(sample_bilinear(grid, x, y, RES), 2 * (x / RES - 0.5) + 3 * (y / RES - 0.5))
    grid[0, 0] = np.inf
    assert np.isfinite(sample_bilinear(grid, 0.15, 0.15, RES))
    grid[:] = np.inf
    assert sample_bilinear(grid, 0.5, 0.5, RES) == np.inf


def test_simulator_start_and_collision():
    cave = generate_cave(0, 0.0)
    sim = Simulator(cave, seed=0)
    st = sim.reset(current=(0.0, 0.0))
    assert not st.collided and st.clearance >= 0.4
    assert sim.geodesic_at(st.state.x, st.state.y) == pytest.approx(cave.shortest_path_length, abs=0.3)
    for _ in range(600):                                     # full thrust, no steering: the cave winds, so we hit a wall
        st = sim.step((1.0, 0.0))
        if st.collided:
            break
    assert st.collided and st.clearance < 0.4


def test_current_sampling():
    cave = generate_cave(0, 0.0)
    sim = Simulator(cave, VehicleConfig(max_current=0.3), seed=1)
    mags = [np.linalg.norm(sim.reset() and sim.current) for _ in range(200)]
    assert max(mags) <= 0.3 and min(mags) < 0.05 and max(mags) > 0.25
