import numpy as np
import pytest

from submarine.world import test_seeds as held_out_seeds
from submarine.world import CaveConfig, generate_cave, geodesic_distance, is_test_seed, train_seeds

DIFFICULTIES = [0.0, 0.25, 0.5, 0.75, 1.0]


def test_reproducible():
    a, b = generate_cave(7, 0.6), generate_cave(7, 0.6)
    assert np.array_equal(a.rock, b.rock)
    assert a.start == b.start and a.goal == b.goal


def test_seeds_differ():
    assert not np.array_equal(generate_cave(1, 0.5).rock, generate_cave(2, 0.5).rock)


@pytest.mark.parametrize("d", DIFFICULTIES)
def test_solvable_and_well_formed(d):
    for seed in range(20):
        cave = generate_cave(seed, d)
        w = cave.world
        assert cave.rock.shape == (w.ny, w.nx)
        # border is solid rock
        assert cave.rock[0].all() and cave.rock[-1].all() and cave.rock[:, 0].all() and cave.rock[:, -1].all()
        # start and goal are feasible and connected
        for p in (cave.start, cave.goal):
            assert cave.feasible[cave.world_to_cell(*p)]
        assert cave.geodesic[cave.world_to_cell(*cave.goal)] == 0
        L = cave.shortest_path_length
        straight = np.hypot(cave.goal[0] - cave.start[0], cave.goal[1] - cave.start[1])
        assert straight <= L < 3 * straight
        # clearance is zero in rock, feasible matches the vehicle radius
        assert (cave.clearance[cave.rock] == 0).all()
        assert np.array_equal(cave.feasible, cave.clearance >= w.vehicle_radius)


def test_hard_caves_have_cracks():
    caves = [generate_cave(s, 1.0) for s in range(20)]
    assert np.mean([len(c.cracks) for c in caves]) >= 2
    assert all(abs(c.cracks[0][2] - 1.7) < 1e-9 for c in caves if c.cracks)


def test_easy_caves_have_no_cracks_or_roughness():
    cfg = CaveConfig.from_difficulty(0.0)
    assert cfg.n_cracks_max == 0 and cfg.roughness == 0
    assert all(not generate_cave(s, 0.0).cracks for s in range(10))


def test_geodesic_matches_euclidean_in_open_space():
    feasible = np.ones((60, 100), dtype=bool)
    dist = geodesic_distance(feasible, (0, 0), 0.2)
    for iy, ix in [(0, 99), (59, 0), (59, 99), (45, 60), (30, 80)]:
        ratio = dist[iy, ix] / (0.2 * np.hypot(iy, ix))
        assert 1.0 - 1e-6 <= ratio < 1.03


def test_geodesic_goes_around_walls():
    feasible = np.ones((50, 50), dtype=bool)
    feasible[:40, 25] = False                      # wall with a gap at the top
    dist = geodesic_distance(feasible, (0, 0), 1.0)
    assert dist[0, 49] > 2 * 40                    # must detour through the gap
    assert np.isinf(dist[0, 25])


def test_seed_split_disjoint():
    tr, te = set(train_seeds(1000)), set(held_out_seeds(1000))
    assert not tr & te
    assert all(is_test_seed(s) for s in te) and not any(is_test_seed(s) for s in tr)
