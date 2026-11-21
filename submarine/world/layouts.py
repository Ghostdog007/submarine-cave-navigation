"""Additional cave layouts modelled on situations real cave explorers face.

The default "tunnel" layout (one winding passage) is built in generator.py and is
unchanged. The layouts here are carved from thick polyline segments ("passages")
and discs, with the same wall roughness, and finished exactly like tunnels
(clearance, start / goal, geodesic distance, solvability check).

  maze       labyrinth of junctions; a spanning tree, so one route and many dead ends
  branches   main passage with long, curving dead-end side passages, some heading
             toward the exit (the goal bearing lies)
  crack      glacier crevasse / ice fissure: angular zigzag, tight restrictions,
             side fissures, some too narrow to enter
  breakdown  boulder choke after a roof collapse: a wide chamber full of rubble

Memory scenarios (memory-v3): situations where the present observation misleads and only memory helps.
  decoy      the true route detours AWAY from the exit, while long dead-end passages (beyond sonar range)
             point straight AT it: the goal bearing recommends every decoy; only remembering one already tried
             avoids going back in
  loops      a maze with extra passages that close cycles: without memory an agent can circle indefinitely

Difficulty d in [0, 1] narrows passages, adds wiggle / roughness and clutter.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

LAYOUTS = ("tunnel", "maze", "branches", "crack", "breakdown", "decoy", "loops")


def _lerp(a, b, t):
    return a + (b - a) * t


def _path(p0, p1, rng, wiggle: float, step: float = 0.8) -> np.ndarray:
    """Points along p0 -> p1 with a smooth sideways wiggle that is zero at both ends."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.hypot(*d))
    n = max(2, int(L / step) + 1)
    t = np.linspace(0, 1, n)
    normal = np.array([-d[1], d[0]]) / max(L, 1e-9)
    off = np.zeros(n)
    for _ in range(2):
        k = rng.uniform(0.5, 2.0)
        off += rng.uniform(0.5, 1.0) * np.sin(np.pi * k * t + rng.uniform(0, 2 * np.pi))
    off *= wiggle * np.sin(np.pi * t)
    return p0 + t[:, None] * d + off[:, None] * normal


def _segments(points: np.ndarray, width: float, rng=None, vary: float = 0.0):
    out = []
    for a, b in zip(points[:-1], points[1:]):
        w = width * (1 + vary * rng.uniform(-1, 1)) if rng is not None and vary else width
        out.append((a[0], a[1], b[0], b[1], w))
    return out


def _carve(shape, res, segments, noise, rough):
    """Free-space mask: union of thick segments, walls perturbed by smooth noise."""
    ny, nx = shape
    free = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1, w in segments:
        amp = min(rough, 0.3 * w / 2)
        pad = w / 2 + amp + 0.3
        i0, i1 = max(int((min(x0, x1) - pad) / res), 0), min(int((max(x0, x1) + pad) / res) + 1, nx)
        j0, j1 = max(int((min(y0, y1) - pad) / res), 0), min(int((max(y0, y1) + pad) / res) + 1, ny)
        if i0 >= i1 or j0 >= j1:
            continue
        X = (np.arange(i0, i1) + 0.5) * res
        Y = (np.arange(j0, j1) + 0.5) * res
        X, Y = np.meshgrid(X, Y)
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1) if L2 > 0 else 0.0
        dist = np.hypot(X - (x0 + t * dx), Y - (y0 + t * dy))
        free[j0:j1, i0:i1] |= dist - w / 2 < amp * noise[j0:j1, i0:i1]
    return free


def _disc(x, y, r):
    return (x, y, x, y, 2 * r)          # zero-length segment = disc of radius r


# ---------------------------------------------------------------------------- layouts


def maze(rng, d, world, extra_edges: int = 0):
    """Spanning tree over a jittered 6 x 3 grid of junctions: exactly one route, the rest are dead ends.
    extra_edges > 0 adds that many non-tree passages, closing cycles (the "loops" layout)."""
    cols, rows = 6, 3
    xs = np.linspace(5.0, world.length - 5.0, cols)
    ys = np.linspace(7.0, world.height - 7.0, rows)
    node = {(c, r): (xs[c] + rng.uniform(-1.2, 1.2), ys[r] + rng.uniform(-1.2, 1.2))
            for c in range(cols) for r in range(rows)}
    r_start, r_goal = int(rng.integers(rows)), int(rng.integers(rows))

    # randomised Kruskal -> spanning tree with many short side branches (dead ends), like a real labyrinth
    cand = [((c, r), (c + dc, r + dr)) for (c, r) in node for dc, dr in ((1, 0), (0, 1)) if (c + dc, r + dr) in node]
    order = rng.permutation(len(cand))
    parent = {n: n for n in node}

    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    edges, spare = [], []
    for i in order:
        a, b = cand[i]
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb
            edges.append((a, b))
        else:
            spare.append((a, b))
    loops = spare[:extra_edges]                                  # random non-tree passages: each closes a cycle
    edges = edges + loops

    w = _lerp(3.4, 2.2, d)
    wiggle = _lerp(0.4, 1.2, d)
    segs = []
    for a, b in edges:
        segs += _segments(_path(node[a], node[b], rng, wiggle), w * rng.uniform(0.85, 1.15))
    for p in node.values():
        segs.append(_disc(*p, 0.65 * w))                      # junction chambers
    s, g = node[(0, r_start)], node[(cols - 1, r_goal)]
    segs += _segments(_path((1.0, s[1]), s, rng, 0.3), w)    # entrance and exit stubs
    segs += _segments(_path(g, (world.length - 1.0, g[1]), rng, 0.3), w)
    return segs, [], {"junctions": len(node), "dead_ends": _leaves(edges, (0, r_start), (cols - 1, r_goal)),
                      "loops": len(loops)}


def loops(rng, d, world):
    """Maze with 3-5 extra passages that close cycles."""
    return maze(rng, d, world, extra_edges=int(rng.integers(3, 6)))


def decoy(rng, d, world):
    """The route makes a long detour away from the exit; dead ends off it point straight at the exit.

    Main route: start -> the far edge of the cave (top or bottom, ~12 m from the goal's height) -> along it ->
    up/down to the goal. From the detour, 2-3 decoy passages leave toward the goal side (35-65 deg off the
    detour, so they are distinct passages) and end blind after 7-11 m. Decoys sit only where the goal bearing
    is closer to the decoy's direction than to the true route's, so a goal-following navigator enters each one."""
    L, H = world.length, world.height
    w = _lerp(3.4, 2.4, d)
    wiggle = _lerp(0.3, 0.8, d)
    y_goal = H / 2 + rng.uniform(-2.0, 2.0)
    side = rng.choice([-1.0, 1.0])
    far = H / 2 + side * (H / 2 - 3.5)                         # the detour runs along the top or bottom edge
    start, goal = np.array([1.0, y_goal + rng.uniform(-2, 2)]), np.array([L - 1.0, y_goal])
    way = [start, np.array([6.0, start[1]]), np.array([10.0, far]), np.array([L - 7.0, far]),
           np.array([L - 5.0, goal[1]]), goal]
    main = np.vstack([_path(a, b, rng, wiggle)[:-1] for a, b in zip(way[:-1], way[1:])] + [goal[None]])
    segs = _segments(main, w, rng, 0.1)
    n = 2 + int(rng.random() < 0.5 + 0.5 * d)
    made = 0
    for xj in np.linspace(19.0, 33.0, n) + rng.uniform(-1.0, 1.0, n):
        i = int(np.argmin(np.abs(main[:, 0] - xj) + 10.0 * (np.abs(main[:, 1] - far) > 2.0)))
        p = main[i]
        toward = np.sign(goal[1] - p[1])
        goal_ang = abs(np.arctan2(goal[1] - p[1], goal[0] - p[0]))   # route runs at ~0 deg
        ang = np.clip(max(goal_ang + rng.uniform(0.0, 0.35), np.deg2rad(35)), 0, np.deg2rad(65))
        if abs(goal_ang - ang) >= goal_ang:
            continue                                            # the goal bearing would not favour this decoy
        length = min(rng.uniform(8.0, 11.0), (L - 9.0 - p[0]) / np.cos(ang),
                     (abs(goal[1] - p[1]) + 1.0) / np.sin(ang))
        if length < 7.0:
            continue                                            # too short to hide its end from the sonar
        q = p + length * np.array([np.cos(ang), toward * np.sin(ang)])
        segs += _segments(_path(p, q, rng, 0.4), w * rng.uniform(0.85, 1.0), rng, 0.1)
        segs.append(_disc(*q, 0.6 * w))
        made += 1
    return segs, [], {"decoys": made}


def _leaves(edges, start, goal):
    deg = {}
    for a, b in edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return sum(1 for n, k in deg.items() if k == 1 and n not in (start, goal))


def branches(rng, d, world):
    """Main passage plus long curving dead-end side passages (beyond sonar range), some angled toward the exit."""
    L, H = world.length, world.height
    wx = np.linspace(1.0, L - 1.0, 5)
    wy = np.clip(H / 2 + rng.uniform(-7, 7, 5), 6, H - 6)
    w = _lerp(4.0, 2.4, d)
    main = np.vstack([_path((wx[i], wy[i]), (wx[i + 1], wy[i + 1]), rng, _lerp(0.8, 2.0, d))[:-1]
                      for i in range(4)] + [[[wx[-1], wy[-1]]]])
    segs = _segments(main, w, rng, 0.1)

    n_br = 3 + int(round(2 * d))
    usable = np.flatnonzero((main[:, 0] > 7) & (main[:, 0] < L - 12))
    picks = np.sort(rng.choice(usable, size=min(n_br, len(usable)), replace=False))
    deceptive = 0
    for i in picks:
        tangent = main[min(i + 1, len(main) - 1)] - main[max(i - 1, 0)]
        heading = np.arctan2(tangent[1], tangent[0])
        toward_exit = rng.random() < 0.5
        side = rng.choice([-1, 1])
        heading += side * (rng.uniform(0.6, 0.9) if toward_exit else rng.uniform(1.1, 1.6))
        deceptive += toward_exit
        p = main[i].copy()
        pts = [p.copy()]
        for _ in range(4):                                      # 4 legs of 3-4.5 m, gently curving
            heading += rng.uniform(-0.45, 0.45)
            q = p + rng.uniform(3.0, 4.5) * np.array([np.cos(heading), np.sin(heading)])
            if not (2.5 < q[0] < L - 11 and 2.5 < q[1] < H - 2.5):
                break
            pts.append(q)
            p = q
        if len(pts) < 3:
            continue
        branch = np.vstack([_path(a, b, rng, 0.3)[:-1] for a, b in zip(pts[:-1], pts[1:])] + [pts[-1][None]])
        segs += _segments(branch, w * rng.uniform(0.8, 1.05), rng, 0.1)
        segs.append(_disc(*pts[-1], 0.6 * w))                  # small chamber at the dead end
    return segs, [], {"dead_ends": len(picks), "toward_exit": int(deceptive)}


def crack(rng, d, world):
    """Angular zigzag fissure with 1-2 tight restrictions and short side fissures (some too narrow to enter)."""
    L, H = world.length, world.height
    w = _lerp(2.5, 1.8, d)
    w_squeeze = 1.4                                             # 0.8 m vehicle + ~0.3 m each side
    p = np.array([1.0, rng.uniform(8, H - 8)])
    pts, sign = [p.copy()], rng.choice([-1, 1])
    while p[0] < L - 1.0:
        ang = sign * rng.uniform(np.deg2rad(20), np.deg2rad(_lerp(35, 48, d)))
        q = p + rng.uniform(4.5, 7.5) * np.array([np.cos(ang), np.sin(ang)])
        q[1] = np.clip(q[1], 4.0, H - 4.0)
        q[0] = min(q[0], L - 1.0)
        pts.append(q)
        p, sign = q, -sign
    pts = np.array(pts)
    legs = len(pts) - 1
    inner = np.arange(1, max(legs - 1, 1))
    squeeze = set(rng.choice(inner, size=min(1 + int(d > 0.5), len(inner)), replace=False)) if len(inner) else set()
    segs = []
    for k in range(legs):
        segs.append((*pts[k], *pts[k + 1], w_squeeze if k in squeeze else w))
    for k in range(legs + 1):
        segs.append(_disc(*pts[k], 0.55 * w))                   # rounded kinks
    n_side = int(rng.integers(4, 8))
    for _ in range(n_side):                                     # side fissures off random legs
        k = int(rng.integers(legs))
        t = rng.uniform(0.2, 0.8)
        a = pts[k] + t * (pts[k + 1] - pts[k])
        leg = pts[k + 1] - pts[k]
        ang = np.arctan2(leg[1], leg[0]) + rng.choice([-1, 1]) * rng.uniform(1.0, 1.9)
        b = a + rng.uniform(2.0, 5.0) * np.array([np.cos(ang), np.sin(ang)])
        b = np.clip(b, 2.0, [L - 2.0, H - 2.0])
        segs.append((*a, *b, rng.uniform(0.7, 1.6)))            # < 1.0 m: visible, but too narrow to enter
    return segs, [], {"restrictions": len(squeeze), "side_fissures": n_side}


def breakdown(rng, d, world):
    """Wide chamber filled with rubble from a roof collapse; boulders may touch, leaving narrow gaps."""
    L, H = world.length, world.height
    y0, y1 = rng.uniform(10, H - 10, 2)
    hall = _path((1.0, y0), (L - 1.0, y1), rng, 2.0)
    w = _lerp(14.0, 11.0, d)
    segs = _segments(hall, w, rng, 0.1)
    n = int(_lerp(18, 38, d))
    boulders = []
    for _ in range(n * 20):
        if len(boulders) >= n:
            break
        i = int(rng.integers(len(hall)))
        if not 7.0 < hall[i, 0] < L - 7.0:
            continue
        x, y = hall[i] + rng.uniform(-w / 2, w / 2, 2)
        r = rng.uniform(0.5, _lerp(1.4, 2.0, d))
        if any(np.hypot(x - bx, y - by) < 0.6 * (r + br) for bx, by, br in boulders):
            continue                                            # allow touching piles, not full overlap
        boulders.append((float(x), float(y), float(r)))
    return segs, boulders, {"boulders": len(boulders)}


BUILDERS = {"maze": maze, "branches": branches, "crack": crack, "breakdown": breakdown, "decoy": decoy,
            "loops": loops}


def build_free(rng, cfg, world, noise):
    """Free-space mask, boulders (x, y, r) and layout statistics for a non-tunnel layout."""
    res = world.resolution
    segs, boulders, stats = BUILDERS[cfg.layout](rng, cfg.difficulty, world)
    rough = _lerp(0.05, 0.12, cfg.difficulty) if cfg.layout == "crack" else _lerp(0.1, 0.45, cfg.difficulty)
    free = _carve((world.ny, world.nx), res, segs, noise, rough)
    if boulders:
        xs = (np.arange(world.nx) + 0.5) * res
        ys = (np.arange(world.ny) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        for bx, by, br in boulders:
            # irregular rock: radius modulated by the roughness noise
            free &= np.hypot(X - bx, Y - by) >= br * (1 + 0.25 * noise)
    return free, boulders, stats


def heading_from_route(geodesic, cell, res):
    """Initial heading: along the shortest route out of the start cell."""
    from .geodesic import descend

    path = descend(geodesic, cell, res, max_length=2.0)
    (y0, x0), (y1, x1) = path[0], path[-1]
    return float(np.arctan2(y1 - y0, x1 - x0)) if len(path) > 1 else 0.0
