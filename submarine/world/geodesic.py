"""Geodesic (shortest path through feasible space) distance on the grid.

Dijkstra on a 16-connected grid (orthogonal, diagonal and knight moves), which
keeps the error versus true Euclidean path length under about 3 percent
(8-connected would be up to about 8 percent). A move is only allowed if the
cells it passes through are also feasible, so paths never cut rock corners.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# (dy, dx) -> intermediate cells that must also be feasible. Only half of the
# moves are listed; the graph is treated as undirected.
_MOVES: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 1): (),
    (1, 0): (),
    (1, 1): ((0, 1), (1, 0)),
    (1, -1): ((0, -1), (1, 0)),
    (1, 2): ((0, 1), (1, 1)),
    (2, 1): ((1, 0), (1, 1)),
    (1, -2): ((0, -1), (1, -1)),
    (2, -1): ((1, 0), (1, -1)),
}


_NEIGHBOURS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def descend(field: np.ndarray, cell: tuple[int, int], resolution: float,
            max_length: float = np.inf) -> list[tuple[int, int]]:
    """Follow steepest descent on a distance field from `cell` toward its zero.

    Returns the visited cells (including the start), stopping at the target,
    at a local minimum, or after `max_length` metres.
    """
    ny, nx = field.shape
    iy, ix = cell
    path = [(iy, ix)]
    travelled = 0.0
    while travelled < max_length and field[iy, ix] > 0:
        best, step = field[iy, ix], None
        for dy, dx in _NEIGHBOURS_8:
            jy, jx = iy + dy, ix + dx
            if 0 <= jy < ny and 0 <= jx < nx and field[jy, jx] < best:
                best, step = field[jy, jx], (dy, dx)
        if step is None:
            break
        iy, ix = iy + step[0], ix + step[1]
        travelled += resolution * float(np.hypot(*step))
        path.append((iy, ix))
    return path


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """out[y, x] = a[y + dy, x + dx], False outside the array."""
    ny, nx = a.shape
    out = np.zeros_like(a)
    out[max(-dy, 0):ny + min(-dy, 0), max(-dx, 0):nx + min(-dx, 0)] = \
        a[max(dy, 0):ny + min(dy, 0), max(dx, 0):nx + min(dx, 0)]
    return out


def geodesic_distance(feasible: np.ndarray, target: tuple[int, int], resolution: float,
                      cost: np.ndarray | None = None) -> np.ndarray:
    """Distance in metres from every cell to `target` (iy, ix) through feasible cells.

    Infeasible or unreachable cells get +inf. With `cost` (per-cell multiplier
    >= 1), each move is weighted by the mean cost of its two end cells, which
    gives a cost-weighted distance (used by the oracle to keep away from walls).
    """
    ny, nx = feasible.shape
    if not feasible[target]:
        raise ValueError("target cell is not feasible")
    idx = np.arange(ny * nx).reshape(ny, nx)

    src, dst, weight = [], [], []
    for (dy, dx), mids in _MOVES.items():
        ok = feasible & _shift(feasible, dy, dx)
        for my, mx in mids:
            ok &= _shift(feasible, my, mx)
        s = idx[ok]
        d = s + dy * nx + dx
        w = np.full(s.size, resolution * np.hypot(dy, dx))
        if cost is not None:
            flat = cost.ravel()
            w = w * 0.5 * (flat[s] + flat[d])
        src.append(s)
        dst.append(d)
        weight.append(w)

    graph = csr_matrix(
        (np.concatenate(weight), (np.concatenate(src), np.concatenate(dst))),
        shape=(ny * nx, ny * nx),
    )
    dist = dijkstra(graph, directed=False, indices=target[0] * nx + target[1])
    return dist.reshape(ny, nx).astype(np.float32)
