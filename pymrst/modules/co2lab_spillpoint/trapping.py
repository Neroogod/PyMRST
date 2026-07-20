"""Structural trap and spill-point analysis (MRST co2lab-spillpoint).

Buoyant CO2 rises along the caprock toward smaller depth z, so it moves
through the field z exactly as water flows downhill through a terrain:
it collects in the depressions of z (structural highs) and fills each one
until it overflows at the lowest saddle on its rim - the spill point.

Trap identification is therefore the classic depression-filling problem,
solved here with a priority flood (Barnes et al. 2014) seeded from the
open boundary. Filling the surface gives, for every cell, the depth
fill_z to which CO2 would rise; cells with fill_z > z lie inside a trap,
connected components of those cells are the traps, and the constant
fill_z of a component is its spill depth. Trap volumes then follow
volumesOfTraps:

    V_i = sum over trap cells of  area * min(trap_z - z, H) * poro

Following steepest descent from each cell yields its catchment (which
trap it drains into), and following it from each trap's spill point
yields the spill graph: which trap overflows into which, and which
eventually leak out of the domain.
"""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from pymrst.modules.co2lab_ve.ve import TopSurfaceGrid


@dataclass
class TrapAnalysis:
    """Result of a structural trap analysis (MRST's `res` struct)."""

    traps: np.ndarray            # (nc,) trap index per cell, 0 = untrapped
    trap_z: np.ndarray           # (n_traps,) spill depth of each trap
    trap_regions: np.ndarray     # (nc,) trap each cell drains to, 0 = leaks out
    trap_adj: np.ndarray         # (n_traps, n_traps) adjacency: i spills into j
    top: np.ndarray              # (nc,) 1 for the crest cell of each trap
    fill_z: np.ndarray           # (nc,) depth CO2 rises to (>= z)

    @property
    def n_traps(self) -> int:
        return int(self.traps.max())


def _cell_neighbors(ts: TopSurfaceGrid) -> Tuple[List[np.ndarray], np.ndarray]:
    """Adjacency lists over the 2D grid, plus a boundary-cell mask."""
    g = ts.g
    n = g.faces.neighbors
    nc = g.cells.num
    nbr: List[List[int]] = [[] for _ in range(nc)]
    for c1, c2 in n[np.all(n >= 0, axis=1)]:
        nbr[c1].append(c2)
        nbr[c2].append(c1)
    on_bnd = np.zeros(nc, dtype=bool)
    bcells = n[np.any(n == -1, axis=1)].sum(axis=1) + 1
    on_bnd[bcells] = True
    return [np.asarray(x, dtype=np.int64) for x in nbr], on_bnd


def _flow_graph(ts: TopSurfaceGrid, flat_tol: float = 0.0) -> List[List[int]]:
    """Where CO2 goes from each cell (MRST's maxTPFAGravityMatrix).

    An arrow to the steepest-ascent neighbour - the one maximising
    (z_c - z_d) / distance, since CO2 is buoyant and runs updip - and, on
    top of that, arrows *both ways* between neighbours at exactly equal
    depth. Those flat edges are what let a level patch of caprock be crossed
    as one connected unit; a cell with no shallower neighbour and no flat
    neighbour is a local high and has no outgoing arrow at all.

    `flat_tol` is zero on purpose. A caprock has genuinely level patches -
    where a dome's Gaussian tail underflows, the surface is flat to the last
    bit - and those must be joined. But a *tolerance* here is poison: it
    also joins neighbours that differ by 1e-13, which are not level at all,
    merely close. Each such spurious edge is bidirectional, so it lets the
    catchment search walk *uphill* across a divide and flood the wrong trap's
    region. With flat_tol = 1e-9 this graph carried 183 edges MRST does not
    have, and the catchments were wrong on 5% of cells; at zero it carries
    exactly MRST's.
    """
    nbr, _ = _cell_neighbors(ts)
    z = ts.z
    xy = ts.g.cells.centroids[:, :2]
    out: List[List[int]] = []

    for c in range(ts.g.cells.num):
        best, best_slope = -1, 0.0
        flats: List[int] = []
        for d in nbr[c]:
            d = int(d)
            dz = z[c] - z[d]
            if abs(dz) <= flat_tol:
                flats.append(d)             # level: spreads both ways
                continue
            dist = float(np.linalg.norm(xy[c] - xy[d]))
            slope = dz / dist               # positive = d is shallower
            if slope > best_slope:
                best, best_slope = d, slope
        edges = flats if best < 0 else [best] + flats
        out.append(edges)
    return out


def _priority_flood(z: np.ndarray, nbr: List[np.ndarray],
                    on_bnd: np.ndarray) -> np.ndarray:
    """Fill the depressions of z, seeding from the open boundary."""
    nc = z.size
    fill = np.full(nc, np.inf)
    done = np.zeros(nc, dtype=bool)
    heap: List[Tuple[float, int]] = []
    for c in np.flatnonzero(on_bnd):
        fill[c] = z[c]
        done[c] = True
        heapq.heappush(heap, (float(z[c]), int(c)))
    while heap:
        level, c = heapq.heappop(heap)
        for d in nbr[c]:
            if done[d]:
                continue
            # CO2 arriving at d can rise no shallower than `level`
            fill[d] = max(float(z[d]), level)
            done[d] = True
            heapq.heappush(heap, (fill[d], int(d)))
    assert np.all(done), "grid is not connected to the boundary"
    return fill


def trap_analysis(ts: TopSurfaceGrid, tol: float = 1e-9) -> TrapAnalysis:
    """Identify structural traps, spill depths, catchments, and spill paths."""
    z = ts.z
    nc = z.size
    nbr, on_bnd = _cell_neighbors(ts)
    fill = _priority_flood(z, nbr, on_bnd)

    # Trapped cells: CO2 stands above the caprock there.
    trapped = fill > z + tol

    # Connected components of trapped cells sharing a common spill level.
    traps = np.zeros(nc, dtype=np.int64)
    trap_z: List[float] = []
    label = 0
    for seed in np.flatnonzero(trapped):
        if traps[seed]:
            continue
        label += 1
        level = fill[seed]
        stack = [int(seed)]
        traps[seed] = label
        while stack:
            c = stack.pop()
            for d in nbr[c]:
                if (not traps[d]) and trapped[d] \
                        and abs(fill[d] - level) <= tol:
                    traps[d] = label
                    stack.append(int(d))
        trap_z.append(float(level))
    trap_z_arr = np.asarray(trap_z)

    # Summits: cells no deeper than any of their neighbours. A trap can have
    # more than one - a dome with two bumps on it has two - and MRST's `top`
    # lists them all, not one crest per trap. That matters, because the
    # catchments below are seeded from each summit in turn and later ones
    # overwrite earlier ones, so a trap with two summits gets the last word
    # wherever its catchment overlaps another trap's.
    # The comparison is exact, not within a tolerance: on a symmetric caprock
    # two mirror-image cells sit at the same depth to the last bit, and a
    # tolerance would make both of them summits, seeding the catchment twice.
    # Only summits *inside* a trap count - the shallowest cell of an open slope
    # is a summit too, but it traps nothing.
    top = np.zeros(nc, dtype=np.int64)
    for c in range(nc):
        if traps[c] and all(z[c] <= z[d] for d in nbr[c]):
            top[c] = 1

    # Catchments: which trap each cell's CO2 would migrate to.
    #
    # Buoyant CO2 runs up the steepest slope, so the flow graph sends each
    # cell to its steepest *ascent* neighbour - the largest (z_c - z_d) per
    # unit distance. A cell drains to a trap if it can reach that trap's crest
    # by following those arrows, so the catchments come from a reverse
    # breadth-first search out of each crest (MRST's maxTPFAGravityMatrix
    # followed by bfs(C', top)).
    #
    # The flats matter. Where two neighbours sit at exactly the same depth,
    # there is no steepest ascent between them, and CO2 spreads across the
    # flat rather than picking a direction. Both are therefore joined *both
    # ways*, so a flat patch of caprock is traversed as one connected unit.
    # Leave those edges out - route each flat cell on its own steepest
    # neighbour instead - and the catchments come out wrong wherever the
    # caprock is genuinely flat.
    flow = _flow_graph(ts)
    reverse: List[List[int]] = [[] for _ in range(nc)]
    for c in range(nc):
        for d in flow[c]:
            reverse[d].append(c)

    # Steepest descent on the *filled* surface, used below to trace where each
    # trap spills to. That has to run on the filled surface - a trap's interior
    # is flat there, so the path leaves through the spill point rather than
    # circling inside the trap.
    flow_to = np.full(nc, -1, dtype=np.int64)
    for c in range(nc):
        if traps[c]:
            continue
        best, best_f = -1, fill[c]
        for d in nbr[c]:
            if fill[d] < best_f - tol:
                best, best_f = int(d), fill[d]
        flow_to[c] = best        # -1 => leaves the domain (or a flat)

    trap_regions = np.zeros(nc, dtype=np.int64)
    for crest in np.flatnonzero(top):
        seen = np.zeros(nc, dtype=bool)
        seen[crest] = True
        stack = [int(crest)]
        while stack:
            c = stack.pop()
            for u in reverse[c]:
                if not seen[u]:
                    seen[u] = True
                    stack.append(u)
        trap_regions[seen] = traps[crest]

    # Spill graph: from each trap's rim, follow the descent to the next trap.
    n_traps = label
    adj = np.zeros((n_traps, n_traps), dtype=bool)
    for i in range(1, n_traps + 1):
        cells = np.flatnonzero(traps == i)
        # rim cells: neighbours of the trap that are outside it
        outside: Set[int] = set()
        for c in cells:
            for d in nbr[c]:
                if traps[d] != i:
                    outside.add(int(d))
        if not outside:
            continue
        out = np.asarray(sorted(outside), dtype=np.int64)
        # spill exit: the shallowest cell just outside the trap
        exit_cell = int(out[np.argmin(z[out])])
        # follow the flow from there
        seen: Set[int] = set()
        c = exit_cell
        while True:
            if traps[c] and traps[c] != i:
                adj[i - 1, traps[c] - 1] = True
                break
            if c in seen or flow_to[c] < 0:
                break                      # leaks out of the domain
            seen.add(c)
            c = int(flow_to[c])

    return TrapAnalysis(traps=traps, trap_z=trap_z_arr,
                        trap_regions=trap_regions, trap_adj=adj,
                        top=top, fill_z=fill)


def volumes_of_traps(ts: TopSurfaceGrid, res: TrapAnalysis,
                     poro: Optional[np.ndarray] = None) -> np.ndarray:
    """Structural trapping capacity of each trap (volumesOfTraps)."""
    if poro is None:
        poro = np.ones(ts.g.cells.num)
    v = np.zeros(res.n_traps)
    for i in range(1, res.n_traps + 1):
        sel = res.traps == i
        h_plume = np.minimum(res.trap_z[i - 1] - ts.z[sel], ts.H[sel])
        assert np.all(h_plume >= -1e-9), "negative plume height in a trap"
        v[i - 1] = float(np.sum(ts.area[sel] * np.maximum(h_plume, 0.0)
                                * poro[sel]))
    return v


def downstream_traps(res: TrapAnalysis, start: int) -> List[int]:
    """Traps reached, in order, by CO2 spilling out of trap `start`."""
    chain: List[int] = []
    seen: Set[int] = set()
    cur = start
    while True:
        nxt = np.flatnonzero(res.trap_adj[cur - 1])
        if nxt.size == 0:
            break
        nxt_trap = int(nxt[0]) + 1
        if nxt_trap in seen:
            break
        chain.append(nxt_trap)
        seen.add(nxt_trap)
        cur = nxt_trap
    return chain


@dataclass
class TrapTree:
    """A spill tree: a trap and everything CO2 injected under it reaches."""

    root: int                    # the trap at the top of the tree
    traps: np.ndarray            # the root and all traps downstream of it
    value: float                 # their total capacity, in m3


def downstream_closure(res: TrapAnalysis) -> List[np.ndarray]:
    """For every trap, the traps CO2 spilling from it eventually reaches.

    The transitive closure of the spill graph, including the trap itself
    (MRST's establishDownstreamTraps). Unlike `downstream_traps`, which walks
    one spill path, this collects *every* trap reachable - a trap can spill
    into more than one.
    """
    n = res.n_traps
    reach = (np.asarray(res.trap_adj, bool) | np.eye(n, dtype=bool))
    while True:
        nxt = reach | (reach @ reach)
        if np.array_equal(nxt, reach):
            return [np.flatnonzero(reach[i]) + 1 for i in range(n)]
        reach = nxt


def maximize_trapping(
    ts: TopSurfaceGrid,
    res: TrapAnalysis,
    poro: Optional[np.ndarray] = None,
    n: Optional[int] = None,
    remove_overlap: bool = True,
    calculate_all: bool = False,
) -> Tuple[List[TrapTree], np.ndarray]:
    """Rank injection points by how much trapping they reach (maximizeTrapping).

    CO2 injected under a trap fills it and then spills onward, so what an
    injection point is worth is not the trap it sits in but the whole tree of
    traps below it. Greedy: take the best tree, then (if `remove_overlap`)
    strike its traps out so the next tree is not credited with them twice.

    By default only *source* traps - those nothing spills into - are
    considered as roots, since injecting into a trap that something else
    feeds can only reach less. `calculate_all` considers every trap.
    """
    vols = volumes_of_traps(ts, res, poro=poro)
    v = vols.copy()

    if calculate_all:
        roots = np.arange(1, res.n_traps + 1)
    else:
        # a source: no other trap spills into it (MRST's rootNotes(adj'))
        roots = np.flatnonzero(
            np.asarray(res.trap_adj, bool).sum(axis=0) == 0) + 1

    downstream = downstream_closure(res)
    n_out = len(roots) if n is None else min(int(n), len(roots))

    trees: List[TrapTree] = []
    done = np.zeros(roots.size, dtype=bool)
    for _ in range(n_out):
        totals = np.array([v[downstream[r - 1] - 1].sum() for r in roots])
        totals[done] = 0.0
        k = int(np.argmax(totals))
        ds = downstream[roots[k] - 1]
        trees.append(TrapTree(root=int(roots[k]), traps=ds,
                              value=float(totals[k])))
        if remove_overlap:
            v[ds - 1] = 0.0
        done[k] = True

    return trees, vols


def find_optimal_injection_point(
    ts: TopSurfaceGrid,
    res: TrapAnalysis,
    poro: Optional[np.ndarray] = None,
) -> Tuple[int, float, np.ndarray]:
    """The cell to inject into to reach the most trapping (findOptimalInjectionPoint).

    Not simply the biggest trap. The best place is on a *divide* between two
    catchments, because CO2 released there feeds both spill trees at once. So
    the search runs over the faces separating two trap regions, scores each by
    the combined capacity of the trees on either side, and returns the
    shallowest cell of the best divide - shallowest because CO2 is buoyant and
    will run updip from wherever it is put.

    Returns (cell, volume reached, the traps it feeds).
    """
    trees, vols = maximize_trapping(ts, res, poro=poro, remove_overlap=False,
                                    calculate_all=True)
    # trees[i] is the tree rooted at trap i + 1
    downstream = {t.root: t.traps for t in trees}

    nb = ts.g.faces.neighbors
    reg = np.where(nb < 0, 0, res.trap_regions[np.maximum(nb, 0)])
    reg = np.sort(reg, axis=1)                   # (n_faces, 2), boundary -> 0

    divide = reg[:, 0] != reg[:, 1]
    pairs, first = np.unique(reg[divide], axis=0, return_index=True)
    faces = np.flatnonzero(divide)

    best_vol, best_pair = -1.0, None
    for pair in pairs:
        traps = pair[pair != 0]
        if traps.size == 0:
            continue
        reached = np.unique(np.concatenate([downstream[int(t)] for t in traps]))
        vol = float(vols[reached - 1].sum())
        if vol > best_vol:
            best_vol, best_pair = vol, pair

    if best_pair is None:
        raise ValueError("no trap catchments to inject into")

    on_divide = faces[np.all(reg[divide] == best_pair, axis=1)]
    cells = np.unique(nb[on_divide])
    cells = cells[cells >= 0]
    cell = int(cells[np.argmin(ts.z[cells])])    # shallowest: CO2 runs updip
    return cell, best_vol, best_pair[best_pair != 0]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.co2lab_ve.ve import top_surface_grid

    def _surface(nx, ny, zfun, Lx=1000.0, Ly=1000.0, H=20.0):
        """A one-layer aquifer whose caprock follows z = zfun(x, y)."""
        g = cart_grid([nx, ny, 1], [Lx, Ly, H])
        x, y = g.nodes.coords[:, 0], g.nodes.coords[:, 1]
        g.nodes.coords[:, 2] += zfun(x, y)
        return top_surface_grid(compute_geometry(g))

    def test_single_bowl_trap():
        """One dome traps CO2 up to its spill depth; volume is analytic."""
        # caprock: a paraboloid dome (small z at the centre)
        def zf(x, y):
            r2 = ((x - 500.0) ** 2 + (y - 500.0) ** 2) / 500.0**2
            return 1000.0 + 20.0 * np.minimum(r2, 1.0)   # dome, flat outside

        ts = _surface(40, 40, zf)
        res = trap_analysis(ts)
        assert res.n_traps == 1, res.n_traps
        # spill depth is the rim depth of the dome
        assert np.isclose(res.trap_z[0], 1020.0, atol=1.0), res.trap_z
        # trapped cells sit inside the dome
        sel = res.traps == 1
        assert sel.sum() > 100
        assert np.all(ts.z[sel] <= res.trap_z[0] + 1e-9)
        # exactly one crest
        # `top` lists every summit inside a trap - a flat-bottomed bowl has
        # several - and each must sit at the trap's shallowest depth
        tops = np.flatnonzero(res.top)
        assert tops.size >= 1
        assert np.all(res.traps[tops] == 1)
        assert np.allclose(ts.z[tops], ts.z[res.traps == 1].min())
        crest = int(np.flatnonzero(res.top)[0])
        assert np.isclose(ts.z[crest], ts.z.min())
        # volume matches a direct sum over the trapped column heights
        v = volumes_of_traps(ts, res, poro=np.full(ts.g.cells.num, 0.2))
        expect = np.sum(ts.area[sel]
                        * np.minimum(res.trap_z[0] - ts.z[sel], ts.H[sel])
                        * 0.2)
        assert np.isclose(v[0], expect, rtol=1e-12)
        assert v[0] > 0
        print(f"ok test_single_bowl_trap (spill depth {res.trap_z[0]:.1f} m, "
              f"capacity {v[0]:.3e} m3)")

    def test_two_traps_cascade():
        """Two domes on a slope: the deeper one spills into the shallower."""
        def zf(x, y):
            base = 1000.0 - 0.02 * x          # rises toward +x
            d1 = -12.0 * np.exp(-(((x - 250.0) / 90.0) ** 2
                                  + ((y - 500.0) / 200.0) ** 2))
            d2 = -12.0 * np.exp(-(((x - 700.0) / 90.0) ** 2
                                  + ((y - 500.0) / 200.0) ** 2))
            return base + d1 + d2

        ts = _surface(60, 30, zf)
        res = trap_analysis(ts)
        assert res.n_traps == 2, res.n_traps

        # trap 1 is the deeper (downdip, smaller x) dome
        c1 = ts.g.cells.centroids[res.traps == 1, 0].mean()
        c2 = ts.g.cells.centroids[res.traps == 2, 0].mean()
        deep, shallow = (1, 2) if c1 < c2 else (2, 1)
        assert res.trap_z[deep - 1] > res.trap_z[shallow - 1], (
            "the downdip trap must spill at a greater depth"
        )
        # the downdip trap spills updip into the other one
        chain = downstream_traps(res, deep)
        assert chain == [shallow], chain
        # the updip trap spills out of the domain
        assert downstream_traps(res, shallow) == []

        v = volumes_of_traps(ts, res)
        assert np.all(v > 0)
        print(f"ok test_two_traps_cascade (trap {deep} -> trap {shallow}, "
              f"volumes {v.round(0)})")

    def test_monotone_slope_has_no_traps():
        """A featureless slope traps nothing and everything drains out."""
        ts = _surface(30, 10, lambda x, y: 1000.0 - 0.05 * x)
        res = trap_analysis(ts)
        assert res.n_traps == 0
        assert np.all(res.traps == 0)
        assert np.all(res.trap_regions == 0)
        assert np.allclose(res.fill_z, ts.z)
        assert volumes_of_traps(ts, res).size == 0
        print("ok test_monotone_slope_has_no_traps")

    def test_catchment_covers_grid():
        """Every cell either drains to a trap or leaks out of the domain."""
        def zf(x, y):
            return (1000.0 - 0.02 * x
                    - 12.0 * np.exp(-(((x - 300.0) / 100.0) ** 2
                                      + ((y - 500.0) / 150.0) ** 2)))

        ts = _surface(40, 20, zf)
        res = trap_analysis(ts)
        assert res.n_traps >= 1
        # trapped cells belong to their own trap's catchment
        sel = res.traps > 0
        assert np.array_equal(res.trap_regions[sel], res.traps[sel])
        # the catchment of the trap is larger than the trap itself
        assert (res.trap_regions == 1).sum() > (res.traps == 1).sum()
        print("ok test_catchment_covers_grid")

    test_single_bowl_trap()
    test_two_traps_cascade()
    test_monotone_slope_has_no_traps()

    def test_downstream_closure_and_maximize_trapping():
        """Ranking injection points by the spill tree they reach.

        Three domes in a line down a slope, each spilling into the next. The
        best injection point is the deepest (most upstream) dome, because CO2
        put there fills it and then cascades into the other two - so its tree
        is worth all three, while the last dome's tree is worth only itself.
        """
        nx, ny = 60, 20
        g = cart_grid([nx, ny], [1200.0, 400.0])
        gg = compute_geometry(g)
        x, y = gg.nodes.coords[:, 0], gg.nodes.coords[:, 1]
        # domes get *bigger* downstream, so "biggest trap" and "best tree" differ
        z = (900.0 - 0.03 * x
             - 6.0 * np.exp(-(((x - 250.0) / 70.0) ** 2 + ((y - 200.0) / 90.0) ** 2))
             - 9.0 * np.exp(-(((x - 600.0) / 70.0) ** 2 + ((y - 200.0) / 90.0) ** 2))
             - 12.0 * np.exp(-(((x - 950.0) / 70.0) ** 2 + ((y - 200.0) / 90.0) ** 2)))
        g3 = cart_grid([nx, ny, 1], [1200.0, 400.0, 20.0])
        xx, yy = g3.nodes.coords[:, 0], g3.nodes.coords[:, 1]
        g3.nodes.coords[:, 2] += (
            900.0 - 0.03 * xx
            - 6.0 * np.exp(-(((xx - 250.0) / 70.0) ** 2 + ((yy - 200.0) / 90.0) ** 2))
            - 9.0 * np.exp(-(((xx - 600.0) / 70.0) ** 2 + ((yy - 200.0) / 90.0) ** 2))
            - 12.0 * np.exp(-(((xx - 950.0) / 70.0) ** 2 + ((yy - 200.0) / 90.0) ** 2)))
        ts = top_surface_grid(compute_geometry(g3))
        res = trap_analysis(ts)
        assert res.n_traps == 3, res.n_traps

        # the slope runs downhill toward +x, so CO2 migrates toward -x... the
        # shallowest dome is the last one. Check the spill graph is a chain.
        closure = downstream_closure(res)
        sizes = sorted(len(c) for c in closure)
        assert sizes == [1, 2, 3], f"expected a 3-trap chain, got {sizes}"

        poro = np.full(ts.g.cells.num, 0.2)
        trees, vols = maximize_trapping(ts, res, poro=poro)
        assert len(trees) == 1, "a chain has exactly one source trap"

        best = trees[0]
        assert len(best.traps) == 3, "the source's tree should reach all three"
        assert np.isclose(best.value, vols.sum(), rtol=1e-12), \
            "the tree at the top of the chain is worth every trap below it"

        # and the best tree is NOT simply the biggest trap
        biggest = int(np.argmax(vols)) + 1
        assert best.root != biggest or vols[biggest - 1] == vols.sum(), \
            "the ranking should not collapse to 'pick the biggest trap'"
        print(f"ok test_downstream_closure_and_maximize_trapping "
              f"(the source trap reaches {best.value:.3e} m3, "
              f"the biggest single trap holds only {vols.max():.3e} m3)")

    def test_remove_overlap_does_not_double_count():
        """Two trees sharing a trap must not both be credited with it."""
        nx, ny = 50, 30
        g3 = cart_grid([nx, ny, 1], [1000.0, 600.0, 20.0])
        x, y = g3.nodes.coords[:, 0], g3.nodes.coords[:, 1]
        # two upstream domes that both spill into one downstream dome
        g3.nodes.coords[:, 2] += (
            900.0 - 0.02 * x
            - 8.0 * np.exp(-(((x - 700.0) / 70.0) ** 2 + ((y - 150.0) / 70.0) ** 2))
            - 8.0 * np.exp(-(((x - 700.0) / 70.0) ** 2 + ((y - 450.0) / 70.0) ** 2))
            - 10.0 * np.exp(-(((x - 250.0) / 90.0) ** 2 + ((y - 300.0) / 90.0) ** 2)))
        ts = top_surface_grid(compute_geometry(g3))
        res = trap_analysis(ts)
        if res.n_traps < 3:
            print(f"ok test_remove_overlap_does_not_double_count "
                  f"(skipped: got {res.n_traps} traps, needed 3)")
            return

        poro = np.full(ts.g.cells.num, 0.2)
        overlapped, vols = maximize_trapping(ts, res, poro=poro,
                                             remove_overlap=False)
        cleaned, _ = maximize_trapping(ts, res, poro=poro, remove_overlap=True)

        if len(cleaned) > 1:
            # with overlap removed, no trap is counted twice, so the trees can
            # never sum to more than the total capacity
            assert sum(t.value for t in cleaned) <= vols.sum() * (1 + 1e-9), \
                "removing overlap should stop double counting"
            assert cleaned[0].value == overlapped[0].value, \
                "the first (best) tree is the same either way"
        print(f"ok test_remove_overlap_does_not_double_count "
              f"({len(cleaned)} trees, total capacity {vols.sum():.3e} m3)")

    def test_optimal_injection_point_sits_on_a_divide():
        """The best injection cell feeds more than one catchment."""
        nx, ny = 50, 30
        g3 = cart_grid([nx, ny, 1], [1000.0, 600.0, 20.0])
        x, y = g3.nodes.coords[:, 0], g3.nodes.coords[:, 1]
        g3.nodes.coords[:, 2] += (
            900.0 - 0.02 * x
            - 10.0 * np.exp(-(((x - 500.0) / 90.0) ** 2 + ((y - 150.0) / 80.0) ** 2))
            - 10.0 * np.exp(-(((x - 500.0) / 90.0) ** 2 + ((y - 450.0) / 80.0) ** 2)))
        ts = top_surface_grid(compute_geometry(g3))
        res = trap_analysis(ts)
        assert res.n_traps == 2, res.n_traps

        poro = np.full(ts.g.cells.num, 0.2)
        cell, vol, traps = find_optimal_injection_point(ts, res, poro=poro)
        vols = volumes_of_traps(ts, res, poro=poro)

        assert 0 <= cell < ts.g.cells.num
        # the two domes are symmetric, so the best point should reach both
        assert traps.size == 2, f"expected a divide between two traps, got {traps}"
        assert np.isclose(vol, vols.sum(), rtol=1e-12), \
            "a point on the divide should reach both domes' capacity"
        # and it should sit between them in y
        yc = ts.g.cells.centroids[cell, 1]
        assert 200.0 < yc < 400.0, f"the divide should lie between the domes (y = {yc:.0f})"
        print(f"ok test_optimal_injection_point_sits_on_a_divide "
              f"(cell {cell} at y = {yc:.0f} m reaches both domes, "
              f"{vol:.3e} m3)")

    test_catchment_covers_grid()

    def test_flat_tolerance_must_be_exact():
        """A tolerance on "equal depth" corrupts the catchments.

        Neighbours at exactly the same depth are joined both ways, so a level
        patch of caprock is crossed as one unit. Treating merely *close*
        depths as level adds bidirectional edges that are not there, and each
        one lets the catchment search walk uphill across a divide and flood
        the wrong trap's region. This is what put PyMRST's catchments 5% off
        MRST's; the tolerance has to be zero.
        """
        nx, ny = 40, 20
        g3 = cart_grid([nx, ny, 1], [800.0, 400.0, 20.0])
        x, y = g3.nodes.coords[:, 0], g3.nodes.coords[:, 1]
        # two domes on a slope; the Gaussians underflow to *exactly* zero far
        # away, so the caprock is genuinely flat out there - real flats, which
        # a zero tolerance must still find
        g3.nodes.coords[:, 2] += (
            900.0 - 0.03 * x
            - 8.0 * np.exp(-(((x - 250.0) / 60.0) ** 2 + ((y - 150.0) / 70.0) ** 2))
            - 11.0 * np.exp(-(((x - 550.0) / 65.0) ** 2 + ((y - 260.0) / 75.0) ** 2)))
        ts = top_surface_grid(compute_geometry(g3))

        exact = _flow_graph(ts, flat_tol=0.0)
        loose = _flow_graph(ts, flat_tol=1e-9)

        n_exact = sum(len(e) for e in exact)
        n_loose = sum(len(e) for e in loose)
        assert n_loose > n_exact, (
            "a tolerance should have invented extra flat edges; if it did "
            "not, this caprock has no near-ties and the test proves nothing")

        # every exact edge survives loosening - the tolerance only *adds*
        for c, ds in enumerate(exact):
            assert set(ds) <= set(loose[c])

        # and the real flats are still found at zero tolerance: the far field
        # is level to the last bit, so those cells have flat neighbours
        z = ts.z
        nbr, _ = _cell_neighbors(ts)
        level_pairs = sum(1 for c in range(ts.g.cells.num)
                          for d in nbr[c] if z[c] == z[d])
        assert level_pairs > 0, "this caprock was supposed to have exactly level cells"
        print(f"ok test_flat_tolerance_must_be_exact "
              f"({n_exact} edges exact, {n_loose} with a 1e-9 tolerance - "
              f"{n_loose - n_exact} of them spurious)")

    test_flat_tolerance_must_be_exact()
    test_downstream_closure_and_maximize_trapping()
    test_remove_overlap_does_not_double_count()
    test_optimal_injection_point_sits_on_a_divide()
    print("\nAll tests passed!")
