"""Node-based trap analysis (MRST co2lab-spillpoint/trapping/node_based).

The cell-based analysis in `trapping.py` puts the caprock's depth at cell
centres and finds the depressions of that field. This one puts it at the
grid's *nodes*, so a trap's rim can fall between cells rather than being
forced onto a cell edge - which matters on a coarse grid, where a whole cell
is a poor place to have to decide "in the trap or out of it".

The algorithm is not a variant of the cell-based one; it is a different
thing:

  1. Every node points at its **shallowest neighbour** (buoyant CO2 runs
     updip). Neighbours include the *diagonals*, which a cell-based analysis
     has no notion of - two nodes joined by two distinct two-step paths are
     treated as adjacent.
  2. A node whose shallowest neighbour is itself is a **summit**. Following
     the arrows from any node lands on one, and that defines a **region** -
     one per summit.
  3. Regions are separated by **spill edges**: an edge whose two nodes lie in
     different regions. The depth at which a region overflows is the
     shallowest of its spill edges, an edge's depth being the *deeper* of its
     two nodes - a saddle is only as high as its lowest point.
  4. Two regions that spill into *each other* are not two traps: they fill to
     a common level and are one. So the connectivity graph is searched for
     **cycles**, each cycle is contracted to a single trap, its spill point is
     recomputed, and the search repeats until no cycles remain. That
     contraction is the heart of it.

`node_trap_analysis` returns the traps on the nodes; `nodes_to_cells`
projects them onto cells so the result can be used like the cell-based one.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from pymrst.modules.co2lab_ve.ve import TopSurfaceGrid


@dataclass
class NodeTrapAnalysis:
    """Traps, regions and spill structure, on the grid's nodes."""

    traps: np.ndarray            # (n_nodes,) trap index, 0 = not in a trap
    trap_z: np.ndarray           # (n_traps,) spill depth of each trap
    trap_regions: np.ndarray     # (n_nodes,) region each node drains to
    trap_adj: np.ndarray         # (n_traps, n_traps) i spills into j
    upstream: np.ndarray         # (n_nodes,) shallowest neighbour, -1 at summits
    summits: np.ndarray          # (n_nodes,) 1 at a summit inside a trap

    @property
    def n_traps(self) -> int:
        return int(self.traps.max())


def _node_edges(ts: TopSurfaceGrid) -> np.ndarray:
    """The grid's edges, as node pairs. A 2D face *is* an edge."""
    g = ts.g
    npos = g.faces.nodePos
    assert np.all(np.diff(npos) == 2), (
        "the node-based analysis needs a 2D grid whose faces are edges")
    return g.faces.nodes.reshape(-1, 2)


def _node_adjacency(ts: TopSurfaceGrid, edges: np.ndarray) -> List[np.ndarray]:
    """Node neighbours, *including diagonals* (MRST's findUpslopeNeighbor).

    Two nodes joined by two distinct two-step paths are diagonal neighbours -
    on a Cartesian grid, the corners of a cell. MRST finds them as the entries
    where the square of the adjacency matrix equals 2. Without them the flow
    directions are quantised to the axes and a diagonal ridge cannot be seen.
    """
    import scipy.sparse as sp

    n = ts.g.nodes.num
    a = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                      shape=(n, n)).tocsr()
    a = a + a.T
    two_step = (a @ a)
    diag = (two_step == 2)                       # exactly two distinct paths
    full = ((a + diag).astype(bool)).tolil()
    full.setdiag(True)                           # a node is its own neighbour
    full = full.tocsr()
    return [full.indices[full.indptr[i]:full.indptr[i + 1]] for i in range(n)]


def _shallowest_neighbour(ts: TopSurfaceGrid,
                          nbr: List[np.ndarray]) -> np.ndarray:
    """Each node's steepest updip neighbour (MRST's findUpslopeNeighbor).

    The score is sign(dz) dz^2 / dist^2 - a signed squared slope - and the
    node itself scores zero. So a node with no shallower neighbour picks
    itself, which is what makes it a summit.
    """
    z = ts.node_z
    xyz = np.column_stack([ts.g.nodes.coords[:, :2], z])
    up = np.zeros(ts.g.nodes.num, dtype=np.int64)

    for j in range(ts.g.nodes.num):
        cand = nbr[j]
        dz = z[j] - z[cand]                      # > 0 => cand is shallower
        d2 = np.sum((xyz[cand] - xyz[j]) ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.sign(dz) * dz**2 / d2
        slope[~np.isfinite(slope)] = 0.0         # the node itself
        up[j] = int(cand[np.argmax(slope)])
    return up


def _boundary_nodes(ts: TopSurfaceGrid) -> np.ndarray:
    g = ts.g
    on_bnd = np.zeros(g.nodes.num, dtype=bool)
    bfaces = np.flatnonzero(g.faces.neighbors.min(axis=1) < 0)
    for f in bfaces:
        on_bnd[g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]] = True
    return on_bnd


def _spill_field(ts: TopSurfaceGrid, closed_boundary: Optional[np.ndarray]
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summits, regions and the arrows between them (MRST's nodeSpillField)."""
    edges = _node_edges(ts)
    nbr = _node_adjacency(ts, edges)
    up = _shallowest_neighbour(ts, nbr)

    open_bnd = _boundary_nodes(ts)
    if closed_boundary is not None:
        open_bnd[np.asarray(closed_boundary, np.int64)] = False

    summit = (up == np.arange(ts.g.nodes.num)) & ~open_bnd

    region = np.full(ts.g.nodes.num, -1, dtype=np.int64)
    region[open_bnd] = 0                          # leaks out of the domain
    region[summit] = np.arange(1, summit.sum() + 1)

    # everyone else takes the region of the node it points at
    todo = np.flatnonzero(region < 0)
    while todo.size:
        nxt = region[up[todo]]
        known = nxt >= 0
        if not np.any(known):
            raise RuntimeError("the spill field has a cycle in it")
        region[todo[known]] = nxt[known]
        todo = todo[~known]

    up = up.copy()
    up[open_bnd] = -1
    up[summit] = -1
    return up, region, edges


def _spill_edges(ts: TopSurfaceGrid, region: np.ndarray,
                 edges: np.ndarray) -> np.ndarray:
    """Edges whose two nodes lie in different regions.

    Columns: [edge, region_a, region_b, z_a, z_b, node_a, node_b].
    """
    ra, rb = region[edges[:, 0]], region[edges[:, 1]]
    sel = np.flatnonzero(ra != rb)
    return np.column_stack([
        sel, ra[sel], rb[sel],
        ts.node_z[edges[sel, 0]], ts.node_z[edges[sel, 1]],
        edges[sel, 0], edges[sel, 1],
    ])


def _neighbours_of(cluster: Set[int], spill: np.ndarray
                   ) -> Tuple[Set[int], float, np.ndarray]:
    """Where a cluster of regions spills to, and at what depth.

    An edge is a way out only if exactly one of its ends is in the cluster.
    Its depth is the *deeper* of its two nodes - a saddle is only as high as
    its lowest point - and the cluster overflows at the shallowest such edge.
    """
    in_a = np.array([r in cluster for r in spill[:, 1].astype(int)])
    in_b = np.array([r in cluster for r in spill[:, 2].astype(int)])
    out = np.flatnonzero(in_a ^ in_b)
    if out.size == 0:
        return set(), np.inf, np.zeros(0, dtype=np.int64)

    depth = np.maximum(spill[out, 3], spill[out, 4])
    z = float(depth.min())
    at = out[depth == z]

    neigh: Set[int] = set()
    for k in at:
        for r in (int(spill[k, 1]), int(spill[k, 2])):
            if r != 0 and r not in cluster:
                neigh.add(r)
    return neigh, z, spill[at, 0].astype(np.int64)


def _contract_cycles(regions: Sequence[int], spill: np.ndarray):
    """Merge regions that spill into each other (MRST's track_stream).

    Two regions that each overflow into the other cannot be two traps: CO2
    filling either one runs into the other and they come to a common level.
    So the spill graph is searched for cycles, every cycle is contracted into
    a single trap, its spill point is recomputed at the merged rim - which may
    create a *new* cycle with a third trap - and the search repeats.
    """
    clusters: List[Set[int]] = [{int(r)} for r in regions]
    zvals: List[float] = []
    sp_edges: List[np.ndarray] = []
    adj: List[Set[int]] = []

    def recompute(i: int):
        neigh, z, edges = _neighbours_of(clusters[i], spill)
        # map region ids to cluster ids
        cl = set()
        for r in neigh:
            for j, c in enumerate(clusters):
                if r in c:
                    cl.add(j)
        return cl, z, edges

    for i in range(len(clusters)):
        cl, z, e = recompute(i)
        adj.append(cl)
        zvals.append(z)
        sp_edges.append(e)

    while True:
        cyc = _find_cycle(adj)
        if cyc is None:
            break
        keep = min(cyc)
        merged = set()
        for i in cyc:
            merged |= clusters[i]
        clusters[keep] = merged

        drop = sorted(set(cyc) - {keep}, reverse=True)
        for i in drop:
            clusters.pop(i)
            zvals.pop(i)
            sp_edges.pop(i)
            adj.pop(i)

        # renumber and rebuild the graph: a merged trap has a new rim, so its
        # spill point (and everyone pointing at it) has to be found again
        adj = []
        zvals = []
        sp_edges = []
        for i in range(len(clusters)):
            cl, z, e = recompute(i)
            adj.append(cl)
            zvals.append(z)
            sp_edges.append(e)

    return clusters, np.asarray(zvals, float), sp_edges, adj


def _find_cycle(adj: List[Set[int]]) -> Optional[List[int]]:
    """Any cycle in the spill graph, as a list of nodes. None if acyclic."""
    n = len(adj)
    colour = [0] * n                    # 0 unseen, 1 on the stack, 2 done
    parent = [-1] * n

    for s in range(n):
        if colour[s]:
            continue
        stack = [(s, iter(sorted(adj[s])))]
        colour[s] = 1
        while stack:
            v, it = stack[-1]
            advanced = False
            for w in it:
                if colour[w] == 0:
                    colour[w] = 1
                    parent[w] = v
                    stack.append((w, iter(sorted(adj[w]))))
                    advanced = True
                    break
                if colour[w] == 1:      # back edge: found a cycle
                    cyc = [w]
                    x = v
                    while x != w and x != -1:
                        cyc.append(x)
                        x = parent[x]
                    return sorted(set(cyc))
            if not advanced:
                colour[v] = 2
                stack.pop()
    return None


def node_trap_analysis(ts: TopSurfaceGrid,
                       closed_boundary: Optional[np.ndarray] = None
                       ) -> NodeTrapAnalysis:
    """Structural traps, resolved on the grid's nodes (computeNodeTraps)."""
    assert ts.node_z is not None, (
        "the node-based analysis needs node depths; rebuild the top-surface "
        "grid with top_surface_grid")

    up, region, edges = _spill_field(ts, closed_boundary)
    spill = _spill_edges(ts, region, edges)

    n_regions = int(region.max())
    clusters, zvals, sp_edges, adj = _contract_cycles(
        range(1, n_regions + 1), spill)

    n_traps = len(clusters)
    traps = np.zeros(ts.g.nodes.num, dtype=np.int64)
    trap_regions = np.zeros(ts.g.nodes.num, dtype=np.int64)

    for i, cl in enumerate(clusters, start=1):
        member = np.isin(region, list(cl))
        trap_regions[member] = i
        # a node is *in* the trap if the CO2 standing at the spill depth
        # covers it
        traps[member & (ts.node_z <= zvals[i - 1])] = i

    adj_mat = np.zeros((n_traps, n_traps), dtype=bool)
    for i, nb in enumerate(adj):
        for j in nb:
            adj_mat[i, j] = True

    summits = np.zeros(ts.g.nodes.num, dtype=np.int64)
    summits[(up < 0) & (trap_regions > 0)] = 1

    return NodeTrapAnalysis(traps=traps, trap_z=zvals,
                            trap_regions=trap_regions, trap_adj=adj_mat,
                            upstream=up, summits=summits)


def nodes_to_cells(ts: TopSurfaceGrid, res: NodeTrapAnalysis):
    """Project node traps onto cells (MRST's n2cTraps).

    A cell takes the region of its **shallowest node** - not the region most
    of its nodes drain to. Buoyant CO2 collects in the shallowest corner of a
    cell, so that corner is what decides where the cell's CO2 goes; a majority
    vote gets a handful of cells along every trap's rim wrong.

    A cell is then *in* a trap if its own depth is above that trap's spill
    depth. Regions that lose every cell this way - a trap that lives entirely
    between cell centres, which is exactly what the node-based method can see
    and the cell-based one cannot - are dropped, and their catchments handed to
    whatever they spill into.
    """
    from pymrst.modules.co2lab_spillpoint.trapping import TrapAnalysis

    g = ts.g
    nc = g.cells.num
    cell_region = np.zeros(nc, dtype=np.int64)

    for c in range(nc):
        lo, hi = g.cells.facePos[c], g.cells.facePos[c + 1]
        nodes = np.unique(np.concatenate([
            g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
            for f in g.cells.faces[lo:hi]]))
        shallowest = nodes[np.argmin(ts.node_z[nodes])]
        cell_region[c] = res.trap_regions[shallowest]

    traps = np.zeros(nc, dtype=np.int64)
    for i in range(1, res.n_traps + 1):
        sel = (cell_region == i) & (ts.z < res.trap_z[i - 1])
        traps[sel] = i

    # traps that caught no cell at all are below the grid's resolution
    kept = np.array([i for i in range(1, res.n_traps + 1)
                     if np.any(traps == i)], dtype=np.int64)
    remap = np.zeros(res.n_traps + 1, dtype=np.int64)
    remap[kept] = np.arange(1, kept.size + 1)

    lost = [i for i in range(1, res.n_traps + 1) if remap[i] == 0]
    for i in lost:
        # hand the catchment to the first trap downstream that survived
        target = 0
        seen = {i}
        nxt = np.flatnonzero(res.trap_adj[i - 1])
        while nxt.size:
            j = int(nxt[0]) + 1
            if j in seen:
                break
            if remap[j]:
                target = remap[j]
                break
            seen.add(j)
            nxt = np.flatnonzero(res.trap_adj[j - 1])
        cell_region[cell_region == i] = 0 if target == 0 else -target

    cell_region = np.where(cell_region < 0, -cell_region, remap[cell_region])
    traps = remap[traps]
    trap_z = res.trap_z[kept - 1]

    adj = res.trap_adj[np.ix_(kept - 1, kept - 1)]
    top = np.zeros(nc, dtype=np.int64)
    for i in range(1, kept.size + 1):
        cells = np.flatnonzero(traps == i)
        if cells.size:
            top[cells[np.argmin(ts.z[cells])]] = 1

    fill = ts.z.copy()
    for i in range(1, kept.size + 1):
        sel = traps == i
        fill[sel] = trap_z[i - 1]

    return TrapAnalysis(traps=traps, trap_z=trap_z, trap_regions=cell_region,
                        trap_adj=adj, top=top, fill_z=fill)


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.co2lab_spillpoint.trapping import trap_analysis, volumes_of_traps
    from pymrst.modules.co2lab_ve.ve import top_surface_grid

    def _surface(fn, nx=40, ny=30, lx=800.0, ly=600.0):
        g = cart_grid([nx, ny, 1], [lx, ly, 20.0])
        x, y = g.nodes.coords[:, 0], g.nodes.coords[:, 1]
        g.nodes.coords[:, 2] += fn(x, y)
        return top_surface_grid(compute_geometry(g))

    def _one_dome(x, y):
        return (1000.0 - 0.02 * x
                - 12.0 * np.exp(-(((x - 400.0) / 110.0) ** 2
                                  + ((y - 300.0) / 110.0) ** 2)))

    def _two_domes(x, y):
        return (1000.0 - 0.02 * x
                - 10.0 * np.exp(-(((x - 250.0) / 80.0) ** 2
                                  + ((y - 300.0) / 90.0) ** 2))
                - 14.0 * np.exp(-(((x - 600.0) / 85.0) ** 2
                                  + ((y - 300.0) / 90.0) ** 2)))

    def test_summits_are_the_shallowest_nodes():
        """Every node's arrow points updip, and a summit has nowhere to go."""
        ts = _surface(_one_dome)
        res = node_trap_analysis(ts)

        up = res.upstream
        moved = np.flatnonzero(up >= 0)
        # every arrow goes to a strictly shallower node
        assert np.all(ts.node_z[up[moved]] <= ts.node_z[moved] + 1e-12), \
            "an arrow pointed downdip"

        # and the summits are local minima of depth
        for s in np.flatnonzero(res.summits):
            assert up[s] < 0
        assert res.n_traps >= 1
        print(f"ok test_summits_are_the_shallowest_nodes "
              f"({int(res.summits.sum())} summits, {res.n_traps} traps)")

    def test_one_dome_traps_the_nodes_above_its_spill_point():
        """The trap is exactly the nodes shallower than the spill depth."""
        ts = _surface(_one_dome)
        res = node_trap_analysis(ts)
        assert res.n_traps == 1, res.n_traps

        z_spill = res.trap_z[0]
        inside = res.traps == 1
        assert np.all(ts.node_z[inside] <= z_spill + 1e-9)
        # and the crest is well above the spill point
        assert ts.node_z[inside].min() < z_spill - 1.0
        # every trapped node is in the trap's own region
        assert np.all(res.trap_regions[inside] == 1)
        print(f"ok test_one_dome_traps_the_nodes_above_its_spill_point "
              f"({int(inside.sum())} nodes, spill depth {z_spill:.2f} m)")

    def test_two_domes_spill_downdip():
        """The updip dome fills and overflows into the downdip one."""
        ts = _surface(_two_domes)
        res = node_trap_analysis(ts)
        assert res.n_traps == 2, res.n_traps

        # CO2 migrates toward smaller x... no: the slope is -0.02 x, so the
        # surface is shallower at large x, and CO2 runs that way. The dome at
        # x = 250 is therefore *upstream* of the one at x = 600.
        xs = ts.g.nodes.coords[:, 0]
        cx = [xs[res.traps == i].mean() for i in (1, 2)]
        up_ix = int(np.argmin(cx))          # the dome at smaller x
        dn_ix = 1 - up_ix

        assert res.trap_adj[up_ix, dn_ix], (
            "the updip trap should spill into the downdip one")
        assert not res.trap_adj[dn_ix, up_ix], "and not the other way"
        print(f"ok test_two_domes_spill_downdip "
              f"(trap at x = {cx[up_ix]:.0f} m spills into the one at "
              f"x = {cx[dn_ix]:.0f} m)")

    def test_cycle_contraction_merges_a_shared_saddle():
        """Two bowls in one basin are one trap, not two.

        Two domes joined by a saddle *below* both their crests but with no way
        out between them fill to a common level: CO2 in either runs into the
        other. The cycle contraction is what notices. Without it they would be
        reported as two traps with two capacities, and the sum would be wrong.
        """
        def twin(x, y):
            # a single broad high with two bumps on it, inside one basin
            base = -20.0 * np.exp(-(((x - 400.0) / 200.0) ** 2
                                    + ((y - 300.0) / 160.0) ** 2))
            b1 = -4.0 * np.exp(-(((x - 330.0) / 55.0) ** 2
                                 + ((y - 300.0) / 55.0) ** 2))
            b2 = -4.0 * np.exp(-(((x - 470.0) / 55.0) ** 2
                                 + ((y - 300.0) / 55.0) ** 2))
            return 1000.0 - 0.005 * x + base + b1 + b2

        ts = _surface(twin, nx=60, ny=44, lx=800.0, ly=600.0)
        res = node_trap_analysis(ts)

        # the two bumps are separate *regions* (two summits) but one trap
        n_summits = int(res.summits.sum())
        assert n_summits >= 2, n_summits
        assert res.n_traps < n_summits, (
            f"{n_summits} summits collapsed to {res.n_traps} traps - the cycle "
            f"contraction did nothing")

        # and the merged trap contains both bumps' crests
        big = 1 + int(np.argmax([np.sum(res.traps == i)
                                 for i in range(1, res.n_traps + 1)]))
        crests = np.flatnonzero(res.summits)
        assert np.sum(res.traps[crests] == big) >= 2, \
            "the merged trap should hold both summits"
        print(f"ok test_cycle_contraction_merges_a_shared_saddle "
              f"({n_summits} summits contract to {res.n_traps} trap(s))")

    def test_projection_to_cells_agrees_with_the_cell_based_analysis():
        """Projected onto cells, the two analyses find the same traps.

        They discretise the caprock differently - one at nodes, one at cell
        centres - so they will not agree cell for cell on a coarse grid. But
        they must find the same *structures*: the same number of traps, in the
        same places, holding comparable volumes. If they did not, one of them
        would be wrong.
        """
        ts = _surface(_two_domes, nx=60, ny=44)
        node = nodes_to_cells(ts, node_trap_analysis(ts))
        cell = trap_analysis(ts)

        assert node.n_traps == cell.n_traps == 2, (node.n_traps, cell.n_traps)

        poro = np.full(ts.g.cells.num, 0.2)
        vn = np.sort(volumes_of_traps(ts, node, poro=poro))
        vc = np.sort(volumes_of_traps(ts, cell, poro=poro))
        rel = np.abs(vn - vc) / vc
        assert np.all(rel < 0.35), (
            f"the two analyses disagree on capacity by {rel}\n"
            f"  node-based: {vn}\n  cell-based: {vc}")

        # the traps sit in the same places
        xn = [ts.g.cells.centroids[node.traps == i, 0].mean() for i in (1, 2)]
        xc = [ts.g.cells.centroids[cell.traps == i, 0].mean() for i in (1, 2)]
        assert np.abs(np.sort(xn) - np.sort(xc)).max() < 40.0, (xn, xc)
        print(f"ok test_projection_to_cells_agrees_with_the_cell_based_analysis "
              f"(capacities {np.round(vn / 1e3, 1)} vs "
              f"{np.round(vc / 1e3, 1)} x1e3 m3, within "
              f"{100 * rel.max():.0f}%)")

    def test_a_monotone_slope_has_no_traps():
        """Nothing to trap on a plane."""
        ts = _surface(lambda x, y: 1000.0 - 0.03 * x + 0.01 * y)
        res = node_trap_analysis(ts)
        assert res.n_traps == 0, f"found {res.n_traps} traps on a plane"
        assert np.all(res.traps == 0)
        print("ok test_a_monotone_slope_has_no_traps")

    test_summits_are_the_shallowest_nodes()
    test_a_monotone_slope_has_no_traps()
    test_one_dome_traps_the_nodes_above_its_spill_point()
    test_two_domes_spill_downdip()
    test_cycle_contraction_merges_a_shared_saddle()
    test_projection_to_cells_agrees_with_the_cell_based_analysis()
    print("\nAll tests passed!")
