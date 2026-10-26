"""Grid, rock and transmissibility from ECLIPSE output (MRST
``deckformat/resultinput/initGridFromEclipseOutput.m``).

A field model's EGRID/INIT pair is not just a copy of its deck. It is the
simulator's *own* processed grid, and with it the transmissibilities ECLIPSE
actually used - including across the non-neighbour connections a faulted model
depends on. `fieldModelNorneExample` builds its model from these rather than
from the deck for exactly that reason: reprocessing the deck gives a grid, but
not the same one, and not the same fault transmissibilities.

Three things decide whether the result is usable, and each is a place a plausible
shortcut is wrong:

* **a cell is active when `ACTNUM` says so *and* `PORV > 0`.** ECLIPSE drops
  zero-pore-volume cells whatever ACTNUM claims. Using ACTNUM alone leaves the
  index map one cell out of step from the first such cell onward, and every
  permeability after it belongs to a different cell;
* **`PORV` is written for the whole Cartesian box, the rock arrays only for
  active cells.** They are indexed differently and mixing them up is silent;
* **a connection is kept only when both its cells are active and `T > 0`.**
  A zero transmissibility is ECLIPSE saying the connection is closed - a
  sealing fault, usually - and keeping it reopens the fault.

The non-neighbour connections come from the EGRID's `NNC1`/`NNC2` with their
transmissibilities in `TRANNNC`. Where the file gives the pairs but no
transmissibility, MRST substitutes infinity and warns; that is preserved,
because a silent zero would seal the faults instead of opening them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, FT, METER, MILLI, POISE

#: `INTEHEAD[2]`: 1 metric, 2 field, 3 lab.
UNIT_SYSTEMS = {1: "metric", 2: "field", 3: "lab"}


@dataclass
class OutputUnits:
    length: float
    perm: float
    trans: float
    resvolume: float


def output_units(name: str) -> OutputUnits:
    """The units ECLIPSE wrote the file in."""
    if name == "metric":
        return OutputUnits(length=METER, perm=MILLI*DARCY,
                           trans=CENTI*POISE*METER**3/(DAY*BARSA),
                           resvolume=METER**3)
    if name == "field":
        from pymrst.utils.units import PSIA, STB
        return OutputUnits(length=FT, perm=MILLI*DARCY,
                           trans=CENTI*POISE*STB/(DAY*PSIA), resvolume=STB)
    if name == "lab":
        from pymrst.utils.units import ATM, CENTI as C, HOUR
        return OutputUnits(length=C*METER, perm=MILLI*DARCY,
                           trans=CENTI*POISE*(C*METER)**3/(HOUR*ATM),
                           resvolume=(C*METER)**3)
    raise ValueError(f"unknown ECLIPSE unit system {name!r}")


def init_grid_from_eclipse_output(init: Dict[str, Any], grid: Dict[str, Any],
                                  *, output_sim_grid: bool = False):
    """Return ``(grids, rock, N, T)`` from an INIT/EGRID pair.

    ``grids`` is the visualisation grid, or ``[G_viz, G_sim]`` when
    ``output_sim_grid`` is set - MRST's two-element cell array, where the
    second is the grid the simulation actually runs on.
    """
    from pymrst.gridprocessing import compute_geometry, process_grdecl
    from pymrst.params.rock import Rock

    u = output_units(UNIT_SYSTEMS[int(_values(init, "INTEHEAD")[2])])
    cart_dims, coord, zcorn, actnum_grid = _geometry(grid)

    # ECLIPSE drops zero-pore-volume cells whatever ACTNUM says.
    porv = _values(init, "PORV")
    active = np.logical_and(actnum_grid.astype(bool), porv > 0.0)

    grdecl = {"cartDims": tuple(int(d) for d in cart_dims),
              "COORD": np.asarray(coord, float)*u.length,
              "ZCORN": np.asarray(zcorn, float)*u.length,
              "ACTNUM": active.astype(np.int64)}
    # `split_disconnected=False` is MRST's `'SplitDisconnected', false`. A
    # field model routinely has cells the reprocessing sees as disconnected;
    # splitting them returns several grids, and the INIT's arrays are indexed
    # against the one ECLIPSE built.
    viz = compute_geometry(process_grdecl(grdecl, split_disconnected=False))

    n_active = int(active.sum())
    e_map = _eclipse_map(viz, active, n_active)
    viz.cells.eMap = e_map

    depth = _values(init, "DEPTH", optional=True)
    if depth is not None and depth.size:
        # ECLIPSE's own cell depths, which need not equal the reprocessed
        # grid's centroids - and it is ECLIPSE's that the transmissibilities
        # and the equilibration were computed against.
        viz.cells.centroids[:, 2] = depth[e_map]*u.length
    viz.cells.PORV = porv[np.asarray(viz.cells.indexMap, dtype=np.int64)]*u.resvolume
    # ECLIPSE's own cell dimensions. The simulation grid is a connection list
    # with no nodes, so a well index cannot derive them from geometry - these
    # are what MRST's `simGridTPFA` carries them for.
    cell_sizes = {}
    for name in ("DX", "DY", "DZ"):
        values = _values(init, name, optional=True)
        if values is not None and values.size:
            cell_sizes[name] = values[e_map]*u.length
            setattr(viz.cells, name, cell_sizes[name])

    rock = Rock(
        perm=np.column_stack([_values(init, name) for name in
                              ("PERMX", "PERMY", "PERMZ")])*u.perm,
        poro=_values(init, "PORO").copy(),
    )
    ntg = _values(init, "NTG", optional=True)
    if ntg is not None and ntg.size:
        rock.ntg = ntg.copy()

    if not output_sim_grid:
        N, T = active_neighbors(init, grid, active, cart_dims)
        return viz, rock, N, T*u.trans
    N, T, ix = active_neighbors(init, grid, active, cart_dims, nargout=3)
    T = T*u.trans
    sim = sim_grid_tpfa(viz, rock, neighbors=N, porv=porv[active]*u.resvolume,
                        depth=(depth*u.length if depth is not None and depth.size else None),
                        actnum=active)
    sim.faces.ix = ix
    aquifer = _values(init, "AQUIFERN", optional=True)
    if aquifer is not None:
        sim.cells.aquifer = aquifer
    for name, values in cell_sizes.items():
        setattr(sim.cells, name, values)
    sim.cells.PORV = porv[active]*u.resvolume
    return [viz, sim], rock, N, T


initGridFromEclipseOutput = init_grid_from_eclipse_output


def active_neighbors(init, grid, active: np.ndarray, cart_dims, nargout: int = 2):
    """MRST's `[N, T, ix] = getActiveNeighbors(init, grid, actNum, cartDims)`:
    the connections ECLIPSE kept and their transmissibilities.

    Cartesian connections are kept where both cells are active and the
    transmissibility positive (a zero is ECLIPSE closing the connection, a
    sealing fault). With `nargout=3` (the simulation grid's path) the NNCs
    are oriented from the lower cell to the higher, kept where their
    transmissibility is positive, and `ix` maps the restart's I/J/K/NNC flux
    arrays onto the connections (`iIe` etc. select, `iI` etc. number them,
    `sN` signs the reoriented NNCs). Missing NNC transmissibilities are
    infinite, with MRST's warning. Cells are 0-based.
    """
    dims = tuple(int(d) for d in cart_dims)
    n_cells = int(active.sum())
    # 1-based active index per Cartesian cell, 0 for inactive - MRST's actInx.
    act_index = np.zeros(int(np.prod(dims)), dtype=np.int64)
    act_index[active] = np.arange(1, n_cells + 1)
    padded = np.zeros(tuple(d + 1 for d in dims), dtype=np.int64)
    padded[:-1, :-1, :-1] = act_index.reshape(dims, order="F")
    NX = padded[1:, :-1, :-1].ravel(order="F")[active]
    NY = padded[:-1, 1:, :-1].ravel(order="F")[active]
    NZ = padded[:-1, :-1, 1:].ravel(order="F")[active]
    own = np.arange(1, n_cells + 1)
    nnc_src = grid if _values(grid, "NNC1", optional=True) is not None else init
    nnc1 = _values(nnc_src, "NNC1", optional=True)
    trannnc = _values(init, "TRANNNC", optional=True)
    if trannnc is None and nnc1 is not None:
        warnings.warn("Transmissibilities for NNCs not given, values are set to inf!")
        trannnc = np.full(nnc1.size, np.inf)
    if trannnc is not None and nnc1 is not None:
        NNC1 = act_index[nnc1.astype(np.int64) - 1]
        NNC2 = act_index[_values(nnc_src, "NNC2").astype(np.int64) - 1]
    else:
        NNC1 = NNC2 = np.zeros(0, dtype=np.int64)
        trannnc = np.zeros(0)
    TI, TJ, TK = (_values(init, name) for name in ("TRANX", "TRANY", "TRANZ"))
    if nargout < 3:
        N = np.vstack([np.column_stack([own, NX]), np.column_stack([own, NY]),
                       np.column_stack([own, NZ]), np.column_stack([NNC1, NNC2])])
        T = np.concatenate([TI, TJ, TK, trannnc])
        keep = (T > 0) & (N.prod(axis=1) > 0)
        return N[keep] - 1, T[keep]
    from types import SimpleNamespace
    ix = SimpleNamespace()
    blocks, tblocks, offset = [], [], 0
    for d, nb, t in (("I", NX, TI), ("J", NY, TJ), ("K", NZ, TK)):
        Nd = np.column_stack([own, nb])
        inx = (t > 0) & (Nd.prod(axis=1) > 0)
        setattr(ix, f"i{d}e", inx)
        setattr(ix, f"i{d}", np.arange(int(inx.sum())) + offset)
        setattr(ix, f"s{d}", 1)
        offset += int(inx.sum())
        blocks.append(Nd[inx])
        tblocks.append(t[inx])
    NN = np.column_stack([NNC1, NNC2])
    is_pos = NNC1 < NNC2
    if NN.size:
        NN[~is_pos] = NN[~is_pos][:, ::-1]
        inx = trannnc > 0
    else:
        inx = np.zeros(0, dtype=bool)
    ix.iNe = inx
    ix.iN = np.arange(int(inx.sum())) + offset
    ix.sN = 2*is_pos[inx] - 1
    N = np.vstack(blocks + [NN[inx]])
    T = np.concatenate(tblocks + [trannnc[inx]])
    return N - 1, T, ix


getActiveNeighbors = active_neighbors


def sim_grid_tpfa(G, rock, *, neighbors=None, porv=None, depth=None, actnum=None,
                  trans=None, face_centroids=None):
    """MRST `Gs = simGridTPFA(G, rock, 'neighbors', N, 'porv', ..., 'depth', ...)`.

    The grid the simulation runs on: the cells, with the connection list as
    faces. Volumes are `porv/poro` (and `/ntg`), so pore volumes are ECLIPSE's;
    centroids are `G`'s (NaN where `G` has none) with `depth` as their third
    coordinate; face centroids are the mean of their two cells'; face areas
    are NaN. Cells and connections are 0-based (a boundary neighbour is -1).
    """
    from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes

    emap = getattr(G.cells, "eMap", None)
    consistent = emap is None or (isinstance(emap, str) and emap == ":") or (
        np.size(emap) == G.cells.num and np.array_equal(emap, np.arange(G.cells.num)))
    if neighbors is None:
        nc = int(G.cells.num)
        N = np.asarray(G.faces.neighbors, dtype=np.int64)
        face_pos, faces = np.asarray(G.cells.facePos), np.asarray(G.cells.faces)
        index_map = np.asarray(G.cells.indexMap, dtype=np.int64)
    else:
        N = np.asarray(neighbors, dtype=np.int64).reshape(-1, 2)
        nf = N.shape[0]
        if consistent:
            nc = int(G.cells.num)
            index_map = np.asarray(G.cells.indexMap, dtype=np.int64)
        else:
            assert actnum is not None, "Numeric 'eMap'-field requires non-empty actnum"
            nc = int(max(np.max(emap), N.max())) + 1
            index_map = np.flatnonzero(actnum)
        owners = N.ravel(order="F")
        fno = np.tile(np.arange(nf), 2)
        keep = owners >= 0
        owners, fno = owners[keep], fno[keep]
        order = np.lexsort((fno, owners))
        counts = np.bincount(owners[order], minlength=nc)
        face_pos = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        faces = fno[order]
    if porv is not None:
        volumes = np.asarray(porv, dtype=float)/np.asarray(rock.poro, dtype=float).ravel()
        ntg = getattr(rock, "ntg", None)
        if ntg is not None:
            volumes = volumes/np.asarray(ntg, dtype=float).ravel()
    else:
        volumes = np.asarray(G.cells.volumes, dtype=float).copy()
    centroids = np.full((int(G.cells.num), 3), np.nan)
    if getattr(G.cells, "centroids", None) is not None:
        rows = np.arange(G.cells.num) if consistent else np.asarray(emap)
        centroids[rows, :] = G.cells.centroids
    if depth is not None:
        centroids[:, 2] = depth
    if face_centroids is None:
        face_centroids = (centroids[N[:, 0], :] + centroids[N[:, 1], :])/2
    grid = Grid(
        cells=Cells(num=nc, facePos=face_pos, faces=faces, indexMap=index_map,
                    volumes=volumes, centroids=centroids),
        faces=Faces(num=N.shape[0], nodePos=np.zeros(N.shape[0] + 1, dtype=np.int64),
                    nodes=np.zeros(0, dtype=np.int64), neighbors=N),
        nodes=Nodes(num=0, coords=np.zeros((0, 3))),
        griddim=3, cartDims=G.cartDims, type=["tpfaGrid"],
    )
    grid.faces.centroids = face_centroids
    grid.faces.areas = np.full(int(G.faces.num), np.nan)
    if trans is not None:
        grid.faces.TRANS = trans
    return grid


simGridTPFA = sim_grid_tpfa


def _geometry(grid):
    """`(cartDims, COORD, ZCORN, ACTNUM)` from an EGRID or a deck GRID."""
    if "cartDims" in grid:
        dims = np.asarray(grid["cartDims"], dtype=np.int64).ravel()[:3]
        return (dims, _values(grid, "COORD"), _values(grid, "ZCORN"),
                _values(grid, "ACTNUM"))
    head = _values(grid, "GRIDHEAD").astype(np.int64)
    return (head[1:4], _values(grid, "COORD"), _values(grid, "ZCORN"),
            _values(grid, "ACTNUM"))


def _eclipse_map(viz, active: np.ndarray, n_active: int) -> np.ndarray:
    """Index from the reprocessed grid's cells to ECLIPSE's active ordering.

    `process_grdecl` can drop a cell ECLIPSE kept - a pinched-out layer, say -
    so the two orderings need not agree. Assuming they do puts every array read
    from the INIT one cell out of step from the first disagreement on.
    """
    if int(viz.cells.num) == n_active:
        return np.arange(n_active, dtype=np.int64)
    table = np.zeros(active.size, dtype=np.int64)
    table[active] = np.arange(n_active, dtype=np.int64)
    return table[np.asarray(viz.cells.indexMap, dtype=np.int64)]


def _values(source, name: str, *, optional: bool = False):
    found = source.get(name) if hasattr(source, "get") else None
    if found is None:
        if optional:
            return None
        raise KeyError(f"the ECLIPSE output carries no {name!r}")
    values = getattr(found, "values", found)
    return np.asarray(values).ravel()


__all__ = ["OutputUnits", "UNIT_SYSTEMS", "active_neighbors",
           "getActiveNeighbors", "initGridFromEclipseOutput",
           "init_grid_from_eclipse_output", "output_units", "simGridTPFA",
           "sim_grid_tpfa"]


if __name__ == "__main__":
    from pathlib import Path

    from pymrst.modules.deckformat.resultinput.read_eclipse_output_file_un_fmt import (
        read_eclipse_output_file_un_fmt,
    )

    PREFIX = (Path(__file__).resolve().parents[5] / "mrst-2024a-Geomech" / "examples" / "data"
              / "norne_field_model" / "opm-tests" / "norne" / "opm-simulation-reference"
              / "flow_legacy" / "NORNE_ATW2013")

    def test_the_simulation_grid_reproduces_eclipses_pore_volume():
        if not Path(str(PREFIX) + ".INIT").exists():
            print("skip test_the_simulation_grid_reproduces_eclipses_pore_volume")
            return
        init = read_eclipse_output_file_un_fmt(str(PREFIX) + ".INIT")
        egrid = read_eclipse_output_file_un_fmt(str(PREFIX) + ".EGRID")
        (viz, sim), rock, N, T = init_grid_from_eclipse_output(init, egrid,
                                                               output_sim_grid=True)
        pv = sim.cells.volumes*np.asarray(rock.poro, float)*np.asarray(rock.ntg, float)
        np.testing.assert_allclose(pv, sim.cells.PORV, rtol=1e-12)
        nnc = N[sim.faces.ix.iN]
        assert np.all(nnc[:, 0] < nnc[:, 1]), "NNCs run from the lower cell to the higher"
        n = sim.faces.ix.iI.size + sim.faces.ix.iJ.size + sim.faces.ix.iK.size + sim.faces.ix.iN.size
        assert n == N.shape[0] == T.size and np.all(T > 0)
        assert np.all(np.isfinite(sim.faces.centroids))
        print(f"ok test_the_simulation_grid_reproduces_eclipses_pore_volume "
              f"({sim.cells.num} cells, {N.shape[0]} connections)")

    def test_the_visualisation_path_filters_without_reorienting():
        init = {"TRANX": np.array([1.0, 0.0]), "TRANY": np.zeros(2), "TRANZ": np.zeros(2),
                "NNC1": np.array([2.0]), "NNC2": np.array([1.0]), "TRANNNC": np.array([5.0])}
        N, T = active_neighbors(init, {}, np.array([True, True]), (2, 1, 1))
        np.testing.assert_array_equal(N, [[0, 1], [1, 0]])
        N3, T3, ix = active_neighbors(init, {}, np.array([True, True]), (2, 1, 1), nargout=3)
        np.testing.assert_array_equal(N3, [[0, 1], [0, 1]])
        assert ix.sN[0] == -1
        print("ok test_the_visualisation_path_filters_without_reorienting")

    test_the_simulation_grid_reproduces_eclipses_pore_volume()
    test_the_visualisation_path_filters_without_reorienting()
    print("\nAll tests passed!")
