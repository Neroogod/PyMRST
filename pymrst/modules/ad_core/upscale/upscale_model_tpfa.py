"""MRST ``ad-core/upscale/upscaleModelTPFA.m``: a model coarsened onto the
grid of a partition.

The partition is checked (`processPartition`, `compressPartition`), the
coarse grid generated and its geometry computed (`generateCoarseGrid`,
`coarsenGeometry`), the rock upscaled - porosity volume-weighted, each
permeability component harmonically averaged over the block - and the coarse
transmissibility taken from that rock (`getFaceTransmissibility`), or from
`transCoarse` (face, half-face or interface values), or, with
`transFromRock` false, summed from the fine transmissibilities behind each
coarse face. The model keeps its class, fluid and settings; its grid, rock
and operators are replaced and its flow discretization reset.

The coarse grid is the port's grid view of the coarse grid
(`coarse_grid_as_grid`), carrying `partition` and the coarse grid's face
connections (`.coarse_grid`). Partitions are 0-based, and outside is -1 in
`faces.neighbors` (MRST: 0).

`upscale_model_tpfa_ad`, below, is PyMRST's earlier version for the legacy
models, kept for its callers. Its notes follow.

Coarsen an AD model onto a coarse TPFA grid (MRST `upscaleModelTPFA`).

Rebuilds the model on a coarse grid implied by a partition: the coarse rock keeps
the same total pore volume per block (volume-weighted porosity) and a harmonic
permeability average (the MRST `upscaleModelTPFA` default - resistances in series
add, so `1/k` averages), and the coarse transmissibility follows from that coarse
rock. The result is the same model class, same fluid and wells-less, ready to be
driven by an upscaled schedule and initialised by an upscaled state.

This is the AD-side companion to the flow-based `upscaling` module: it produces a
*runnable coarse model*, not just coarse properties. It carries the `partition`
so `upscale_state` and `upscale_schedule` can map fine quantities onto it.
"""

from typing import Any

import numpy as np

from pymrst.modules.upscaling.coarsegrid.generate_coarse_grid import (
    CoarseGrid,
    generate_coarse_grid,
)
from pymrst.modules.upscaling.flow.upscale_poro import upscale_poro


def upscale_model_tpfa(model: Any, partition: Any, *, validatePartition: bool = True,
                       transFromRock: bool = True, transCoarse: Any = None,
                       permCoarse: Any = None, neighborship: Any = None,
                       poroCoarse: Any = None) -> Any:
    """MRST `model = upscaleModelTPFA(model, partition, ...)`."""
    from pymrst.modules.ad_core.ad_oo.wrapper_model import value_copy_model
    G = model.G
    rock = model.rock
    CG = _get_grid(G, partition, validatePartition)
    rock_c = _get_rock(rock, CG, poroCoarse, permCoarse)
    Tc, Nc = _get_transmissibility(CG, rock_c, neighborship, transCoarse)
    if not transFromRock and transCoarse is None:
        cg = CG.coarse_grid
        cts = np.repeat(np.arange(cg.num_faces), np.diff(cg.conn_pos))
        Tc = np.bincount(cts, np.asarray(model.operators.t_all)[cg.fconn], minlength=cg.num_faces)
    model = value_copy_model(model)
    model.G = CG
    model.rock = rock_c
    model = model.setup_operators(CG, rock_c, neighbors=Nc, trans=Tc)
    model.FlowDiscretization = None
    return model


def _get_grid(G: Any, partition: Any, validate: bool) -> Any:
    from pymrst.modules.coarsegrid.compress_partition import compress_partition
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.process_partition import process_partition
    from pymrst.modules.coarsegrid.utils.coarse_grid_as_grid import coarse_grid_as_grid
    from pymrst.modules.coarsegrid.utils.coarsen_geometry import coarsen_geometry
    if hasattr(partition, "coarse_grid"):
        return partition
    partition = np.asarray(partition, dtype=np.int64).ravel()
    if validate:
        partition = process_partition(G, partition)
        partition = compress_partition(partition)
    cg = generate_coarse_grid(G, partition)
    coarsen_geometry(cg)
    return coarse_grid_as_grid(cg)


def _get_rock(rock: Any, CG: Any, poro_c: Any, perm_c: Any) -> Any:
    from pymrst.params.rock import make_rock
    p = np.asarray(CG.partition, dtype=np.int64)
    n = CG.cells.num
    vol = np.asarray(CG.coarse_grid.parent.cells.volumes, dtype=float)
    coarsevol = np.bincount(p, vol, minlength=n)
    counts = np.bincount(p, minlength=n).astype(float)
    if getattr(rock, "poro", None) is not None and poro_c is None:
        poro = np.asarray(rock.poro, dtype=float).ravel()
        if getattr(rock, "ntg", None) is not None:
            poro = poro*np.asarray(rock.ntg, dtype=float).ravel()
        poro_c = np.bincount(p, poro*vol, minlength=n)/coarsevol
    if getattr(rock, "perm", None) is not None and perm_c is None:
        perm = np.atleast_2d(np.asarray(rock.perm, dtype=float))
        perm = perm.T if perm.shape[0] != p.size else perm
        perm_c = np.column_stack([np.bincount(p, 1/perm[:, i], minlength=n)
                                  for i in range(perm.shape[1])])
        perm_c = 1/(perm_c/counts[:, None])
    rock_c = make_rock(CG, perm_c, poro_c)
    regions = getattr(rock, "regions", None)
    if regions:
        rock_c.regions = {}
        for t in ("saturation", "pvt"):
            if t in regions:
                rock_c.regions[t] = _coarsen_region(CG, regions[t])
    return rock_c


def _coarsen_region(CG: Any, reg_fine: Any) -> np.ndarray:
    import warnings
    p = np.asarray(CG.partition)
    reg_fine = np.asarray(reg_fine)
    out = np.zeros(CG.cells.num, dtype=reg_fine.dtype)
    for i in range(CG.cells.num):
        u = np.unique(reg_fine[p == i])
        if u.size > 1:
            warnings.warn(f"Multiple regions found for block {i + 1}, will take first "
                          "encountered value")
        out[i] = u[0]
    return out


def _get_transmissibility(CG: Any, rock_c: Any, neighborship: Any, transCoarse: Any):
    from pymrst.solvers.compute_trans import compute_trans
    from pymrst.solvers.get_face_transmissibility import get_face_transmissibility
    N = np.asarray(CG.faces.neighbors if neighborship is None else neighborship)
    Tc = None if transCoarse is None else np.asarray(transCoarse, dtype=float).ravel()
    nT = 0 if Tc is None else Tc.size
    cell_faces = np.asarray(CG.cells.faces)
    nHf = cell_faces.shape[0]
    nF = N.shape[0]
    intx = np.all(N >= 0, axis=1)
    nIF = int(intx.sum())
    if nT == 0:
        Tc = get_face_transmissibility(CG, rock_c)
    elif nT == nF:
        pass
    elif nT == nHf:
        faces = cell_faces[:, 0] if cell_faces.ndim == 2 else cell_faces
        Tc = 1/np.bincount(faces, 1/Tc, minlength=nF)
    elif nT == nIF:
        # MRST's: half-face transmissibilities, indexed by the face mask.
        Tc_dummy = np.asarray(compute_trans(CG, rock_c), dtype=float).ravel()
        Tc_intf = Tc
        Tc = np.zeros(nF)
        Tc[intx] = Tc_intf[intx] if Tc_intf.size == nF else Tc_intf
        Tc[~intx] = Tc_dummy[:nF][~intx]
    else:
        raise ValueError(f"Number of input transmissibility entries ({nT}) does not match "
                         f"either half face count ({nHf}), interface count ({nIF}) or "
                         f"facecount ({nF}).")
    return Tc, N[intx]


upscaleModelTPFA = upscale_model_tpfa


# ---------------------------------------------------------------------------
# PyMRST's earlier version.

def _harmonic_perm(g, rock, cg: CoarseGrid) -> np.ndarray:
    """Per-block harmonic mean of permeability, MRST's upscaleModelTPFA default.

    `perm_c = counts / sum(1/perm)` per column - the average that preserves flow
    resistance through cells in series.
    """
    perm = np.atleast_2d(np.asarray(rock.perm, dtype=float))
    if perm.shape[0] == 1 and g.cells.num > 1:
        perm = perm.T
    if perm.ndim == 1:
        perm = perm[:, None]
    p = cg.partition
    counts = np.bincount(p, minlength=cg.n_blocks)
    out = np.zeros((cg.n_blocks, perm.shape[1]))
    for i in range(perm.shape[1]):
        inv_sum = np.bincount(p, weights=1.0 / perm[:, i], minlength=cg.n_blocks)
        out[:, i] = counts / inv_sum
    return out


def upscale_model_tpfa_ad(model: Any, partition: np.ndarray, coarse_grid=None):
    """Return a coarse copy of `model` on the grid implied by `partition`.

    `partition` is a 0-based fine-to-coarse cell map. The coarse model is the
    same class as `model`, built with the coarse grid, coarse rock and the same
    fluid; it carries `partition` and the fine `CoarseGrid` on `.coarse_grid`.
    """
    from pymrst.gridprocessing import compute_geometry
    from pymrst.params.rock import make_rock

    partition = np.asarray(partition, dtype=np.int64).ravel()
    cg = coarse_grid if coarse_grid is not None else generate_coarse_grid(
        model.G, partition)
    if getattr(cg, "g", None) is not None:
        coarse_g = compute_geometry(cg.g)
    else:
        # `CoarseGrid.g` exists only for a *tensor-product* partition, which
        # `partition_cart_grid` gives and almost nothing else does. MRST builds
        # its coarse grid from `coarsenGeometry` for any partition, and this
        # did not - so a partition split along faults, which is what
        # `simulatorWorkflowExample` builds, raised on `None.faces`.
        from pymrst.modules.coarsegrid.utils.coarse_grid_as_grid import (
            coarse_grid_as_grid,
        )
        coarse_g = coarse_grid_as_grid(cg)

    poro_c = upscale_poro(model.G, model.rock, cg)
    perm_c = _harmonic_perm(model.G, model.rock, cg)
    rock_c = make_rock(coarse_g, perm_c, poro_c.reshape(-1, 1))

    fluid = getattr(model, "raw_fluid", model.fluid)
    coarse = type(model)(coarse_g, rock_c, fluid, [],
                         disgas=getattr(model, "disgas", False),
                         vapoil=getattr(model, "vapoil", False))
    coarse.partition = partition
    coarse.coarse_grid = cg
    return coarse



if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD,
    )
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.modules.upscaling.coarsegrid.partition_cart_grid import (
        partition_cart_grid,
    )
    from pymrst.params.rock import make_rock, pore_volume
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _fine():
        g = compute_geometry(cart_grid([8, 8, 1], [800.0, 800.0, 10.0]))
        rng = np.random.default_rng(0)
        perm = ((50 + 100 * rng.random(g.cells.num)) * MILLI * DARCY).reshape(-1, 1)
        poro = (0.2 + 0.1 * rng.random(g.cells.num)).reshape(-1, 1)
        rock = make_rock(g, perm, poro)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 3.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[0.0, 0.0], p_ref=200 * BARSA)
        return GenericBlackOilModelAD(g, rock, fluid, [])

    def test_coarse_cell_count_matches_partition():
        m = _fine()
        part = partition_cart_grid([8, 8, 1], [4, 4, 1])       # 2x2 blocks
        cm = upscale_model_tpfa_ad(m, part)
        assert cm.G.cells.num == 16, cm.G.cells.num
        assert np.array_equal(cm.partition, part)
        print("ok test_coarse_cell_count_matches_partition")

    def test_total_pore_volume_is_preserved():
        """The reason to weight porosity by volume: coarse pore volume equals
        the fine total, block by block and overall."""
        m = _fine()
        part = partition_cart_grid([8, 8, 1], [4, 4, 1])
        cm = upscale_model_tpfa_ad(m, part)
        pvf = pore_volume(m.G, m.rock)
        pvc = pore_volume(cm.G, cm.rock)
        assert np.isclose(pvc.sum(), pvf.sum(), rtol=1e-12)
        blockwise = np.bincount(part, weights=pvf)
        assert np.allclose(pvc, blockwise, rtol=1e-12)
        print("ok test_total_pore_volume_is_preserved")

    def test_coarse_perm_is_harmonic_mean():
        m = _fine()
        part = partition_cart_grid([8, 8, 1], [4, 4, 1])
        cm = upscale_model_tpfa_ad(m, part)
        perm_f = np.asarray(m.rock.perm).ravel()
        counts = np.bincount(part)
        expect = counts / np.bincount(part, weights=1.0 / perm_f)
        assert np.allclose(np.asarray(cm.rock.perm).ravel(), expect)
        print("ok test_coarse_perm_is_harmonic_mean")

    def test_coarse_model_assembles_operators():
        """The coarse model is runnable: its TPFA operators are well-formed."""
        m = _fine()
        part = partition_cart_grid([8, 8, 1], [2, 2, 1])       # 4x4 blocks
        cm = upscale_model_tpfa_ad(m, part)
        assert cm.operators.pv.shape[0] == cm.G.cells.num
        assert cm.operators.T.shape[0] == cm.operators.N.shape[0]
        assert np.all(cm.operators.T > 0)
        print("ok test_coarse_model_assembles_operators")

    test_coarse_cell_count_matches_partition()
    test_total_pore_volume_is_preserved()
    test_coarse_perm_is_harmonic_mean()
    test_coarse_model_assembles_operators()
    print("\nAll tests passed!")
