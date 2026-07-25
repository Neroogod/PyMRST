"""Coarsen an AD model onto a coarse TPFA grid (MRST `upscaleModelTPFA`).

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


def upscale_model_tpfa(model: Any, partition: np.ndarray, coarse_grid=None):
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
    coarse_g = compute_geometry(cg.g)

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
        GenericBlackOilModel,
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
        return GenericBlackOilModel(g, rock, fluid, [])

    def test_coarse_cell_count_matches_partition():
        m = _fine()
        part = partition_cart_grid([8, 8, 1], [4, 4, 1])       # 2x2 blocks
        cm = upscale_model_tpfa(m, part)
        assert cm.G.cells.num == 16, cm.G.cells.num
        assert np.array_equal(cm.partition, part)
        print("ok test_coarse_cell_count_matches_partition")

    def test_total_pore_volume_is_preserved():
        """The reason to weight porosity by volume: coarse pore volume equals
        the fine total, block by block and overall."""
        m = _fine()
        part = partition_cart_grid([8, 8, 1], [4, 4, 1])
        cm = upscale_model_tpfa(m, part)
        pvf = pore_volume(m.G, m.rock)
        pvc = pore_volume(cm.G, cm.rock)
        assert np.isclose(pvc.sum(), pvf.sum(), rtol=1e-12)
        blockwise = np.bincount(part, weights=pvf)
        assert np.allclose(pvc, blockwise, rtol=1e-12)
        print("ok test_total_pore_volume_is_preserved")

    def test_coarse_perm_is_harmonic_mean():
        m = _fine()
        part = partition_cart_grid([8, 8, 1], [4, 4, 1])
        cm = upscale_model_tpfa(m, part)
        perm_f = np.asarray(m.rock.perm).ravel()
        counts = np.bincount(part)
        expect = counts / np.bincount(part, weights=1.0 / perm_f)
        assert np.allclose(np.asarray(cm.rock.perm).ravel(), expect)
        print("ok test_coarse_perm_is_harmonic_mean")

    def test_coarse_model_assembles_operators():
        """The coarse model is runnable: its TPFA operators are well-formed."""
        m = _fine()
        part = partition_cart_grid([8, 8, 1], [2, 2, 1])       # 4x4 blocks
        cm = upscale_model_tpfa(m, part)
        assert cm.operators.pv.shape[0] == cm.G.cells.num
        assert cm.operators.T.shape[0] == cm.operators.N.shape[0]
        assert np.all(cm.operators.T > 0)
        print("ok test_coarse_model_assembles_operators")

    test_coarse_cell_count_matches_partition()
    test_total_pore_volume_is_preserved()
    test_coarse_perm_is_harmonic_mean()
    test_coarse_model_assembles_operators()
    print("\nAll tests passed!")
