"""Grid coarsening and flow-based upscaling (MRST coarsegrid + upscaling).

A fine geological model is partitioned into coarse blocks; each block's
effective permeability is then found by *solving flow through the block*
rather than by averaging. For each axis d, a unit pressure drop is imposed
across the block with the four lateral sides sealed, and

    K_d = v_d / (dp / L_d),        v_d = (total flux out) / (outlet area)

with the fluid viscosity set to 1 Pa s (upscalePermeabilityFixed). This
"fixed" (sealed-sides) method is the standard workhorse: it is exact for
layered media and brackets the true effective permeability between the
harmonic and arithmetic averages.

Porosity is additive, so it upscales by pore-volume conservation.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.gridprocessing.tensor_grid import tensor_grid
from pymrst.gridprocessing.geometry import compute_geometry
from pymrst.modules.incomp import incomp_tpfa, init_single_fluid
from pymrst.params.rock import Rock, make_rock, pore_volume
from pymrst.params.wells_and_bc import add_bc
from pymrst.solvers.state import init_res_sol
from pymrst.solvers.trans import compute_trans


def partition_cart_grid(cart_dims: Sequence[int],
                        part_dims: Sequence[int]) -> np.ndarray:
    """Uniform Cartesian partition (partitionCartGrid), 0-based block index."""
    cart = list(cart_dims) + [1] * (3 - len(cart_dims))
    part = list(part_dims) + [1] * (3 - len(part_dims))
    nx, ny, nz = cart
    cnx, cny, cnz = part
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                          indexing="ij")
    bi = np.minimum((i * cnx) // nx, cnx - 1)
    bj = np.minimum((j * cny) // ny, cny - 1)
    bk = np.minimum((k * cnz) // nz, cnz - 1)
    p = bi + cnx * (bj + cny * bk)
    return np.ravel(p, order="F").astype(np.int64)


@dataclass
class CoarseGrid:
    """A coarse grid and its mapping back to the fine model."""

    g: Grid                       # the coarse grid itself
    partition: np.ndarray         # (nc_fine,) coarse block of each fine cell
    n_blocks: int

    def blocks(self):
        """Fine cells of each coarse block, in block order."""
        order = np.argsort(self.partition, kind="stable")
        counts = np.bincount(self.partition, minlength=self.n_blocks)
        out, start = [], 0
        for c in counts:
            out.append(order[start:start + c])
            start += c
        return out


def generate_coarse_grid(g: Grid, partition: np.ndarray) -> CoarseGrid:
    """Build the coarse grid implied by a Cartesian partition."""
    assert g.cartDims is not None, "coarsening needs a Cartesian fine grid"
    partition = np.asarray(partition, dtype=np.int64).ravel()
    n_blocks = int(partition.max()) + 1

    # coarse block extents follow the fine coordinate lines
    dims = list(g.cartDims) + [1] * (3 - len(g.cartDims))
    ijk = np.array(np.unravel_index(np.arange(g.cells.num), dims, order="F")).T
    lines = []
    for d in range(len(g.cartDims)):
        nodes = np.unique(g.nodes.coords[:, d])
        # block boundaries: where the partition changes along axis d
        edges = [nodes[0]]
        for idx in range(1, dims[d]):
            prev = partition[np.flatnonzero(ijk[:, d] == idx - 1)[0]]
            cur = partition[np.flatnonzero(ijk[:, d] == idx)[0]]
            if prev != cur:
                edges.append(nodes[idx])
        edges.append(nodes[-1])
        lines.append(np.asarray(edges))

    cg = compute_geometry(tensor_grid(*lines))
    assert cg.cells.num == n_blocks, (
        f"partition implies {n_blocks} blocks but the coarse grid has "
        f"{cg.cells.num}; only rectangular Cartesian partitions are supported"
    )
    return CoarseGrid(g=cg, partition=partition, n_blocks=n_blocks)


def upscale_poro(g: Grid, rock: Rock, cg: CoarseGrid) -> np.ndarray:
    """Volume-weighted porosity (additive: pore volume is conserved)."""
    pv = pore_volume(g, rock)
    num = np.bincount(cg.partition, weights=pv, minlength=cg.n_blocks)
    den = np.bincount(cg.partition, weights=g.cells.volumes,
                      minlength=cg.n_blocks)
    return num / den


def _block_subgrid(g: Grid, cells: np.ndarray
                   ) -> Tuple[Grid, np.ndarray]:
    """The rectangular sub-grid spanned by a Cartesian block of cells."""
    dims = list(g.cartDims) + [1] * (3 - len(g.cartDims))
    nd = len(g.cartDims)
    ijk = np.array(np.unravel_index(cells, dims, order="F")).T[:, :nd]
    lo, hi = ijk.min(axis=0), ijk.max(axis=0)

    lines = []
    for d in range(nd):
        nodes = np.unique(g.nodes.coords[:, d])
        lines.append(nodes[lo[d]:hi[d] + 2])
    sub = compute_geometry(tensor_grid(*lines))

    # order the fine cells the way the sub-grid orders its own
    sub_dims = [hi[d] - lo[d] + 1 for d in range(nd)]
    local = ijk - lo
    lin = np.ravel_multi_index(tuple(local.T), sub_dims, order="F")
    order = np.argsort(lin)
    assert sub.cells.num == cells.size, (
        "coarse block is not a rectangular box of fine cells"
    )
    return sub, cells[order]


def upscale_perm_fixed(g: Grid, rock: Rock, cg: CoarseGrid,
                       dp: float = 1.0) -> np.ndarray:
    """Flow-based permeability upscaling with sealed sides.

    Returns (n_blocks, griddim) diagonal permeabilities.
    """
    nd = g.griddim
    fluid = init_single_fluid(1.0, 1.0)          # mu = 1 Pa s, as in MRST
    kk = np.zeros((cg.n_blocks, nd))

    for b, cells in enumerate(cg.blocks()):
        sub, order = _block_subgrid(g, cells)
        srock = make_rock(sub, rock.perm[order], rock.poro[order, None])
        t = compute_trans(sub, srock)

        bd = np.any(sub.faces.neighbors == -1, axis=1)
        fc = sub.faces.centroids
        for d in range(nd):
            lo = fc[:, d].min()
            hi = fc[:, d].max()
            inlet = np.flatnonzero(bd & (fc[:, d] < lo + 1e-9))
            outlet = np.flatnonzero(bd & (fc[:, d] > hi - 1e-9))
            bc = add_bc(None, inlet, "pressure", dp)
            bc = add_bc(bc, outlet, "pressure", 0.0)
            st = incomp_tpfa(init_res_sol(sub, 0.0), sub, t, fluid, bc=bc)

            flux = float(np.abs(st.flux[outlet]).sum())
            area = float(sub.faces.areas[outlet].sum())
            length = hi - lo
            kk[b, d] = (flux / area) / (dp / length)
    return kk


def upscale_rock(g: Grid, rock: Rock, cg: CoarseGrid) -> Rock:
    """Upscaled rock: flow-based permeability, volume-weighted porosity."""
    return Rock(perm=upscale_perm_fixed(g, rock, cg),
                poro=upscale_poro(g, rock, cg))


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid
    from pymrst.utils.units import DARCY, MILLI

    def test_partition_cart_grid():
        """A uniform partition groups fine cells into equal blocks."""
        p = partition_cart_grid([4, 4, 2], [2, 2, 1])
        assert p.size == 32 and p.max() == 3
        assert np.bincount(p).tolist() == [8, 8, 8, 8]
        # cell (0,0,0) and (1,1,1) share block 0
        assert p[0] == 0 and p[4 + 1] == 0
        print("ok test_partition_cart_grid")

    def test_homogeneous_upscales_to_itself():
        """A homogeneous block upscales to its own permeability."""
        g = compute_geometry(cart_grid([8, 8, 4], [80.0, 80.0, 40.0]))
        k = 250 * MILLI * DARCY
        rock = make_rock(g, k, 0.2)
        cg = generate_coarse_grid(g, partition_cart_grid([8, 8, 4],
                                                         [2, 2, 1]))
        assert cg.n_blocks == 4
        kk = upscale_perm_fixed(g, rock, cg)
        assert np.allclose(kk, k, rtol=1e-10), kk
        assert np.allclose(upscale_poro(g, rock, cg), 0.2, rtol=1e-12)
        print("ok test_homogeneous_upscales_to_itself")

    def test_layered_medium_is_exact():
        """Flow along layers gives the arithmetic mean, across them the
        harmonic mean - the two limits the fixed method reproduces exactly."""
        nx, nz = 1, 2
        g = compute_geometry(cart_grid([nx, 1, nz], [10.0, 10.0, 10.0]))
        k1, k2 = 100 * MILLI * DARCY, 1 * MILLI * DARCY
        perm = np.array([k1, k2])            # two layers of equal thickness
        rock = make_rock(g, perm[:, None], 0.2)
        cg = generate_coarse_grid(g, np.zeros(2, dtype=np.int64))
        kk = upscale_perm_fixed(g, rock, cg)[0]

        arith = 0.5 * (k1 + k2)              # flow parallel to the layers
        harm = 2.0 / (1.0 / k1 + 1.0 / k2)   # flow across the layers
        assert np.isclose(kk[0], arith, rtol=1e-10), (kk[0], arith)
        assert np.isclose(kk[1], arith, rtol=1e-10)
        assert np.isclose(kk[2], harm, rtol=1e-10), (kk[2], harm)
        print(f"ok test_layered_medium_is_exact (parallel {kk[0]/(MILLI*DARCY):.1f} mD, "
              f"across {kk[2]/(MILLI*DARCY):.2f} mD)")

    def test_upscaled_perm_is_bracketed():
        """A random block upscales to something between the harmonic and
        arithmetic averages."""
        rng = np.random.default_rng(7)
        g = compute_geometry(cart_grid([6, 6, 6], [60.0, 60.0, 60.0]))
        perm = 10 ** rng.uniform(-14.0, -12.0, g.cells.num)
        rock = make_rock(g, perm[:, None], 0.2)
        cg = generate_coarse_grid(g, np.zeros(g.cells.num, dtype=np.int64))
        kk = upscale_perm_fixed(g, rock, cg)[0]
        harm = 1.0 / np.mean(1.0 / perm)
        arith = np.mean(perm)
        assert np.all(kk > harm) and np.all(kk < arith), (kk, harm, arith)
        print("ok test_upscaled_perm_is_bracketed")

    def test_coarse_grid_geometry():
        """The coarse grid conserves volume and has the right block count."""
        g = compute_geometry(cart_grid([12, 8, 4], [120.0, 80.0, 40.0]))
        cg = generate_coarse_grid(g, partition_cart_grid([12, 8, 4],
                                                          [3, 2, 2]))
        assert cg.n_blocks == 12 and cg.g.cells.num == 12
        assert np.isclose(cg.g.cells.volumes.sum(), g.cells.volumes.sum())
        for b, cells in enumerate(cg.blocks()):
            assert np.isclose(cg.g.cells.volumes[b],
                              g.cells.volumes[cells].sum(), rtol=1e-12)
        print("ok test_coarse_grid_geometry")

    test_partition_cart_grid()
    test_homogeneous_upscales_to_itself()
    test_layered_medium_is_exact()
    test_upscaled_perm_is_bracketed()
    test_coarse_grid_geometry()
    print("\nAll tests passed!")
