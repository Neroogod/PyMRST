"""Coupled poroelasticity (MRST `example_poroelasticity`).

A pressurized, producing reservoir solved as one monolithic system in which
mechanics and flow are advanced together:

    [ A          -alpha*gradP ] [ u ]   [ rhs_s - fbc ]
    [ alpha*div   ct + dt*Af  ] [ p ] = [ rhsf        ]

The two off-diagonal blocks are the coupling: `gradP` turns pressure into a
force on the skeleton, and `div` turns skeleton dilation into pore-volume
change. Both fluid terms are present - the interior `div' p` through `gradP`,
and the boundary traction through `add_fluid_contrib_mech_vem`, which is what
makes the pressure boundary act on the mechanics at all.

Being monolithic, each step is a single linear solve; there is no
mechanics/flow iteration to converge.
"""

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.incomp.fluid import init_single_fluid
from pymrst.modules.vemmech.models.linear_compressibility_model import lincomp_tpfa
from pymrst.modules.vemmech.utils.add_fluid_contrib_mech_vem import (
    add_fluid_contrib_mech_vem,
)
from pymrst.modules.vemmech.vem import ElasticBC, create_augmented_grid, enu2c_2d
from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast
from pymrst.params.rock.make_rock import make_rock
from pymrst.params.rock.pore_volume import pore_volume
from pymrst.params.wells_and_bc.add_well import add_well
from pymrst.params.wells_and_bc.pside import pside
from pymrst.solvers.compute_trans import compute_trans
from pymrst.solvers.state_types import State
from pymrst.utils.mcolon import mcolon
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

GRAVITY = 9.80665


@dataclass
class PoroelasticResult:
    grid: object
    u0: np.ndarray
    u: np.ndarray
    pressure: np.ndarray
    times: List[float] = field(default_factory=list)
    pressure_history: List[np.ndarray] = field(default_factory=list)
    settlement_history: List[float] = field(default_factory=list)


def _side_nodes(grid, side: str):
    faces = np.asarray(pside(None, grid, side, 0.0).face, dtype=np.int64)
    inodes = mcolon(grid.faces.nodePos[faces], grid.faces.nodePos[faces + 1])
    return faces, np.unique(np.asarray(grid.faces.nodes, dtype=np.int64)[inodes])


def run_example_poroelasticity(
    *,
    L: Sequence[float] = (10000.0, 2000.0),
    cart_dims: Sequence[int] = (20, 20),
    E: float = 1.0e9,
    nu: float = 0.2,
    density: float = 3000.0,
    steps: int = 10,
    dt: float = DAY,
    initial_pressure: float = 100.0 * BARSA,
    bhp: float = 3000.0 * BARSA,
    boundary_pressure: float = 10.0 * BARSA,
) -> PoroelasticResult:
    """Run the coupled poroelastic problem and return its history."""
    L = [float(v) for v in L]
    grid = create_augmented_grid(compute_geometry(cart_grid(list(cart_dims), L)))
    nc = grid.cells.num
    C = enu2c_2d(np.full(nc, E), np.full(nc, nu))
    load = lambda x: -(GRAVITY * density) * np.tile(  # noqa: E731
        [0.0, 1.0], (np.asarray(x).shape[0], 1))

    # sides roll, the top is free and carries a bell-shaped overburden
    left_f, left = _side_nodes(grid, "Left")
    right_f, right = _side_nodes(grid, "Right")
    back_f, back = _side_nodes(grid, "Back")
    front_f, _ = _side_nodes(grid, "Front")

    nodes = np.concatenate([left, right, back])
    mask = np.ones((nodes.size, 2), dtype=bool)
    mask[:left.size + right.size, 1] = False             # vertical sides roll
    mask[left.size + right.size:, 0] = False             # base slides

    sigma = L[1] / 10.0
    centroids = np.asarray(grid.faces.centroids, dtype=float)[front_f]
    magnitude = (100.0 * BARSA
                 * np.exp(-(((centroids[:, 0] - L[0] / 2.0)) / sigma) ** 2)
                 + 10.0 * BARSA)
    el_bc = ElasticBC(disp_nodes=nodes, disp_mask=mask,
                      disp_vals=np.zeros((nodes.size, 2)),
                      force_faces=front_f,
                      force=magnitude[:, None] * np.array([0.0, -1.0]))

    u0, extra = vem_lin_elast(grid, C, el_bc, load)
    disc = extra["disc"]
    isdirdofs = disc["isdirdofs"]
    free = ~isdirdofs

    # a low-permeability cap over the upper four fifths of the section
    perm = np.full(cart_dims, 1.0 * DARCY, dtype=float)
    perm[:, int(cart_dims[1] // 5):] = 1.0 * MILLI * DARCY
    rock = make_rock(grid, perm=perm.reshape(-1, 1, order="F"),
                     poro=np.full((nc, 1), 0.1))
    rock.alpha = np.ones(nc)
    rock.cr = 1e-4 / BARSA
    fluid = init_single_fluid(mu=1.0 * CENTI * POISE, rho=1000.0)

    trans = compute_trans(grid, rock)
    pv = pore_volume(grid, rock)
    pressure = np.full(nc, initial_pressure)
    state = State(pressure=pressure.copy(), s=np.ones(nc),
                  flux=np.zeros(grid.faces.num))

    centre = np.asarray([5000.0, 200.0])
    well_cell = int(np.argmin(((grid.cells.centroids - centre) ** 2).sum(axis=1)))
    wells = add_well(None, grid, rock, [well_cell], well_type="bhp", val=bhp)
    fluid_bc = pside(None, grid, "Front", boundary_pressure)
    fbc = add_fluid_contrib_mech_vem(grid, fluid_bc, rock, isdirdofs)

    state = lincomp_tpfa(dt, state, grid, trans, pv, fluid, rock,
                         matrix_output=True, wells=wells, bc=fluid_bc)
    flow_matrix = sp.csr_matrix(state.A)
    flow_rhs = np.asarray(state.rhs, dtype=float)
    ct = sp.csr_matrix(state.ct)

    alpha = float(rock.poro[0])
    n_free = int(free.sum())
    n_extra = flow_matrix.shape[0] - nc                  # well/control rows
    coupling_up = sp.hstack([-alpha * disc["gradP"],
                             sp.csr_matrix((n_free, n_extra))])
    coupling_low = sp.vstack([alpha * disc["div"],
                              sp.csr_matrix((n_extra, n_free))])
    system = sp.bmat([[disc["A"], coupling_up],
                      [coupling_low, ct + dt * flow_matrix]], format="csc")

    x = np.zeros(n_free + flow_matrix.shape[0])
    x[n_free:n_free + nc] = pressure
    x[:n_free] = u0.reshape(-1)[free]

    result = PoroelasticResult(grid=grid, u0=u0, u=u0.copy(),
                               pressure=pressure.copy())
    displacement = np.zeros(isdirdofs.size)
    for step in range(steps):
        rhs_flow = np.zeros(flow_matrix.shape[0])
        rhs_flow[:nc] = (flow_rhs[:nc] * dt + ct[:nc, :nc] @ pressure
                         + alpha * (disc["div"] @ x[:n_free]))
        rhs_flow[nc:] = flow_rhs[nc:]
        x = spla.spsolve(system, np.concatenate([disc["rhs"] - fbc, rhs_flow]))

        pressure = x[n_free:n_free + nc]
        displacement[isdirdofs] = disc["V_dir"][isdirdofs]
        displacement[free] = x[:n_free]
        result.u = displacement.reshape(-1, grid.griddim)
        result.pressure = pressure.copy()
        result.times.append((step + 1) * dt)
        result.pressure_history.append(pressure.copy())
        result.settlement_history.append(float(result.u[:, 1].min()))
    return result


if __name__ == "__main__":
    result = run_example_poroelasticity()
    print(f"coupled poroelasticity, {result.grid.cells.num} cells, "
          f"{len(result.times)} steps of {DAY:.0f} s")
    print(f"  initial elastic settlement : {result.u0[:, 1].min():+.4e} m")
    for index in (0, len(result.times) // 2, len(result.times) - 1):
        print(f"  t = {result.times[index] / DAY:5.1f} d  "
              f"p = {result.pressure_history[index].min() / BARSA:8.2f} .. "
              f"{result.pressure_history[index].max() / BARSA:9.2f} bar   "
              f"settlement {result.settlement_history[index]:+.4e} m")
    assert np.all(np.isfinite(result.u))
    assert np.all(np.isfinite(result.pressure))
    print("\nExample ran.")
