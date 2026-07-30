"""Open- versus closed-boundary black-oil sector model.

This is a runnable port of MRST ``blackoilSectorModelExample.m``.  It uses
the SPE1 live-oil PVT tables on a homogeneous Cartesian sector, produces from
the centre on oil-rate control with a BHP limit, and compares pressure support
from constant-pressure lateral boundaries with a sealed reservoir.

The defaults reproduce the MRST example (50 x 50 x 1 cells and ten years).
Pass smaller dimensions and a shorter duration for a quick validation run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.examples.utils.schedule import ExampleResult, run_schedule
from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import blackoil_status
from pymrst.modules.ad_core import State, rampup_timesteps
from pymrst.modules.ad_props import init_deck_blackoil_fluid
from pymrst.modules.deckformat import read_spe1_deck
from pymrst.params import make_rock, pore_volume, pside, vertical_well
from pymrst.utils.units import BARSA, DARCY, DAY, YEAR


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DECK = ROOT / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"


@dataclass
class SectorModelCase:
    open_model: GenericBlackOilModel
    closed_model: GenericBlackOilModel
    state0: State
    timesteps: np.ndarray
    producer_cell: int


@dataclass
class SectorModelResult:
    open_boundary: ExampleResult
    closed_boundary: ExampleResult
    producer_cell: int

    @property
    def gas_at_producer(self) -> dict[str, np.ndarray]:
        cell = self.producer_cell
        return {
            "open": self.open_boundary.history["sG"][:, cell],
            "closed": self.closed_boundary.history["sG"][:, cell],
        }


def build_blackoil_sector_model_example(
    dims=(50, 50, 1),
    physical_dims=(1000.0, 1000.0, 100.0),
    total_time: float = 10.0 * YEAR,
    max_timestep: float = 30.0 * DAY,
    rampup_steps: int = 8,
    deck_path=None,
) -> SectorModelCase:
    """Build both variants of the sector experiment without running them."""
    dims = tuple(int(v) for v in dims)
    if len(dims) != 3 or min(dims) < 1:
        raise ValueError("dims must contain three positive integers")
    if total_time <= 0.0 or max_timestep <= 0.0:
        raise ValueError("total_time and max_timestep must be positive")

    grid = compute_geometry(cart_grid(dims, physical_dims))
    rock = make_rock(grid, 0.3 * DARCY, 0.3)
    deck = read_spe1_deck(Path(deck_path) if deck_path is not None else DEFAULT_DECK)
    fluid = replace(init_deck_blackoil_fluid(deck, rock), kr_w=lambda sw: sw**2)

    pv = float(pore_volume(grid, rock).sum())
    oil_rate = -0.25 * pv / (10.0 * YEAR)
    i, j = int(np.ceil(dims[0] / 2.0)) - 1, int(np.ceil(dims[1] / 2.0)) - 1
    wells = vertical_well(
        None, grid, rock, i, j,
        well_type="orat", val=oil_rate, compi=np.full(3, 1.0 / 3.0),
        sign=-1, name="P1", lims={"bhp": 100.0 * BARSA},
    )
    producer_cell = int(wells[0].cells[0])

    p0 = np.full(grid.cells.num, 300.0 * BARSA)
    sw = np.full(grid.cells.num, 0.2)
    so = np.full(grid.cells.num, 0.8)
    sg = np.zeros(grid.cells.num)
    state0 = State(
        pressure=p0, sW=sw, sO=so, sG=sg,
        rs=np.full(grid.cells.num, 200.0),
        status=blackoil_status(sw, so, sg, disgas=True),
        qWs=np.zeros(1), qOs=np.array([oil_rate]), qGs=np.zeros(1),
        bhp=np.array([300.0 * BARSA]),
    )

    sat = np.array([0.2, 0.8, 0.0])
    bc = None
    for side in ("xmin", "xmax", "ymin", "ymax"):
        bc = pside(bc, grid, side, 300.0 * BARSA, sat=sat)
    bc.dissolution = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 200.0, 1.0]])

    open_model = GenericBlackOilModel(grid, rock, fluid, wells, disgas=True, bc=bc)
    # Each facility owns mutable well controls, so the two runs need separate
    # Well instances even though their initial controls are identical.
    closed_wells = [replace(w, lims=None if w.lims is None else dict(w.lims)) for w in wells]
    closed_model = GenericBlackOilModel(grid, rock, fluid, closed_wells, disgas=True)
    timesteps = rampup_timesteps(float(total_time), float(max_timestep), rampup_steps)
    return SectorModelCase(open_model, closed_model, state0, timesteps, producer_cell)


def run_blackoil_sector_model_example(**kwargs) -> SectorModelResult:
    """Run the pressure-supported and sealed sector simulations."""
    case = build_blackoil_sector_model_example(**kwargs)
    keys = ("pressure", "sW", "sO", "sG", "rs", "qOs", "qGs", "bhp")
    open_result = run_schedule(case.open_model, case.state0, case.timesteps,
                               history_keys=keys)
    closed_result = run_schedule(case.closed_model, case.state0, case.timesteps,
                                 history_keys=keys)
    return SectorModelResult(open_result, closed_result, case.producer_cell)


if __name__ == "__main__":
    # A laptop-friendly run; call without overrides to reproduce the full
    # 50 x 50 x 1, ten-year MRST setup.
    result = run_blackoil_sector_model_example(
        dims=(10, 10, 1), physical_dims=(200.0, 200.0, 20.0),
        total_time=90.0 * DAY, max_timestep=15.0 * DAY,
    )
    po = result.open_boundary.history["pressure"][-1].mean() / BARSA
    pc = result.closed_boundary.history["pressure"][-1].mean() / BARSA
    print(f"completed {len(result.open_boundary.states)} steps per case")
    print(f"final mean pressure: open={po:.3f} bar, closed={pc:.3f} bar")
