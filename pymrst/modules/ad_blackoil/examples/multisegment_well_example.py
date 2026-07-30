"""Full 13-node SPE1 multisegment producer workflow.

Ports MRST ``multisegmentWellExample.m`` without plotting.  The same SPE1
reservoir is run with a conventional six-completion producer and with a
13-node producer containing six tubing segments and six 30-nozzle valves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import (
    setup_spe1_tutorial,
)
from pymrst.modules.ad_blackoil.examples.utils.schedule import ExampleResult, run_schedule
from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
from pymrst.modules.ad_core import State
from pymrst.params import NORM_GRAVITY, add_well
from pymrst.params.wells_and_bc import MultisegmentWell, WellSegment
from pymrst.utils.units import DAY


@dataclass
class MultisegmentSPE1Case:
    simple_model: GenericBlackOilModel
    multisegment_model: GenericBlackOilModel
    state0: State
    timesteps: np.ndarray
    completion_cells: np.ndarray


@dataclass
class MultisegmentSPE1Result:
    simple: ExampleResult
    multisegment: ExampleResult
    completion_cells: np.ndarray

    @property
    def node_pressure(self) -> np.ndarray:
        """All 13 MS node pressures, with BHP as the root node."""
        return np.column_stack([
            self.multisegment.history["bhp"][:, 1],
            self.multisegment.history["ms_pN"],
        ])


def _full_topology(depth: float) -> MultisegmentWell:
    segments = [
        WellSegment(0, -1, depth=depth, diameter=0.1, volume=1.0,
                    flow_model="none")
    ]
    # Nodes 1--6 form the horizontal tubing string.
    for node in range(1, 7):
        segments.append(WellSegment(
            node, node - 1, length=300.0, depth=depth, diameter=0.1,
            roughness=1e-4, volume=1.0, flow_model="tubing",
        ))
    # Nodes 7--12 connect reservoir completions to tubing through valves.
    for completion in range(6):
        segments.append(WellSegment(
            7 + completion, 1 + completion, depth=depth, volume=1.0,
            flow_model="valve", valve_diameter=0.0025,
            discharge_coefficient=0.7, valve_count=30,
        ))
    return MultisegmentWell(segments, np.arange(7, 13, dtype=np.int64))


def build_multisegment_well_example(nsteps: int | None = 1) -> MultisegmentSPE1Case:
    deck, base, reservoir_state = setup_spe1_tutorial()
    grid, rock, fluid = base.G, base.rock, base.raw_fluid
    nx, ny, _ = grid.cartDims
    completion_cells = nx * ny + np.arange(1, 7, dtype=np.int64)
    reference_depth = float(grid.cells.centroids[0, 2])

    producer_rate = -8e5 / DAY
    injector_rate = 2.5e6 / DAY
    injector = add_well(
        None, grid, rock, np.array([99]), name="inj", well_type="rate",
        val=injector_rate, compi=np.array([0.0, 0.0, 1.0]), sign=1,
        ref_depth=reference_depth,
        gravity=np.array([0.0, 0.0, NORM_GRAVITY]),
    )[0]
    producer = add_well(
        None, grid, rock, completion_cells, name="prod", well_type="rate",
        val=producer_rate, compi=np.full(3, 1.0 / 3.0), sign=-1,
        ref_depth=reference_depth,
        gravity=np.array([0.0, 0.0, NORM_GRAVITY]),
    )[0]
    ms_producer = add_well(
        None, grid, rock, completion_cells, name="prod", well_type="rate",
        val=producer_rate, compi=np.full(3, 1.0 / 3.0), sign=-1,
        ref_depth=reference_depth,
        gravity=np.array([0.0, 0.0, NORM_GRAVITY]),
    )[0]
    ms_producer.multisegment = _full_topology(reference_depth)

    simple_model = GenericBlackOilModel(
        grid, rock, fluid, [injector, producer], disgas=True,
        gravity=NORM_GRAVITY, tol_well=1e-8,
    )
    multisegment_model = GenericBlackOilModel(
        grid, rock, fluid, [injector, ms_producer], disgas=True,
        gravity=NORM_GRAVITY, tol_well=1e-8,
    )
    pmean = float(np.mean(reservoir_state["pressure"]))
    rs0 = float(np.mean(reservoir_state["rs"][completion_cells]))
    producer_oil = producer_rate / (1.0 + rs0)
    producer_gas = rs0 * producer_oil
    state0 = State(
        pressure=reservoir_state["pressure"].copy(),
        sW=reservoir_state["sW"].copy(),
        sO=reservoir_state["sO"].copy(),
        sG=reservoir_state["sG"].copy(),
        rs=reservoir_state["rs"].copy(),
        status=reservoir_state["status"].copy(),
        qWs=np.zeros(2), qOs=np.array([0.0, producer_oil]),
        qGs=np.array([injector_rate, producer_gas]),
        bhp=np.array([pmean + 1e5, 100e5]),
    )
    timesteps = np.asarray(deck["tsteps"], dtype=float)
    if nsteps is not None:
        timesteps = timesteps[:int(nsteps)]
    return MultisegmentSPE1Case(
        simple_model, multisegment_model, state0, timesteps, completion_cells
    )


def run_multisegment_well_example(nsteps: int | None = 1) -> MultisegmentSPE1Result:
    case = build_multisegment_well_example(nsteps)
    common = ("pressure", "sW", "sG", "rs", "qWs", "qOs", "qGs", "bhp")
    simple = run_schedule(
        case.simple_model, case.state0, case.timesteps, history_keys=common,
    )
    multisegment = run_schedule(
        case.multisegment_model, case.state0, case.timesteps,
        history_keys=common + ("ms_pN", "ms_rW", "ms_rO", "ms_rG", "ms_vmix"),
    )
    return MultisegmentSPE1Result(simple, multisegment, case.completion_cells)


if __name__ == "__main__":
    result = run_multisegment_well_example(1)
    print("first-step node pressure [bar]:")
    print(result.node_pressure[-1] / 1e5)


__all__ = [
    "MultisegmentSPE1Case", "MultisegmentSPE1Result",
    "build_multisegment_well_example", "run_multisegment_well_example",
]
