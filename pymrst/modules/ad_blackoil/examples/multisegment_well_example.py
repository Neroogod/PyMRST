"""MRST ``ad-blackoil/examples/multisegmentWellExample.m``.

SPE1 is run twice, with the same gas injector and two versions of a producer
completed in six cells of the middle layer: a standard well, and a
multisegment well (`convert2MSWell`) of 13 nodes - the top node, six tubing
nodes 300 m apart and six valve nodes, one per perforation. Tubing segments
lose pressure to wellbore friction (`wellBoreFriction`), valve segments to
thirty 2.5 mm nozzles each (`nozzleValve`). The figures compare the node
pressures at the first and last step, the well curves of both runs, the gas
front, and the pressure in the completed cells, the valves and the tubing
over time.

`multisegment_well_example` is the port. `build_multisegment_well_example`
and `run_multisegment_well_example`, below, are PyMRST's earlier version on
its own multisegment well type; their callers keep them.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_blackoil.examples.utils.schedule import ExampleResult, run_schedule
from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
from pymrst.modules.ad_core import State
from pymrst.params import NORM_GRAVITY, add_well
from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1_ad
from pymrst.params.wells_and_bc import MultisegmentWell, WellSegment
from pymrst.utils.units import DAY


def multisegment_well_example(plot: bool = True) -> SimpleNamespace:
    """MRST `multisegmentWellExample`."""
    import scipy.sparse as sp

    from pymrst.modules.ad_blackoil.models import ThreePhaseBlackOilModel
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.models.facilities.combine_ms_with_regular_wells import (
        combine_ms_with_regular_wells,
    )
    from pymrst.modules.ad_core.models.facilities.nozzle_valve import nozzle_valve
    from pymrst.modules.ad_core.models.facilities.well_bore_friction import (
        well_bore_friction,
    )
    from pymrst.modules.ad_core.utils.convert2_ms_well import convert2_ms_well
    from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
        convert_deck_schedule_to_mrst,
    )
    from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1
    from pymrst.utils.adi import vertcat_adi

    # Set up the SPE1 model and its schedule
    G, rock, fluid, deck, state = setup_spe1()
    nx, ny = int(G.cartDims[0]), int(G.cartDims[1])
    model = ThreePhaseBlackOilModel(G, rock, fluid, inputdata=deck)
    model.extraStateOutput = True
    schedule = convert_deck_schedule_to_mrst(model, deck)
    # The producer perforates cells 2 to 7 of the middle layer (MRST's
    # 1-based `c = nx*ny + (2:7)'`)
    c = nx*ny + np.arange(1, 7)
    if plot:
        _plot_layout(G, c)
    # A standard producer
    prodS = add_well([], G, rock, c, name="prod", ref_depth=G.cells.centroids[0, 2],
                     well_type="rate", val=-8e5/DAY)[0]
    # The multisegment producer: node 0 is the top, nodes 1-6 the tubing and
    # nodes 7-12 the valves, one per perforation
    topo = np.array([[1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7],
                     [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]).T - 1
    cell2node = sp.csr_matrix((np.ones(6), (np.arange(7, 13), np.arange(6))), shape=(13, 6))
    lengths = np.concatenate([300*np.ones(6), np.full(6, np.nan)])
    diam = np.concatenate([0.1*np.ones(6), np.full(6, np.nan)])
    depths = G.cells.centroids[c[np.r_[0, 0:6, 0:6]], 2]
    vols = np.ones(13)
    prodMS = convert2_ms_well(prodS, cell2node=cell2node, topo=topo, G=G, vol=vols,
                              nodeDepth=depths, segLength=lengths, segDiam=diam)
    # Segment pressure drops: friction in the tubing, nozzles in the valves
    wbix, vix = np.arange(6), np.arange(6, 12)
    roughness, nozzleD, discharge, nValves = 1e-4, .0025, .7, 30
    prodMS.segments.flowModel = lambda v, rho, mu: vertcat_adi(
        well_bore_friction(v[wbix], rho[wbix], mu[wbix], prodMS.segments.diam[wbix],
                           prodMS.segments.length[wbix], roughness, "massRate"),
        nozzle_valve(v[vix]/nValves, rho[vix], nozzleD, discharge, "massRate"))
    # A gas injector in the corner
    inj = add_well([], G, rock, 99, name="inj", well_type="rate",
                   compi=np.array([0, 0, 1]), val=2.5e6/DAY,
                   ref_depth=G.cells.centroids[0, 2])[0]
    # Run with the standard producer
    W_simple = [inj, prodS]
    schedule.controls[0].W = W_simple
    wellSolsSimple, statesSimple, reportSimple = simulate_schedule_ad(state, model, schedule)
    # Run with the multisegment producer
    W = combine_ms_with_regular_wells(inj, prodMS)
    schedule.controls[0].W = W
    model = model.validate_model()
    model.FacilityModel = model.FacilityModel.setup_wells(W)
    print(model.FacilityModel.WellModels[0])
    print(model.FacilityModel.WellModels[1])
    wellSols, states, report = simulate_schedule_ad(state, model, schedule)
    out = SimpleNamespace(G=G, model=model, schedule=schedule, c=c, W=W, prodS=prodS,
                          wellSols=wellSols, states=states, report=report,
                          wellSolsSimple=wellSolsSimple, statesSimple=statesSimple,
                          reportSimple=reportSimple)
    if plot:
        _plot_multisegment(out, [inj, prodS])
    return out


def _plot_layout(G, c: np.ndarray) -> None:
    """The six completed cells in the reservoir and a sketch of the well:
    the tubing above them, the valves below, and the riser at the heel."""
    import pyvista as pv

    from pymrst.modules.ad_blackoil.examples.utils.figures import set_view
    from pymrst.plotting.plot_grid import plot_grid

    p = pv.Plotter()
    flag = np.ones(G.cells.num, dtype=bool)
    flag[np.r_[1, c]] = False
    plot_grid(G, flag, color=(1, 1, .7), plotter=p, show=False)
    plot_grid(G, c, color="red", opacity=.3, edge_color="red", plotter=p, show=False)
    x, y, z = (G.cells.centroids[c, k] for k in range(3))
    tubing = np.column_stack([x, y, z + 1.5])
    valves = np.column_stack([x, y, z - 1.5])
    riser = np.array([[x[0], y[0], z[0]], [x[0], y[0], z[0] - 15]])
    for line in (riser, tubing, valves, *(np.vstack([t, v]) for t, v in zip(tubing, valves))):
        p.add_mesh(pv.lines_from_points(line), color="red", line_width=2)
    p.add_points(np.vstack([tubing, valves]), color=(.6, .6, .6), point_size=7,
                 render_points_as_spheres=True)
    _matlab_box(p, G)
    set_view(p, (-30, 25))
    p.show()


def _matlab_box(plotter, G) -> None:
    """MATLAB's 3-D axes after `plotGrid`: depth downwards (`ZDir` reversed,
    a mirror, so a scale rather than a camera turn) and the box stretched to
    roughly a cube, which lifts a thin reservoir off the page."""
    extent = np.ptp(np.asarray(G.nodes.coords, dtype=float), axis=0)
    plotter.set_scale(zscale=-max(extent[0], extent[1])/extent[2])


def _animate_gas(G, c: np.ndarray, states, wells) -> None:
    """The gas front, cell by cell, inside the reservoir outline with the
    completed cells and the wells (MRST's last figure)."""
    import pyvista as pv

    from pymrst.modules.ad_blackoil.examples.utils.figures import set_view
    from pymrst.plotting._pyvista_grid import to_pyvista
    from pymrst.plotting.plot_cell_data import plot_cell_data
    from pymrst.plotting.plot_well import plot_well

    p = pv.Plotter()
    p.add_mesh(to_pyvista(G).outline(), color="black")
    p.add_mesh(to_pyvista(G, c), style="wireframe", color="red")
    plot_well(G, wells, plotter=p)
    _matlab_box(p, G)
    set_view(p, (37.5, 34))
    interactive = not pv.OFF_SCREEN
    if interactive:
        p.show(interactive_update=True, auto_close=False)
    for state in states:
        sg = np.asarray(state["s"])[:, 2]
        if np.sum(sg > 1e-5) > 0:
            plot_cell_data(G, sg, np.flatnonzero(sg > 1e-4), plotter=p, clim=(0, .5),
                           show=False, name="sg")
            if interactive:
                p.update()
    if interactive:
        p.show()


def _plot_multisegment(out: SimpleNamespace, wells) -> None:
    """MRST's figures: the well in the grid, node pressures at the first and
    last step, both runs' well curves, the gas front, and pressure surfaces."""
    import matplotlib.pyplot as plt

    from pymrst.modules.ad_blackoil.examples.utils.surf_with_outline import (
        surf_with_outline,
    )
    from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
    from pymrst.utils.units import BARSA

    G, c, wellSols, states = out.G, out.c, out.wellSols, out.states
    # Node pressures at the first and the last step
    plt.figure()
    for k in (0, len(wellSols) - 1):
        w = wellSols[k][1]
        plt.plot(np.concatenate([[w.bhp], w.nodePressure[:6]])/BARSA, "-o", linewidth=2,
                 label=f"Step {k + 1}")
    plt.legend(loc="upper left")
    plt.xlabel("Well node")
    plt.ylabel("Pressure [bar]")
    # Well curves of both runs
    plot_well_sols([wellSols, out.wellSolsSimple], out.report["ReservoirTime"],
                   datasetnames=["Multisegment", "Standard"])
    # The gas front
    _animate_gas(G, c, states, wells)
    # Pressure in the completed cells, the valves and the tubing over time
    xx, tt = np.meshgrid(np.arange(1, 7), np.asarray(out.report["ReservoirTime"])/DAY)
    pr = np.vstack([np.asarray(s["pressure"])[c] for s in states])
    pa = np.vstack([w[1].nodePressure[6:12] for w in wellSols])
    pw = np.vstack([w[1].nodePressure[0:6] for w in wellSols])
    plt.figure()
    for z in (pr, pa, pw):
        surf_with_outline(xx, tt, z/BARSA)
    ax = plt.gca()
    ax.view_init(10, 50)
    zlim = ax.get_zlim()
    pr = np.vstack([np.asarray(s["pressure"])[c] for s in out.statesSimple])
    pw = np.vstack([w[1].bhp + np.asarray(w[1].cdp).ravel() for w in out.wellSolsSimple])
    plt.figure()
    for z in (pr, pw):
        surf_with_outline(xx, tt, z/BARSA)
    ax = plt.gca()
    ax.set_zlim(zlim)
    ax.view_init(10, 50)
    plt.show()


@dataclass
class MultisegmentSPE1Case:
    simple_model: GenericBlackOilModelAD
    multisegment_model: GenericBlackOilModelAD
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
    # MRST's `setupSPE1()`: the deck read with the general reader, the
    # corner-point grid it describes, and the equilibrated state MRST wrote.
    grid, rock, struct, deck, reservoir_state = setup_spe1_ad()
    from pymrst.modules.ad_props.props.deck_fluid import deck_fluid_object
    fluid = deck_fluid_object(struct)
    nx, ny, _ = grid.cartDims
    completion_cells = nx * ny + np.arange(1, 7, dtype=np.int64)
    reference_depth = float(grid.cells.centroids[0, -1])

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

    simple_model = GenericBlackOilModelAD(
        grid, rock, fluid, [injector, producer], disgas=True,
        gravity=NORM_GRAVITY, tol_well=1e-8,
    )
    multisegment_model = GenericBlackOilModelAD(
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
    timesteps = np.asarray(deck["SCHEDULE"].step["val"], dtype=float)
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
    from pymrst.utils.units import BARSA

    CASE = build_multisegment_well_example(1)

    def test_the_reservoir_is_spe1_from_its_deck():
        """`setupSPE1`, not the bespoke SPE1 parser: the corner-point grid the
        deck describes and the equilibrated state MRST wrote."""
        model = CASE.simple_model
        assert model.G.cells.num == 300, model.G.cells.num
        assert tuple(model.G.cartDims) == (10, 10, 3)
        assert model.disgas is True, "SPE1 is a live-oil case"
        print("ok test_the_reservoir_is_spe1_from_its_deck")

    def test_the_completions_are_mrsts_six_cells():
        """MRST's `c = nx*ny + (2:7)'` is 1-based; the six cells are the
        second through seventh of the middle layer."""
        assert np.array_equal(CASE.completion_cells,
                              100 + np.arange(1, 7)), CASE.completion_cells
        print("ok test_the_completions_are_mrsts_six_cells")

    def test_the_multisegment_well_has_thirteen_nodes():
        """One root, six tubing, six valves - and the valves are the point."""
        ms = CASE.multisegment_model.facility.wells[1].multisegment
        assert len(ms.segments) == 13, len(ms.segments)
        models = [s.flow_model for s in ms.segments]
        assert models.count("tubing") == 6, models
        assert models.count("valve") == 6, models
        print("ok test_the_multisegment_well_has_thirteen_nodes")

    def test_the_valves_carry_mrsts_thirty_nozzles():
        ms = CASE.multisegment_model.facility.wells[1].multisegment
        valves = [s for s in ms.segments if s.flow_model == "valve"]
        assert all(v.valve_count == 30 for v in valves)
        assert all(np.isclose(v.valve_diameter, 0.0025) for v in valves)
        assert all(np.isclose(v.discharge_coefficient, 0.7) for v in valves)
        print("ok test_the_valves_carry_mrsts_thirty_nozzles")

    def test_the_tubing_segments_are_three_hundred_metres():
        ms = CASE.multisegment_model.facility.wells[1].multisegment
        tubing = [s for s in ms.segments if s.flow_model == "tubing"]
        assert all(np.isclose(s.length, 300.0) for s in tubing)
        assert all(np.isclose(s.diameter, 0.1) for s in tubing)
        assert all(np.isclose(s.roughness, 1e-4) for s in tubing)
        print("ok test_the_tubing_segments_are_three_hundred_metres")

    def test_the_well_rates_are_mrsts():
        injector, producer = CASE.simple_model.facility.wells
        assert np.isclose(injector.val*DAY, 2.5e6), injector.val*DAY
        assert np.isclose(producer.val*DAY, -8e5), producer.val*DAY
        assert np.allclose(injector.compi, [0.0, 0.0, 1.0]), "gas injection"
        print("ok test_the_well_rates_are_mrsts")

    def test_both_wells_share_a_reference_depth():
        """A bottom-hole pressure is only meaningful against a datum;
        comparing two well models at two datums compares nothing."""
        depth = float(CASE.simple_model.G.cells.centroids[0, -1])
        for model in (CASE.simple_model, CASE.multisegment_model):
            for well in model.facility.wells:
                assert np.isclose(well.refDepth, depth), well.name
        print("ok test_both_wells_share_a_reference_depth")

    RESULT = run_multisegment_well_example(1)

    def test_the_valves_drop_pressure():
        """What the example exists to show. The tubing rises smoothly from the
        heel; the six valve nodes sit tens of bar above the tubing they feed,
        and that step is the nozzles. Model them as tubing and it vanishes."""
        p = RESULT.node_pressure[-1]/BARSA
        tubing, valves = p[1:7], p[7:13]
        assert np.all(np.diff(tubing) > 0.0), tubing
        assert np.all(valves > tubing), (tubing, valves)
        drop = float(np.min(valves - tubing))
        assert drop > 5.0, drop
        print(f"ok test_the_valves_drop_pressure "
              f"(heel {p[0]:.0f} bar, toe {tubing[-1]:.0f}, "
              f"valves +{drop:.0f}-{float(np.max(valves - tubing)):.0f} bar)")

    def test_the_two_well_models_give_different_answers():
        """If they agreed, the multisegment model would be an expensive way to
        write the standard one."""
        simple = RESULT.simple.history["bhp"][-1, 1]
        ms = RESULT.multisegment.history["bhp"][-1, 1]
        assert not np.isclose(simple, ms, rtol=1e-3), (simple, ms)
        print(f"ok test_the_two_well_models_give_different_answers "
              f"(standard {simple/BARSA:.0f} bar, "
              f"multisegment {ms/BARSA:.0f} bar)")

    OUT = multisegment_well_example(plot=False)

    def test_the_example_runs_both_producers_over_the_schedule():
        # The standard run must not set up the caller's wells (MATLAB passes
        # the model by value), or the producer stays a SimpleWell here.
        from pymrst.modules.ad_core.ad_oo.multisegment_well import MultisegmentWell
        from pymrst.modules.ad_core.ad_oo.simple_well import SimpleWell
        models = OUT.model.FacilityModel.WellModels
        assert type(models[0]) is SimpleWell and isinstance(models[1], MultisegmentWell)
        assert len(OUT.wellSols) == len(OUT.wellSolsSimple) == 120
        assert OUT.W[1].isMS and not OUT.W[0].isMS
        print("ok test_the_example_runs_both_producers_over_the_schedule")

    def test_the_nozzles_hold_the_valve_nodes_above_the_tubing():
        w = OUT.wellSols[0][1]
        tubing, valves = w.nodePressure[:6], w.nodePressure[6:]
        assert np.all(valves > tubing), (tubing, valves)
        print(f"ok test_the_nozzles_hold_the_valve_nodes_above_the_tubing "
              f"(+{np.min(valves - tubing)/BARSA:.1f} to "
              f"+{np.max(valves - tubing)/BARSA:.1f} bar)")

    def test_node_mass_fractions_sum_to_one():
        for ws in OUT.wellSols:
            np.testing.assert_allclose(ws[1].nodeComp.sum(axis=1), 1.0, rtol=1e-12)
        print("ok test_node_mass_fractions_sum_to_one")

    test_the_example_runs_both_producers_over_the_schedule()
    test_the_nozzles_hold_the_valve_nodes_above_the_tubing()
    test_node_mass_fractions_sum_to_one()
    test_the_reservoir_is_spe1_from_its_deck()
    test_the_completions_are_mrsts_six_cells()
    test_the_multisegment_well_has_thirteen_nodes()
    test_the_valves_carry_mrsts_thirty_nozzles()
    test_the_tubing_segments_are_three_hundred_metres()
    test_the_well_rates_are_mrsts()
    test_both_wells_share_a_reference_depth()
    test_the_valves_drop_pressure()
    test_the_two_well_models_give_different_answers()
    print("\nAll tests passed!")


multisegmentWellExample = multisegment_well_example

__all__ = [
    "MultisegmentSPE1Case", "MultisegmentSPE1Result",
    "build_multisegment_well_example", "multisegmentWellExample",
    "multisegment_well_example", "run_multisegment_well_example",
]
