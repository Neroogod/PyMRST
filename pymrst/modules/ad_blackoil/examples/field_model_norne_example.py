"""MRST ``ad-blackoil/examples/fieldModelNorneExample.m``.

The Norne field model run as MRST runs it: the deck and the reference
results (`getNorneFieldDeckOPMData`), the grid, transmissibilities and
non-neighbour connections from OPM Flow's own EGRID/INIT
(`initGridFromEclipseOutput`), the rock from the deck, a disgas/vapoil
`GenericBlackOilModel` with the deck's hysteresis and threshold pressures,
`initEclipseProblemAD` with CPR, then the script's settings: looser
convergence tolerances, a generic facility model without crossflow and with
volume-balanced crossflow mixing, unsaturated-flag-free viscosities and
shrinkage factors, `rvReduction = 0.5`, update limits, relaxation and a
factor timestep selector. The schedule is run as a packed problem and the
well curves compared with OPM Flow (with and without hysteresis) and
ECLIPSE, and the states shown beside OPM's.

`field_model_norne_example` is the port. `nsteps` (not in MRST) runs only
the first report steps; the full schedule is 247 report steps on 44431
cells. `build_norne_field_model`, `run_field_model_norne_example` and the
other functions below are PyMRST's earlier version on its legacy models;
their callers keep them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from pymrst.modules.ad_blackoil.examples.utils.get_norne_field_deck_opm_data import (
    NorneReferenceOutputs, get_norne_field_deck_opm_data_ad,
)
from pymrst.modules.ad_blackoil.examples.utils.load_aquifer_eclipse_result import _collect
from pymrst.modules.deckformat.norne import NorneFieldDeck
from pymrst.params.rock import Rock
from pymrst.modules.ad_core import State
from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
    convert_deck_schedule_to_mrst_ad,
)
from pymrst.params import NORM_GRAVITY
from pymrst.utils.units import DARCY, MILLI


@dataclass
class NorneGridComponent:
    grid: object
    rock: Rock
    regions: dict[str, np.ndarray]


@dataclass
class NorneFieldCase:
    deck: NorneFieldDeck
    outputs: NorneReferenceOutputs
    components: list[NorneGridComponent]
    initial_state: object
    native_grid: object | None = None
    native_rock: Rock | None = None
    native_schedule: object | None = None

    @property
    def active_cells(self) -> int:
        return sum(component.grid.cells.num for component in self.components)

    @property
    def well_names(self) -> tuple[str, ...]:
        return self.deck.schedule.wells

    def field_history(self) -> dict[str, np.ndarray]:
        """Return field totals and time from the archived OPM summary."""
        summary = self.outputs.opm.summary
        if summary is None:
            raise ValueError("Norne OPM summary is unavailable")
        out = {"time": summary.time.copy()}
        for keyword in ("FOPR", "FWPR", "FGPR", "FWIR", "FGIR", "FOPT", "FWPT"):
            try:
                out[keyword] = np.ravel(summary.get(keyword))
            except KeyError:
                pass
        return out

    def well_history(self, well: str) -> dict[str, np.ndarray]:
        if well not in self.well_names:
            raise KeyError(well)
        summary = self.outputs.opm.summary
        out = {"time": summary.time.copy()}
        for keyword in ("WOPR", "WWPR", "WGPR", "WWIR", "WGIR", "WBHP", "WTHP"):
            try:
                out[keyword] = np.ravel(summary.get(keyword, well))
            except KeyError:
                pass
        return out

    def well_definition(self, well: str) -> dict[str, list[list[str]]]:
        """All geometry and control records belonging to one named well."""
        if well not in self.well_names:
            raise KeyError(well)
        keys = ("WELSPECS", "COMPDAT", "WCONPROD", "WCONINJE", "WCONHIST",
                "WCONINJH", "WELOPEN", "WELTARG", "WPIMULT")
        return {
            key: [record for record in self.deck.schedule.records.get(key, [])
                  if record and record[0] == well]
            for key in keys
        }

    def initial_blackoil_state(self) -> State:
        """Convert report zero to a PyMRST three-phase black-oil state."""
        from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
            blackoil_status,
        )

        raw = self.initial_state.raw
        sw = np.asarray(raw["SWAT"], float)
        sg = np.asarray(raw.get("SGAS", np.zeros_like(sw)), float)
        so = np.maximum(1.0-sw-sg, 0.0)
        nw = len(self.well_names)
        return State(
            pressure=self.initial_state.pressure.copy(),
            sW=sw, sO=so, sG=sg,
            rs=np.asarray(raw.get("RS", np.zeros_like(sw)), float),
            rv=np.asarray(raw.get("RV", np.zeros_like(sw)), float),
            status=blackoil_status(sw, so, sg, disgas=True, vapoil=True),
            qWs=np.zeros(nw), qOs=np.zeros(nw), qGs=np.zeros(nw),
            bhp=np.zeros(nw),
        )

    def regression_signature(self) -> dict:
        """Small O(1) field regression spanning setup, states, and history.

        This is the CI tier: report 0, a middle report, and the final report,
        plus first/middle/final field rates. Full cellwise trajectories remain
        available through ``outputs`` for release validation without making
        every test read hundreds of megabytes.
        """
        state_rows = []
        for index in (0, len(self.outputs.opm.states)//2,
                      len(self.outputs.opm.states)-1):
            state = self.outputs.opm.states[index]
            state_rows.append([
                state.report_step, np.mean(state.pressure), np.std(state.pressure),
                np.mean(state.raw["SWAT"]), np.mean(state.raw["SGAS"]),
            ])
        history = self.field_history()
        columns = (0, len(history["time"])//2, len(history["time"])-1)
        return {
            "dimensions": np.array([
                self.active_cells, len(self.components), len(self.well_names),
                len(self.deck.schedule.dates), len(self.outputs.opm.states),
            ], dtype=int),
            "states": np.asarray(state_rows),
            "time": history["time"][list(columns)],
            "rates": np.vstack([
                history[key][list(columns)]
                for key in ("FOPR", "FWPR", "FGPR", "FWIR", "FGIR")
            ]),
        }

    def build_native_schedule(self):
        """Build the unified active grid, native rock, wells and controls."""
        if self.native_schedule is not None:
            return self.native_grid, self.native_rock, self.native_schedule
        from pymrst.gridprocessing import compute_geometry, process_grdecl

        result_grid = self.outputs.opm.grid
        logical_count = int(np.prod(result_grid.cart_dims))
        actnum = np.zeros(logical_count, dtype=int)
        actnum[result_grid.active] = 1
        grid = process_grdecl({
            "cartDims": result_grid.cart_dims,
            "COORD": result_grid.coord,
            "ZCORN": result_grid.zcorn,
            "ACTNUM": actnum,
        }, split_disconnected=False)
        grid = compute_geometry(grid)

        prefix = self.outputs.opm.prefix
        fields = _collect(prefix.with_suffix(".INIT"))
        compressed = np.full(logical_count, -1, dtype=int)
        compressed[result_grid.active] = np.arange(result_grid.active.size)
        positions = compressed[np.asarray(grid.cells.indexMap, dtype=int)]
        perm = np.column_stack([
            fields[key][-1] for key in ("PERMX", "PERMY", "PERMZ")
        ])*(MILLI*DARCY)
        rock = Rock(
            perm[positions], np.asarray(fields["PORO"][-1])[positions],
            np.asarray(fields["NTG"][-1])[positions],
        )
        # Schedule conversion needs phase names and grid/rock only; fluid PVT
        # construction is deliberately a separate native-simulation stage.
        owner = SimpleNamespace(
            G=grid, rock=rock, gravity_magnitude=NORM_GRAVITY,
            fluid=SimpleNamespace(phases=("water", "oil", "gas")),
        )
        schedule = convert_deck_schedule_to_mrst_ad(
            owner, self.deck.schedule_deck(), ensure_consistent=True
        )
        self.native_grid, self.native_rock, self.native_schedule = grid, rock, schedule
        return grid, rock, schedule

    def build_native_vfp_tables(self):
        """Return sparse, one-based injector and producer VFP table lists."""
        from pymrst.modules.ad_core.models.facilities.vfp import VFPTable

        injector, producer = self.deck.vfp_tables()

        def convert(tables):
            out = [None]*max(tables, default=0)
            for number, table in tables.items():
                out[number - 1] = VFPTable.from_deck(table)
            return out

        return convert(injector), convert(producer)

    def build_reference_operators(self):
        """Build TPFA operators using OPM's fault-processed INIT connections.

        ``TRANX/Y/Z`` already include the deck's faults, region multipliers and
        inactive-cell processing. ``TRANNNC`` and EGRID's ``NNC1/NNC2`` add
        the non-neighbour connections. Reusing these values avoids applying a
        second, geometrically different fault interpretation in Python.
        """
        from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa

        grid, rock, _ = self.build_native_schedule()
        init = _collect(self.outputs.opm.prefix.with_suffix(".INIT"))
        egrid = _collect(self.outputs.opm.prefix.with_suffix(".EGRID"))
        logical_count = int(np.prod(grid.cartDims))
        nnc_logical = np.column_stack([
            np.asarray(egrid["NNC1"][-1], dtype=np.int64) - 1,
            np.asarray(egrid["NNC2"][-1], dtype=np.int64) - 1,
        ])

        # Eclipse METRIC transmissibility unit: cP*m^3/(day*bar).
        trans_unit = 1e-3/(86400.0*1e5)
        nnc_trans = np.asarray(init["TRANNNC"][-1], dtype=float)*trans_unit
        # All interior values are replaced below. Unit half-face values avoid
        # an unnecessary field-scale geometric transmissibility calculation.
        operators = setup_operators_tpfa(
            grid, rock, t_half=np.ones(np.asarray(grid.cells.faces).size)
        )

        active_logical = np.asarray(self.outputs.opm.grid.active, dtype=np.int64)
        position = np.full(logical_count, -1, dtype=np.int64)
        position[active_logical] = np.arange(active_logical.size)
        internal = np.all(grid.faces.neighbors >= 0, axis=1)
        neighbors = np.asarray(grid.faces.neighbors[internal], dtype=np.int64)
        logical = np.asarray(grid.cells.indexMap, dtype=np.int64)[neighbors]
        low = np.minimum(logical[:, 0], logical[:, 1])
        delta = np.abs(logical[:, 1] - logical[:, 0])
        nx, ny, _ = (int(value) for value in grid.cartDims)
        face_trans = np.zeros(neighbors.shape[0], dtype=float)
        ordinary = np.zeros(neighbors.shape[0], dtype=bool)
        for spacing, keyword in ((1, "TRANX"), (nx, "TRANY"), (nx*ny, "TRANZ")):
            mask = delta == spacing
            face_trans[mask] = np.asarray(init[keyword][-1], dtype=float)[position[low[mask]]]
            ordinary |= mask
        # ``process_grdecl`` represents geometrically matched fault/pinch NNCs
        # as ordinary faces. Match those faces to EGRID by their logical cells.
        nnc_lookup = {
            tuple(sorted((int(pair[0]), int(pair[1])))): value
            for pair, value in zip(nnc_logical, nnc_trans)
        }
        pairs = np.sort(logical[~ordinary], axis=1)
        face_trans[~ordinary] = np.asarray([
            nnc_lookup.get((int(pair[0]), int(pair[1])), 0.0) for pair in pairs
        ])
        face_trans[ordinary] *= trans_unit
        # There is one additional geometric match not present in EGRID. It is
        # deliberately retained as a zero-transmissibility face.
        if sum(tuple(map(int, pair)) not in nnc_lookup for pair in pairs) != 1:
            raise ValueError("Norne geometric/EGRID connection mismatch")
        operators.T[:] = face_trans
        operators.t_all[internal] = face_trans
        return operators

    def build_native_fluid(self):
        """Construct a two-region live-oil/wet-gas fluid from the Norne deck."""
        from pymrst.modules.ad_props.props.init_deck_adi_fluid import (
            BlackOilFluid, init_deck_adi_fluid,
        )
        from pymrst.modules.deckformat.deckinput.convert_deck_units import (
            convert_deck_units,
        )
        from pymrst.modules.deckformat.deckinput.read_eclipse_deck import (
            read_eclipse_deck,
        )

        grid, rock, _ = self.build_native_schedule()
        parsed = convert_deck_units(read_eclipse_deck(self.deck.data_file))
        fields = _collect(self.outputs.opm.prefix.with_suffix(".INIT"))
        active = np.asarray(self.outputs.opm.grid.active, dtype=np.int64)
        logical_count = int(np.prod(grid.cartDims))
        parsed["REGIONS"] = {}
        for name in ("PVTNUM", "SATNUM"):
            values = np.ones(logical_count, dtype=np.int64)
            values[active] = np.asarray(fields[name][-1], dtype=np.int64)
            parsed["REGIONS"][name] = values
        assigned = init_deck_adi_fluid(
            parsed, grid=grid, ignore_errors=False,
        )
        logical = np.asarray(grid.cells.indexMap, dtype=np.int64)
        pvt = parsed["REGIONS"]["PVTNUM"][logical] - 1
        sat = parsed["REGIONS"]["SATNUM"][logical] - 1

        def regional(functions, numbers):
            functions = list(functions) if isinstance(functions, list) else [functions]
            if len(functions) == 1:
                return functions[0]
            masks = [numbers == region for region in range(len(functions))]

            def evaluate(*args):
                out = None
                for mask, function in zip(masks, functions):
                    candidate = function(*args)
                    term = mask*candidate
                    out = term if out is None else out + term
                return out
            return evaluate

        # The generic facility interface currently requires one surface
        # density per phase. Norne's two records differ by at most 0.12%; use
        # their mean while keeping every pressure-dependent property regional.
        rho = np.asarray([
            np.mean(np.asarray(assigned[name], dtype=float))
            for name in ("rhoWS", "rhoOS", "rhoGS")
        ])
        swof = parsed["PROPS"]["SWOF"]
        swcon = np.asarray([table[0, 0] for table in swof], dtype=float)[sat]
        fluid = BlackOilFluid(
            rhoS=rho,
            b_w=regional(assigned["bW"], pvt),
            mu_w=regional(assigned["muW"], pvt),
            b_o=regional(assigned["bO"], pvt),
            mu_o=regional(assigned["muO"], pvt),
            rs_sat=regional(assigned["rsSat"], pvt),
            b_g=regional(assigned["bG"], pvt),
            mu_g=regional(assigned["muG"], pvt),
            kr_w=regional(assigned["krW"], sat),
            kr_g=regional(assigned["krG"], sat),
            kr_ow=regional(assigned["krOW"], sat),
            kr_og=regional(assigned["krOG"], sat),
            pv_mult=regional(assigned["pvMultR"], pvt),
            swcon=swcon,
            pc_ow=(regional(assigned["pcOW"], sat) if "pcOW" in assigned else None),
            rv_sat=regional(assigned["rvSat"], pvt),
            vapoil=True,
        )
        self.native_fluid = fluid
        return fluid

    def build_native_model(self):
        """Assemble the native three-phase Norne model and its initial state."""
        from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
            GenericBlackOilModelAD,
        )

        grid, rock, schedule = self.build_native_schedule()
        fluid = self.build_native_fluid()
        operators = self.build_reference_operators()
        model = GenericBlackOilModelAD(
            grid, rock, fluid, schedule.controls[0].W,
            disgas=True, vapoil=True, gravity=NORM_GRAVITY,
            operators=operators,
            # MRST's field-model defaults.  The GenericBlackOilModelAD defaults
            # are deliberately stringent for small analytic unit tests and
            # cannot be met by a 44k-cell history case at double precision.
            tol_cnv=1e-3, tol_mb=1e-7, tol_well=1e-4,
        )
        injector, producer = self.build_native_vfp_tables()
        model.facility.vfp_tables_injector = injector
        model.facility.vfp_tables_producer = producer
        model.facility._bind_vfp_tables()
        self.native_model = model
        state = self.initial_blackoil_state()
        rate_names = ("qWs", "qOs", "qGs")
        for index, well in enumerate(model.wells):
            cells = np.asarray(well.cells, dtype=np.int64)
            if cells.size:
                state["bhp"][index] = float(np.mean(state["pressure"][cells]))
            if well.status and well.type == "resv_history":
                for phase, name in enumerate(rate_names):
                    state[name][index] = well.val*well.compi[phase]
        return model, state, schedule

    @staticmethod
    def save_native_checkpoint(path, state, completed_periods: int) -> Path:
        """Persist a converged report-boundary state for trajectory restart."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        keys = (
            "pressure", "sW", "sO", "sG", "rs", "rv", "status",
            "qWs", "qOs", "qGs", "bhp",
        )
        payload = {
            key: np.asarray(state[key])
            for key in keys if key in state
        }
        payload["completed_periods"] = np.asarray(int(completed_periods))
        np.savez_compressed(target, **payload)
        return target

    def load_native_checkpoint(self, path) -> tuple[State, int]:
        """Load and validate a report-boundary native Norne checkpoint."""
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "pressure", "sW", "sO", "sG", "rs", "rv", "status",
                "qWs", "qOs", "qGs", "bhp", "completed_periods",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    f"Norne checkpoint is missing {sorted(missing)}"
                )
            state = State(**{
                key: np.asarray(archive[key]).copy()
                for key in required if key != "completed_periods"
            })
            completed = int(np.asarray(archive["completed_periods"]).item())
        if state["pressure"].size != self.active_cells:
            raise ValueError(
                "Norne checkpoint cell count does not match this case"
            )
        if state["bhp"].size != len(self.well_names):
            raise ValueError(
                "Norne checkpoint well count does not match this case"
            )
        if completed < 0 or completed > len(self.deck.schedule.dates):
            raise ValueError("invalid completed_periods in Norne checkpoint")
        return state, completed

    def run_native(
        self,
        max_steps: int | None = None,
        solver=None,
        resume_from=None,
        save_checkpoint_to=None,
    ):
        """Run the native PyMRST Norne model.

        ``max_steps`` provides a practical smoke/regression tier (for example,
        one or five report steps); ``None`` requests the complete 247-step
        field history.  The returned model is included so callers can inspect
        controls, operators and facility state alongside the simulation.
        """
        from pymrst.modules.ad_core.simulators.nonlinear_solver import (
            NonLinearSolver,
        )
        from pymrst.modules.ad_core.solvers.linear_solvers import CPRSolver
        from pymrst.modules.ad_core.timesteps import (
            IterationCountTimeStepSelectorAD as IterationCountTimeStepSelector,
        )
        from pymrst.modules.ad_core.utils.schedule import subset_schedule

        model, state, schedule = self.build_native_model()
        first_period = 0
        if resume_from is not None:
            state, first_period = self.load_native_checkpoint(resume_from)
            if first_period >= schedule.n_steps:
                raise ValueError("checkpoint already completes the schedule")
        available = schedule.n_steps-first_period
        if max_steps is not None:
            count = int(max_steps)
            if count < 1 or count > available:
                raise ValueError(
                    f"max_steps must be between 1 and {available}"
                )
        else:
            count = available
        schedule = subset_schedule(
            schedule, np.arange(first_period, first_period+count)
        )
        if solver is None:
            linear = CPRSolver(
                local_stage="block_jacobi",
                fallback_to_ilu=False,
                maxiter=80,
                restart=30,
                rtol=1e-7,
            )
            solver = NonLinearSolver(
                linear_solver=linear,
                max_iterations=25,
                use_relaxation=True,
                use_line_search=True,
                timestep_selector=IterationCountTimeStepSelector(
                    target_iteration_count=6,
                    first_rampup_step=86400.0,
                    max_timestep=30.0*86400.0,
                    reset_on_controls_changed=True,
                ),
            )
        states, report = solver.simulate_schedule(model, state, schedule)
        completed_now = sum(step.converged for step in report.steps)
        checkpoint = None
        if save_checkpoint_to is not None and completed_now:
            checkpoint = self.save_native_checkpoint(
                save_checkpoint_to,
                states[completed_now-1],
                first_period+completed_now,
            )
        return {
            "model": model,
            "initial_state": state,
            "schedule": schedule,
            "states": states,
            "report": report,
            "first_period": first_period,
            "completed_periods": first_period+completed_now,
            "checkpoint": checkpoint,
        }


def _components(outputs: NorneReferenceOutputs) -> list[NorneGridComponent]:
    prefix = outputs.opm.prefix
    fields = _collect(prefix.with_suffix(".INIT"))
    grids = outputs.opm.grid.to_pymrst_grid(compute_geometry=False)
    if not isinstance(grids, list):
        grids = [grids]
    active = outputs.opm.grid.active
    logical_count = int(np.prod(outputs.opm.grid.cart_dims))
    compressed = np.full(logical_count, -1, dtype=int)
    compressed[active] = np.arange(active.size)
    perm = np.column_stack([fields[key][-1] for key in ("PERMX", "PERMY", "PERMZ")])
    perm = perm*(MILLI*DARCY)
    poro = np.asarray(fields["PORO"][-1], float)
    ntg = np.asarray(fields["NTG"][-1], float)
    region_names = ("SATNUM", "PVTNUM", "EQLNUM", "FIPNUM", "FLUXNUM", "IMBNUM")
    components = []
    for grid in grids:
        positions = compressed[np.asarray(grid.cells.indexMap, dtype=int)]
        if np.any(positions < 0):
            raise ValueError("EGRID component contains a cell absent from INIT")
        rock = Rock(perm[positions], poro[positions], ntg[positions])
        regions = {
            name: np.asarray(fields[name][-1], dtype=int)[positions]
            for name in region_names if name in fields
        }
        components.append(NorneGridComponent(grid, rock, regions))
    return components


@dataclass
class NorneFieldModel:
    """MRST's `[state0, model, schedule, nls]`, plus what built them."""

    deck: dict
    G_viz: object
    G_sim: object
    rock: object
    rock_ecl: object
    N: np.ndarray
    T: np.ndarray
    model: object
    state0: object
    schedule: object
    nonlinear: object
    states: list | None = None
    report: object | None = None


def norne_output_prefix_path(root=None) -> Path:
    """`output.opm.location` - the EGRID/INIT prefix MRST reads."""
    from pymrst.modules.ad_blackoil.examples.utils.get_norne_field_deck_opm_data import (
        norne_output_prefix,
    )
    return Path(norne_output_prefix(root, simulator="eclipse"))


def norne_deck_path(root=None) -> Path:
    """`NORNE_ATW2013.DATA` - the deck itself, for the general reader."""
    base = (Path(root) if root is not None else
            Path(__file__).resolve().parents[4].parent
            / "mrst-2024a-Geomech" / "examples" / "data" / "norne_field_model")
    for candidate in (base/"opm-data"/"norne"/"NORNE_ATW2013.DATA",
                      base/"NORNE_ATW2013.DATA"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"NORNE_ATW2013.DATA was not found under {base}")


def build_norne_field_model(root=None, *, nsteps: int | None = None):
    """MRST's chain: the deck, ECLIPSE's grid, and a model on both."""
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
    from pymrst.modules.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad_ad,
    )
    from pymrst.modules.ad_core.timesteps.timestep_selectors import (
        FactorTimeStepSelectorAD as FactorTimeStepSelector,
    )
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.modules.ad_props.props.deck_fluid import deck_fluid_object
    from pymrst.modules.ad_props.props.init_deck_adi_fluid_dispatch import (
        init_deck_adi_fluid,
    )
    from pymrst.modules.deckformat.deckinput.convert_deck_units import (
        convert_deck_units,
    )
    from pymrst.modules.deckformat.deckinput.read_eclipse_deck import (
        read_eclipse_deck,
    )
    from pymrst.modules.deckformat.params.rock.compress_rock import compress_rock
    from pymrst.modules.deckformat.params.rock.init_eclipse_rock import (
        init_eclipse_rock,
    )
    from pymrst.modules.deckformat.resultinput import (
        init_grid_from_eclipse_output, read_eclipse_output_file_un_fmt,
    )

    deck = convert_deck_units(read_eclipse_deck(norne_deck_path(root)))

    prefix = norne_output_prefix_path(root)
    egrid = read_eclipse_output_file_un_fmt(prefix.with_suffix(".EGRID"))
    init = read_eclipse_output_file_un_fmt(prefix.with_suffix(".INIT"))
    (G_viz, G_sim), rock_ecl, N, T = init_grid_from_eclipse_output(
        init, egrid, output_sim_grid=True)

    # Rock from the *deck*, connections from the *output*: MRST takes each from
    # where it is authoritative. `compress_rock` maps the deck's Cartesian
    # arrays onto the active cells ECLIPSE kept.
    rock = compress_rock(init_eclipse_rock(deck), G_sim.cells.indexMap)
    fluid = deck_fluid_object(init_deck_adi_fluid(deck))

    operators = setup_operators_tpfa(G_sim, rock, neighbors=N, trans=T)
    # MRST's own settings for this case, and they are not incidental - see the
    # module docstring. PyMRST's default `tol_cnv = 1e-8` is six orders
    # tighter than the 1e-2 the script asks for, and a field model never
    # reaches it: the step converges and is then declared failed.
    model = GenericBlackOilModelAD(G_sim, rock, fluid, [], disgas=True,
                                 vapoil=True, gravity=NORM_GRAVITY,
                                 operators=operators,
                                 tol_cnv=1e-2, tol_mb=1e-7, tol_well=5e-3)
    model.input_data = deck
    # MRST's script, verbatim:
    #   model.FacilityModel.FacilityFlowDiscretization
    #        .PhaseFlux.allowCrossFlow = false;
    #
    # Norne has wells the schedule holds at a zero rate. With crossflow
    # allowed, such a well can satisfy `q_t = 0` by circulating - some
    # perforations taking fluid in while others put it out - which is a
    # near-indeterminate family of solutions rather than a point, and the well
    # block comes out ill-conditioned. Disabling it is not a tuning knob here;
    # it is what the reference script does for this case.
    model.facility.allow_cross_flow = False

    state0, model, schedule, nonlinear = init_eclipse_problem_ad_ad(
        deck, model=model, timestep_strategy="none", max_iterations=12)
    # MRST: `nls.timeStepSelector = FactorTimeStepSelector()`.
    #
    # Norne's schedule opens with a few short steps and then jumps to month-
    # long ones. Handed a month directly, Newton fails and the solver halves
    # blindly - six cuts, sixty-four ministeps, and the step is abandoned
    # having done more work than the whole report step needed. The selector
    # instead *chooses* each ministep: it grows a successful one by a factor
    # and shrinks on failure, so a hard report step is crossed in as many
    # ministeps as it actually needs rather than in a fixed binary ladder.
    nonlinear.timestep_selector = FactorTimeStepSelector()
    if nsteps is not None and schedule is not None:
        from pymrst.modules.ad_core.utils.schedule import Schedule
        count = min(int(nsteps), int(schedule.n_steps))
        schedule = Schedule(schedule.step_val[:count],
                            schedule.step_control[:count], schedule.controls)

    return NorneFieldModel(deck=deck, G_viz=G_viz, G_sim=G_sim, rock=rock,
                           rock_ecl=rock_ecl, N=N, T=T, model=model,
                           state0=state0, schedule=schedule,
                           nonlinear=nonlinear)


def run_norne_field_model(root=None, *, nsteps: int | None = None,
                          simulate: bool = False, verbose: bool = True):
    """Build MRST's field model; simulate it when asked."""
    case = build_norne_field_model(root, nsteps=nsteps)
    if not simulate:
        return case
    case.nonlinear.verbose = bool(verbose)
    case.states, case.report = case.nonlinear.simulate_schedule(
        case.model, case.state0, case.schedule)
    return case


def setup_field_model_norne(root: str | Path | None = None) -> NorneFieldCase:
    """Initialize full Norne geometry, rock, regions, state and references."""
    deck, outputs = get_norne_field_deck_opm_data_ad(root)
    if not outputs.opm.states:
        raise ValueError("Norne archive contains no restart states")
    return NorneFieldCase(deck, outputs, _components(outputs), outputs.opm.states[0])


def run_field_model_norne_example(root: str | Path | None = None, *,
                                  nsteps: int | None = None,
                                  simulate: bool = False,
                                  verbose: bool = True):
    """MRST's script: build the field model from ECLIPSE's output, and run it.

    This is `run_norne_field_model`; the name is MRST's. The older loader that
    only reported the archive's dimensions is `setup_field_model_norne`, which
    the regression tiers still use.
    """
    return run_norne_field_model(root, nsteps=nsteps, simulate=simulate,
                                 verbose=verbose)


def report_norne_archive_dimensions(root: str | Path | None = None):
    """Load the full field case and report its regression-ready dimensions."""
    case = setup_field_model_norne(root)
    print(
        f"Norne: {case.active_cells} active cells, {len(case.components)} components, "
        f"{len(case.well_names)} wells, {len(case.deck.schedule.dates)} dates, "
        f"{len(case.outputs.opm.states)} restart states"
    )
    return case


def field_model_norne_example(plot: bool = True, nsteps: int | None = None) -> SimpleNamespace:
    """MRST `fieldModelNorneExample`."""
    from pymrst.modules.ad_core.simulators.sim_runner.get_packed_simulator_output import (
        get_packed_simulator_output,
    )
    from pymrst.modules.ad_core.simulators.sim_runner.pack_simulation_problem import (
        pack_simulation_problem,
    )
    from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem import (
        simulate_packed_problem,
    )

    out = setup_field_model_norne_example(nsteps)
    # Pack and simulate
    problem = pack_simulation_problem(out.state0, out.model, out.schedule, "norne",
                                      Name="GenericBlackOil_EGRID" + (
                                          "" if nsteps is None else f"_first{nsteps}"),
                                      NonLinearSolver=out.nls)
    simulate_packed_problem(problem, continueOnError=False)
    out.problem = problem
    out.wellSols, out.states, out.reports, _ = get_packed_simulator_output(problem)
    if plot:
        _plot_norne(out, out.pth/"opm-tests", out.pth)
    return out


fieldModelNorneExample = field_model_norne_example


def setup_field_model_norne_example(nsteps: int | None = None) -> SimpleNamespace:
    """Everything `field_model_norne_example` does before it simulates: the
    deck and references, the grid and model, and the script's settings."""
    from dataclasses import replace

    from pymrst.gridprocessing import compute_geometry
    from pymrst.modules.ad_blackoil.examples.utils.get_norne_field_deck_opm_data import (
        get_norne_field_deck_opm_data,
    )
    from pymrst.modules.ad_blackoil.examples.utils.local_dataset_path import (
        local_dataset_path,
    )
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_core.ad_oo.generic_facility_model import GenericFacilityModel
    from pymrst.modules.ad_core.ad_oo.statefunctions.facility.well_component_total_volume_balance_crossflow import (
        WellComponentTotalVolumeBalanceCrossflow,
    )
    from pymrst.modules.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad,
    )
    from pymrst.modules.ad_core.timesteps.factor_time_step_selector import FactorTimeStepSelector
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.modules.ad_core.utils.schedule import Schedule
    from pymrst.modules.ad_props.props.init_deck_adi_fluid_dispatch import (
        init_deck_adi_fluid,
    )
    from pymrst.modules.deckformat.params.rock.compress_rock import compress_rock
    from pymrst.modules.deckformat.params.rock.init_eclipse_rock import init_eclipse_rock
    from pymrst.modules.deckformat.resultinput import (
        init_grid_from_eclipse_output,
        read_eclipse_output_file_un_fmt,
    )
    from pymrst.params.gravity.gravity import gravity

    gravity("reset", "on")
    # MRST's useMex selects compiled table interpolation and AD kernels; the
    # values are the same, and PyMRST's diagonal backend has its own kernels.
    useMex = True
    pth = local_dataset_path("norne_field_model", "opm-data/norne/NORNE_ATW2013.DATA")
    # Get the deck and the reference results
    deck, output = get_norne_field_deck_opm_data()
    egrid = read_eclipse_output_file_un_fmt(output.opm.location + ".EGRID")
    init = read_eclipse_output_file_un_fmt(output.opm.location + ".INIT")
    (Ge1, G_sim), rock_ecl, Ne, Te = init_grid_from_eclipse_output(init, egrid,
                                                                   output_sim_grid=True)
    G_viz = compute_geometry(Ge1)
    rock = init_eclipse_rock(deck)
    rock = compress_rock(rock, G_sim.cells.indexMap)
    # Set up the model on the simulation grid, with ECLIPSE's transmissibilities
    fluid = init_deck_adi_fluid(deck)
    model = GenericBlackOilModel(G_sim, None, fluid, disgas=True, vapoil=True, inputdata=deck)
    model.rock = rock
    model.operators = setup_operators_tpfa(G_sim, rock, neighbors=Ne, trans=Te)
    state0, model, schedule, nls = init_eclipse_problem_ad(
        deck, model=model, TimestepStrategy="ds", useCPR=True, useMex=useMex)
    nls.LinearSolver.tolerance = 1e-3
    # The model's own settings for this case
    nls.maxIterations = 12
    nls.timeStepSelector = FactorTimeStepSelector()
    model.toleranceCNV = 1e-2
    model.toleranceMB = 1e-7
    model.FacilityModel = GenericFacilityModel(model)
    model.FacilityModel.toleranceWellBHP = 1e-3
    model.FacilityModel.toleranceWellRate = 5e-3
    model.FlowDiscretization = None
    model.FlowPropertyFunctions = None
    model.PVTPropertyFunctions = None
    model.AutoDiffBackend = replace(model.AutoDiffBackend, use_mex=useMex, row_major=True)
    model = model.validate_model()
    xflow = WellComponentTotalVolumeBalanceCrossflow(model)
    xflow.onlyLocalDerivatives = True
    ffd = model.FacilityModel.FacilityFlowDiscretization
    ffd.get_state_function("PhaseFlux").allowCrossFlow = False
    ffd.set_state_function("ComponentTotalFlux", xflow)
    useFlag = False
    pvt = model.PVTPropertyFunctions
    pvt.get_state_function("Viscosity").useSaturatedFlag = useFlag
    pvt.get_state_function("ShrinkageFactors").useSaturatedFlag = useFlag
    RvMax = pvt.get_state_function("RvMax")
    RvMax.rvReduction = 0.5   # Not tested, but in the deck
    pvt.set_state_function("RvMax", RvMax)
    model.dsMaxAbs = 0.2
    model.dpMaxRel = 0.1
    nls.useRelaxation = True
    nls.minRelaxation = 0.5
    nls.maxIterations = 18
    nls.maxTimestepCuts = 10
    nls.oscillationThreshold = 0.5
    nls.acceptanceFactor = 1
    nls.verbose = False
    nls.LinearSolver.verbose = False
    if nsteps is not None:
        schedule = Schedule(step_val=schedule.step_val[:nsteps],
                            step_control=schedule.step_control[:nsteps],
                            controls=schedule.controls)
    return SimpleNamespace(deck=deck, output=output, G_viz=G_viz, G_sim=G_sim, model=model,
                           state0=state0, schedule=schedule, nls=nls, pth=pth)


def _plot_norne(out: SimpleNamespace, opm_tests: Path, pth: Path) -> None:
    """MRST's figures: the well curves against OPM Flow (with and without
    hysteresis) and ECLIPSE, and the states beside OPM's."""
    import matplotlib.pyplot as plt

    from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
    from pymrst.modules.deckformat.resultinput.convert_restart_to_states import (
        convert_restart_to_states,
    )
    from pymrst.modules.deckformat.resultinput.convert_summary_to_well_sols import (
        convert_summary_to_well_sols,
    )
    from pymrst.plotting.extensions.plot_toolbar import plot_toolbar

    ws, schedule = out.wellSols, out.schedule
    nstep = len(ws)
    T = np.cumsum(schedule.step_val[:nstep])
    ws_opm = out.output.opm.wellSols[1:]
    T_opm = np.cumsum(schedule.step_val)
    well_sols, time = [ws, ws_opm], [T, T_opm]
    names = ["MRST (hysteresis)", "Flow legacy (hysteresis)"]
    try:
        pp = opm_tests/"norne"/"NORNE_ATW2013"
        convert_restart_to_states(pp, out.model.G)
        ws_new, T_new = convert_summary_to_well_sols(pp)
        well_sols.append(ws_new)
        time.append(T_new)
        names.append("Flow (Hysteresis)")
    except Exception:                                       # noqa: BLE001 - MRST's catch
        pass
    ecl = opm_tests/"norne"/"ECL.2014.2"/"NORNE_ATW2013"
    ws_ecl, T_ecl = convert_summary_to_well_sols(ecl, "metric")
    well_sols.append(ws_ecl)
    time.append(T_ecl)
    names.append("Eclipse (Hysteresis)")
    smry = load_summary_mat(pth/"opm_smry_nohyst.mat")
    ws_nohyst, T_nohyst = convert_summary_to_well_sols(smry, "metric")
    well_sols.append(ws_nohyst)
    time.append(T_nohyst)
    names.append("Flow (No hystersis)")
    plot_well_sols(well_sols, time, datasetnames=names, linestyles=["-o", "--"])
    plt.figure()
    plot_toolbar(out.G_viz, out.output.opm.states, title="OPM")
    plt.figure()
    plot_toolbar(out.G_viz, out.states, title="MRST")
    plt.show()


def load_summary_mat(fn: Path) -> SimpleNamespace:
    """MATLAB `smry = load(fn); smry = smry.smry` for a summary saved from
    `readEclipseSummaryUnFmt`: its arrays, with the accessors (function
    handles in the file) attached again and its indices made 0-based."""
    from scipy.io import loadmat

    from pymrst.modules.deckformat.resultinput.read_eclipse_summary_un_fmt import (
        _add_smry_funcs,
    )
    s = loadmat(str(fn), squeeze_me=True, struct_as_record=False)["smry"]
    strings = lambda x: [str(v).strip() for v in np.atleast_1d(x)]
    smry = SimpleNamespace(
        WGNAMES=strings(s.WGNAMES), KEYWORDS=strings(s.KEYWORDS), UNITS=strings(s.UNITS),
        STARTDAT=np.asarray(s.STARTDAT), nInx=np.asarray(s.nInx, dtype=np.int64) - 1,
        nInxOrder=np.asarray(s.nInxOrder, dtype=np.int64) - 1,
        kInx=np.asarray(s.kInx, dtype=np.int64) - 1,
        kInxOrder=np.asarray(s.kInxOrder, dtype=np.int64) - 1,
        data=np.asarray(s.data, dtype=float), ministep=np.asarray(s.ministep))
    return _add_smry_funcs(smry)


if __name__ == "__main__":
    def test_the_faithful_model_carries_the_decks_physics():
        out = setup_field_model_norne_example(1)
        m = out.model
        kr = m.FlowPropertyFunctions.get_state_function("RelativePermeability")
        ppd = m.FlowDiscretization.get_state_function("PhasePotentialDifference")
        ctf = m.FacilityModel.FacilityFlowDiscretization.get_state_function("ComponentTotalFlux")
        assert type(kr).__name__ == "HystereticRelativePermeability" and kr.scalingActive
        assert type(ppd).__name__ == "PhasePotentialDifferenceThresholded"
        assert np.count_nonzero(ppd.pressureThreshold) > 0
        assert type(ctf).__name__ == "WellComponentTotalVolumeBalanceCrossflow"
        # The pore volume is ECLIPSE's PORV, not one recomputed from geometry.
        np.testing.assert_allclose(m.operators.pv, m.G.cells.PORV)
        print("ok test_the_faithful_model_carries_the_decks_physics")

    def test_the_history_matched_well_gets_a_reservoir_rate_target():
        out = setup_field_model_norne_example(1)
        model = out.model
        forces = model.get_driving_forces(out.schedule.controls[0])
        model = model.validate_model(forces)
        from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
        state = model.validate_state(copy_state(out.state0))
        model, state = model.prepare_report_step(state, state, float(out.schedule.step_val[0]),
                                                 forces)
        d1h = [wm for wm in model.FacilityModel.WellModels if wm.W.name == "D-1H"][0]
        # MRST 2024a's value for the same deck and state.
        assert d1h.W.type == "resv"
        np.testing.assert_allclose(d1h.W.val, -0.068461762599074091, rtol=1e-12)
        print("ok test_the_history_matched_well_gets_a_reservoir_rate_target")

    test_the_faithful_model_carries_the_decks_physics()
    test_the_history_matched_well_gets_a_reservoir_rate_target()
    try:
        CASE = build_norne_field_model()
    except FileNotFoundError as missing:
        print(f"skip ({missing})")
        raise SystemExit(0)

    def test_the_grid_comes_from_eclipses_output():
        """Not from reprocessing the deck. Norne is faulted, and the EGRID
        carries the geometry ECLIPSE actually built."""
        assert CASE.G_sim.cells.num == 44431, CASE.G_sim.cells.num
        assert tuple(CASE.G_viz.cartDims) == (46, 112, 22)
        print(f"ok test_the_grid_comes_from_eclipses_output "
              f"({CASE.G_sim.cells.num} cells)")

    def test_the_transmissibilities_are_the_init_files():
        """Recomputing them from the reprocessed grid gives different numbers
        across every fault - the reason the grid came from the output file."""
        assert np.allclose(CASE.model.operators.T, CASE.T)
        assert CASE.T.size == CASE.N.shape[0] == 130634, CASE.T.size
        assert np.all(CASE.T > 0.0), "a zero transmissibility is a closed face"
        print(f"ok test_the_transmissibilities_are_the_init_files "
              f"({CASE.T.size} connections)")

    def test_the_faults_conduct_through_non_neighbour_connections():
        i, j = CASE.N[:, 0], CASE.N[:, 1]
        assert np.any(np.abs(i - j) > 46*112), "no cross-fault connections"
        print("ok test_the_faults_conduct_through_non_neighbour_connections")

    def test_the_field_holds_oil():
        """It did not. The EQUIL contacts were supplied shallowest-first, so
        water was anchored at the gas-oil contact and gas at the water-oil
        contact; each flooded its range and the oil was squeezed out of all
        44431 cells."""
        sO = np.asarray(CASE.state0["sO"])
        with_oil = int((sO > 1e-6).sum())
        assert with_oil > 10000, with_oil
        assert sO.max() > 0.5, sO.max()
        print(f"ok test_the_field_holds_oil ({with_oil} cells, "
              f"sO up to {sO.max():.2f})")

    def test_no_cell_carries_an_undetermined_variable():
        """A water-only cell given status 1 carries `rs`, which reaches the
        equations only as `rs * b_o * s_o` and so multiplies zero - 21665
        structurally empty Jacobian columns and an exactly singular factor."""
        from pymrst.utils.adi import combine_equations
        from pymrst.utils.units import DAY

        CASE.model.facility.wells[:] = list(CASE.schedule.controls[0].W)
        jac = combine_equations(*CASE.model.get_equations(
            CASE.state0, CASE.state0.copy(), 1.0*DAY)).jac[0].tocsc()
        assert jac.shape[0] == jac.shape[1], jac.shape
        empty = int((np.diff(jac.indptr) == 0).sum())
        assert empty == 0, f"{empty} structurally empty columns"
        print(f"ok test_no_cell_carries_an_undetermined_variable "
              f"(square {jac.shape[0]}, no empty columns)")

    def test_it_is_a_live_oil_wet_gas_model():
        assert CASE.model.disgas is True and CASE.model.vapoil is True
        assert CASE.model.n_phases == 3
        print("ok test_it_is_a_live_oil_wet_gas_model")

    def test_the_schedule_is_the_decks_field_history():
        assert CASE.schedule.n_steps == 247, CASE.schedule.n_steps
        print(f"ok test_the_schedule_is_the_decks_field_history "
              f"({CASE.schedule.n_steps} steps, "
              f"{len(CASE.schedule.controls)} control periods)")

    def test_mrsts_tolerances_are_carried_over():
        """`toleranceCNV = 1e-2`, not PyMRST's 1e-8. Six orders tighter is not
        a safer choice here: the residuals stall on the tolerance rather than
        the physics and a converged step is declared failed and cut."""
        assert np.isclose(CASE.model.tol_cnv, 1e-2), CASE.model.tol_cnv
        assert np.isclose(CASE.model.tol_mb, 1e-7), CASE.model.tol_mb
        print("ok test_mrsts_tolerances_are_carried_over")

    def test_a_report_step_converges():
        """The whole point, and what three separate fixes were needed for."""
        from pymrst.modules.ad_core.simulators.nonlinear_solver import (
            NonLinearSolver,
        )
        CASE.model.facility.wells[:] = list(CASE.schedule.controls[0].W)
        solver = NonLinearSolver(max_iterations=12)
        solver.verbose = False
        _, report = solver.solve_ministep(
            CASE.model, CASE.state0, CASE.state0.copy(),
            float(CASE.schedule.step_val[0]))
        assert report.converged, report.residuals[-1]
        print(f"ok test_a_report_step_converges "
              f"({report.iterations} Newton iterations)")

    def test_the_loader_used_by_the_regression_tiers_still_works():
        """`setup_field_model_norne` has three callers outside this file."""
        assert setup_field_model_norne().active_cells > 0
        print("ok test_the_loader_used_by_the_regression_tiers_still_works")

    test_the_grid_comes_from_eclipses_output()
    test_the_transmissibilities_are_the_init_files()
    test_the_faults_conduct_through_non_neighbour_connections()
    test_the_field_holds_oil()
    test_no_cell_carries_an_undetermined_variable()
    test_it_is_a_live_oil_wet_gas_model()
    test_the_schedule_is_the_decks_field_history()
    test_mrsts_tolerances_are_carried_over()
    test_a_report_step_converges()
    test_the_loader_used_by_the_regression_tiers_still_works()
    print("\nAll tests passed!")


__all__ = [
    "NorneFieldCase", "NorneFieldModel", "NorneGridComponent", "field_model_norne_example",
    "load_summary_mat", "setup_field_model_norne_example",
    "build_norne_field_model", "fieldModelNorneExample", "norne_deck_path",
    "norne_output_prefix_path", "run_field_model_norne_example",
    "report_norne_archive_dimensions", "run_norne_field_model",
    "setup_field_model_norne",
]
