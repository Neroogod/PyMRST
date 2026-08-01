"""Imported full-field Norne workflow and scalable regression tiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from pymrst.modules.ad_blackoil.examples.utils.get_norne_field_deck_opm_data import (
    NorneReferenceOutputs, get_norne_field_deck_opm_data,
)
from pymrst.modules.ad_blackoil.examples.utils.load_aquifer_eclipse_result import _collect
from pymrst.modules.deckformat.norne import NorneFieldDeck
from pymrst.params.rock import Rock
from pymrst.modules.ad_core import State
from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
    convert_deck_schedule_to_mrst,
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
        from pymrst.modules.ad_blackoil.utils.get_cell_status_vo import (
            get_cell_status_vo,
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
            status=get_cell_status_vo(sw, so, sg, disgas=True, vapoil=True),
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
        schedule = convert_deck_schedule_to_mrst(
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
            GenericBlackOilModel,
        )

        grid, rock, schedule = self.build_native_schedule()
        fluid = self.build_native_fluid()
        operators = self.build_reference_operators()
        model = GenericBlackOilModel(
            grid, rock, fluid, schedule.controls[0].W,
            disgas=True, vapoil=True, gravity=NORM_GRAVITY,
            operators=operators,
            # MRST's field-model defaults.  The GenericBlackOilModel defaults
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
            IterationCountTimeStepSelector,
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


def setup_field_model_norne(root: str | Path | None = None) -> NorneFieldCase:
    """Initialize full Norne geometry, rock, regions, state and references."""
    deck, outputs = get_norne_field_deck_opm_data(root)
    if not outputs.opm.states:
        raise ValueError("Norne archive contains no restart states")
    return NorneFieldCase(deck, outputs, _components(outputs), outputs.opm.states[0])


def run_field_model_norne_example(root: str | Path | None = None):
    """Load the full field case and report its regression-ready dimensions."""
    case = setup_field_model_norne(root)
    print(
        f"Norne: {case.active_cells} active cells, {len(case.components)} components, "
        f"{len(case.well_names)} wells, {len(case.deck.schedule.dates)} dates, "
        f"{len(case.outputs.opm.states)} restart states"
    )
    return case


if __name__ == "__main__":
    run_field_model_norne_example()


__all__ = [
    "NorneFieldCase", "NorneGridComponent", "run_field_model_norne_example",
    "setup_field_model_norne",
]
