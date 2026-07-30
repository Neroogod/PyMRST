"""Boundary conditions and sources for the AD models (MRST ad-core/utils).

The AD models could only be driven by **wells**. A pressure boundary, a flux
boundary, an aquifer standing in for the rest of the field, a source term -
none of them existed. Every case in the port had to be a closed box with a well
in it, which is not how a reservoir sits in the world, and not how most of
MRST's own examples are set up.

Ported from `addFluxesFromSourcesAndBC` / `getBoundaryConditionFluxesAD` /
`getSourceFluxesAD`. Two boundary types:

- **pressure**: the boundary is held at a pressure, and fluid crosses it
  according to Darcy's law through the cell's own half-transmissibility,
  `q = mob * T * (p_bc - p_cell - rho g dz)`. Which mobility applies depends on
  which way it flows: fluid leaving carries the *cell's* mobility, fluid
  entering carries the total mobility split by the boundary's saturation - you
  cannot know a rock's relative permeability to a fluid that is not in it yet.
- **flux**: a surface-volume rate is imposed. Injected fluid arrives with the
  boundary's composition; produced fluid leaves with the cell's fractional flow.

Sources are the same logic without the transmissibility.

**Components, not phases.** MRST adds these per phase and warns that they "do
not fully support problems with vapoil or disgas active". Here the boundary
flux is converted to a *reservoir* volumetric rate and then handed to the same
component machinery the interior flux uses, so dissolved gas is carried across
a boundary the same way it is carried across a face. The upwind side supplies
the densities, exactly as in `ComponentPhaseFlux`.
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.params.wells_and_bc import BoundaryCondition, Source
from pymrst.utils.adi import ADI, value
from pymrst.modules.ad_core.utils.phase_densities_to_bfactor import (
    phase_densities_to_bfactor,
)


@dataclass
class ForceContributions:
    """Normalized output of MRST ``computeSourcesAndBoundaryConditionsAD``."""

    phaseMass: List[Any]
    phaseVolume: List[Any]
    components: List[Any]
    mapping: sp.csr_array
    sourceCells: np.ndarray


def _field(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _surface_densities(model) -> np.ndarray:
    if hasattr(model, "get_surface_densities"):
        return np.asarray(model.get_surface_densities(), dtype=float).ravel()
    if hasattr(model, "getSurfaceDensities"):
        return np.asarray(model.getSurfaceDensities(), dtype=float).ravel()
    return np.asarray(model.fluid.rho_s, dtype=float).ravel()


def _empty_contributions(nph: int, nc: int) -> ForceContributions:
    return ForceContributions(
        phaseMass=[np.zeros(0) for _ in range(nph)],
        phaseVolume=[np.zeros(0) for _ in range(nph)],
        components=[],
        mapping=sp.csr_array((0, 0)),
        sourceCells=np.zeros(0, dtype=np.int64),
    )


def _force_map(cells: np.ndarray, nc: int) -> sp.csr_array:
    base = _scatter(cells, nc)
    # This deliberately retains one row per force.  Repeated cells therefore
    # have identical rows, exactly like MRST's ``map(cells, :)``.
    return base[cells, :].tocsr()


def _contributions(force, q_s, q_r, b, rho_s, cells, dissolved, nc):
    nph = len(q_s)
    if force is None or cells.size == 0:
        return _empty_contributions(nph, nc)
    q_s = list(q_s)
    q0 = list(q_s)
    if dissolved is not None:
        total = np.zeros(cells.size)
        for q in q0:
            total += np.asarray(value(q), dtype=float)
        inj = total > 0
        override = _field(force, "dissolution")
        for i in range(nph):
            for j in range(nph):
                r_cell = dissolved[i][j]
                if r_cell is None:
                    continue
                if override is None:
                    r_inj = 0.0
                else:
                    ds = np.asarray(override)
                    r_inj = ds[:, i, j] if ds.ndim == 3 else ds[i, j]
                r = np.where(inj, r_inj, np.asarray(value(r_cell))[cells])
                q_s[i] = q_s[i] + r * q0[j]
    mass = [q_s[i] * rho_s[i] for i in range(nph)]
    return ForceContributions(mass, list(q_r), [], _force_map(cells, nc), cells)


def compute_sources_and_boundary_conditions_ad(
    model,
    pressure: Sequence[Any],
    saturation: Sequence[Any],
    mobility: Sequence[Any],
    density: Sequence[Any],
    dissolved,
    forces,
) -> Tuple[ForceContributions, ForceContributions]:
    """Compute MRST-compatible normalized source and BC contributions.

    Rates in ``phaseMass`` are phase-pseudocomponent mass rates at surface
    conditions. ``phaseVolume`` contains reservoir-condition phase volumes.
    """
    nph = len(mobility)
    nc = int(model.G.cells.num)
    rho_s = _surface_densities(model)
    b = phase_densities_to_bfactor(density, rho_s, dissolved)
    bc_force = _field(forces, "bc")
    src_force = _field(forces, "src")

    if bc_force is None or np.asarray(_field(bc_force, "face", [])).size == 0:
        bc_out = _empty_contributions(nph, nc)
    else:
        q_s, q_r, bc_cells = _boundary_fluxes_ad(
            model, pressure, saturation, mobility, density, b, bc_force
        )
        bc_out = _contributions(
            bc_force, q_s, q_r, b, rho_s, bc_cells, dissolved, nc
        )

    if src_force is None or np.asarray(_field(src_force, "cell", [])).size == 0:
        src_out = _empty_contributions(nph, nc)
    else:
        q_s, src_cells = _source_surface_fluxes(mobility, src_force)
        q_r = [q_s[i] / b[i][src_cells] for i in range(nph)]
        src_out = _contributions(
            src_force, q_s, q_r, b, rho_s, src_cells, dissolved, nc
        )
    return src_out, bc_out


computeSourcesAndBoundaryConditionsAD = compute_sources_and_boundary_conditions_ad


def _source_surface_fluxes(mobility, src):
    cells = np.asarray(_field(src, "cell"), dtype=np.int64).ravel()
    rate = np.asarray(_field(src, "rate"), dtype=float).ravel()
    sat = np.asarray(_field(src, "sat"), dtype=float)
    assert sat.shape == (cells.size, len(mobility))
    mob_c = [m[cells] for m in mobility]
    total = sum(mob_c[1:], mob_c[0])
    inj = rate > 0
    q_s = [
        rate * sat[:, i] * inj + rate * (mob_c[i] / total) * (~inj)
        for i in range(len(mobility))
    ]
    return q_s, cells


def _boundary_fluxes_ad(model, pressure, saturation, mobility, density, b, bc):
    faces = np.asarray(_field(bc, "face"), dtype=np.int64).ravel()
    kinds = np.char.lower(np.asarray(_field(bc, "type"), dtype=str))
    values = np.asarray(_field(bc, "value"), dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    sat = np.asarray(_field(bc, "sat"), dtype=float).copy()
    nbc, nph = faces.size, len(mobility)
    assert sat.shape == (nbc, nph)
    nbr = np.asarray(model.G.faces.neighbors)[faces]
    assert not np.any(np.all(nbr >= 0, axis=1)), "bc on internal boundary"
    cells = nbr.max(axis=1)
    t = np.asarray(model.operators.t_all)[faces]
    dz = (np.asarray(model.G.cells.centroids)[cells]
          - np.asarray(model.G.faces.centroids)[faces])
    if hasattr(model, "get_gravity_vector"):
        gvec = np.asarray(model.get_gravity_vector())
        gdz = -(dz @ gvec)
    else:
        gdz = -float(getattr(model, "gravity_magnitude", 0.0)) * dz[:, -1]

    no_sat = np.all(sat == 0, axis=1)
    for i in range(nph):
        sat[no_sat, i] = np.asarray(value(saturation[i][cells]))[no_sat]
    outside_mob = _field(bc, "mob")
    outside_rho = _field(bc, "rho")
    mob_in = [m[cells] for m in mobility]
    total = sum(mob_in[1:], mob_in[0])
    if outside_mob is None:
        mob_out = [total * sat[:, i] for i in range(nph)]
    else:
        om = np.asarray(outside_mob, dtype=float)
        mob_out = [om[:, i] for i in range(nph)]
    if outside_rho is None:
        rho_out = [density[i][cells] for i in range(nph)]
        b_out = [b[i][cells] for i in range(nph)]
    else:
        ore = np.asarray(outside_rho, dtype=float)
        rho_s = _surface_densities(model)
        rho_out = [ore[:, i] for i in range(nph)]
        b_out = [ore[:, i] / rho_s[i] for i in range(nph)]

    is_p = kinds == "pressure"
    is_sf = kinds == "flux"
    is_rf = ~(is_p | is_sf)
    q_s, q_r = [], []
    # MRST reservoir-flux boundaries split the imposed total by potential
    # upwinding. Iterate the small local fixed point used by that split.
    qph = [np.zeros(nbc) for _ in range(nph)]
    if np.any(is_rf):
        vt = values.sum(axis=1) if values.shape[1] > 1 else values[:, 0]
        use_out = np.tile((vt > 0)[:, None], (1, nph))
        for _ in range(4):
            mf = [np.where(use_out[:, i], value(mob_out[i]), value(mob_in[i]))
                  for i in range(nph)]
            mt = sum(mf)
            trial = []
            for i in range(nph):
                grav = sum(
                    mf[j] * gdz * (value(rho_out[i] + density[i][cells]) / 2
                                    - value(rho_out[j] + density[j][cells]) / 2)
                    for j in range(nph) if j != i
                )
                trial.append(mf[i] / mt * (vt + t * grav))
            use_out = np.column_stack([value(q) > 0 for q in trial])
            qph = trial

    for i in range(nph):
        sample = pressure[i][cells]
        qr = sample * 0.0
        qs = sample * 0.0
        bc_v = values[:, 0] if values.shape[1] == 1 else values[:, i]
        p_out = sample * 1.0
        p_out = p_out + (bc_v - value(sample)) * is_p
        rho_avg = (rho_out[i] + density[i][cells]) / 2
        dp = p_out - sample - rho_avg * gdz
        inj_p = is_p & (value(dp) > 0)
        prod_p = is_p & ~inj_p
        qr_prod = mob_in[i] * t * dp * prod_p
        qr_inj = mob_out[i] * t * dp * inj_p
        qr = qr + qr_prod + qr_inj
        qs = qs + b[i][cells] * qr_prod + b_out[i] * qr_inj
        inj_sf = is_sf & (bc_v > 0)
        prod_sf = is_sf & ~inj_sf
        qs = qs + bc_v * sat[:, i] * inj_sf
        qs = qs + bc_v * (mob_in[i] / total) * prod_sf
        qr = qr + bc_v * sat[:, i] / b_out[i] * inj_sf
        qr = qr + bc_v * (mob_in[i] / total) / b[i][cells] * prod_sf
        qs = qs + qph[i] * np.where(value(qph[i]) > 0, b_out[i], b[i][cells]) * is_rf
        qr = qr + qph[i] * is_rf
        q_s.append(qs)
        q_r.append(qr)
    return q_s, q_r, cells


def _scatter(cells: np.ndarray, n_cells: int) -> sp.csr_array:
    """Map one value per boundary face / source onto its cell (summing)."""
    n = cells.size
    return sp.csr_array((np.ones(n), (cells, np.arange(n))),
                        shape=(n_cells, n))


def _accumulate(target, contribution, cells: np.ndarray, n_cells: int):
    """target[cells] -= contribution, for plain arrays and ADI alike."""
    spread = _scatter(cells, n_cells)
    if isinstance(contribution, ADI):
        acc = ADI(np.zeros(n_cells),
                  [(spread @ j).tocsr() for j in contribution.jac])
        np.add.at(acc.val, cells, contribution.val)
        return target - acc
    out = np.zeros(n_cells)
    np.add.at(out, cells, np.asarray(contribution, dtype=float))
    return target - out


def boundary_reservoir_fluxes(model, state: State, bc: BoundaryCondition):
    """Per-phase *reservoir* volumetric influx at each boundary face.

    Positive means into the cell. Returns (q_res, cells, inflow_flag).
    """
    faces = np.asarray(bc.face, dtype=np.int64).ravel()
    nb = faces.size
    g = model.G
    nph = model.n_phases

    nbr = g.faces.neighbors[faces]
    assert not np.any(np.all(nbr >= 0, axis=1)), \
        "a boundary condition was placed on an interior face"
    cells = nbr.sum(axis=1) + 1                      # the one real neighbour

    # Half-transmissibility of the boundary face (no harmonic average: there
    # is only one cell).
    t_bc = model.operators.t_all[faces]

    # Gravity head between the cell centre and the face centre.
    dz = (g.cells.centroids[cells, 2] - g.faces.centroids[faces, 2])
    gdz = -model.gravity_magnitude * dz

    p = state["pressure"]
    mob = model.get_props(state, "Mobility")
    rho = model.get_props(state, "Density")

    p_cell = p[cells]
    mob_c = [m[cells] for m in mob]
    tot_mob = mob_c[0]
    for i in range(1, nph):
        tot_mob = tot_mob + mob_c[i]

    sat = np.atleast_2d(np.asarray(bc.sat, dtype=float))
    if sat.shape[0] == 1 and nb > 1:
        sat = np.tile(sat, (nb, 1))
    assert sat.shape == (nb, nph), (
        f"bc.sat must be ({nb}, {nph}); got {sat.shape}"
    )

    is_p = np.array([t.lower() == "pressure" for t in bc.type])
    is_f = np.array([t.lower() == "flux" for t in bc.type])
    assert np.all(is_p | is_f), "only 'pressure' and 'flux' BCs are supported"

    vals = np.asarray(bc.value, dtype=float).ravel()
    q_res: List[Any] = []
    inflow = np.zeros(nb, dtype=bool)

    # -- pressure faces: Darcy through the half-transmissibility -------------
    if np.any(is_p):
        rho_avg = [r[cells] for r in rho]            # the outside is the cell's
        dp = [vals - p_cell - value(rho_avg[i]) * gdz for i in range(nph)]
        # direction is set by the *total* potential, not phase by phase
        dp_tot = vals - value(p_cell) - value(rho_avg[0]) * gdz
        inflow |= is_p & (dp_tot > 0)

    for i in range(nph):
        q = np.zeros(nb)
        q = q * (p_cell * 0.0) if isinstance(p_cell, ADI) else q  # AD sample

        if np.any(is_p):
            rho_i = rho[i][cells]
            dpi = vals - p_cell - rho_i * gdz
            out = is_p & ~inflow
            inn = is_p & inflow
            # leaving: the cell's own mobility to this phase
            q_out = mob_c[i] * t_bc * dpi
            # entering: total mobility, split by the boundary's saturation -
            # you cannot use the cell's kr to a fluid that is not in it yet
            q_in = tot_mob * t_bc * dpi * sat[:, i]
            q = q + q_out * out.astype(float) + q_in * inn.astype(float)

        if np.any(is_f):
            # A flux BC is a *surface* rate; convert it to reservoir volume.
            b_c = model.get_props(state, "ShrinkageFactors")[i][cells]
            inj = is_f & (vals > 0)
            prod = is_f & (vals <= 0)
            inflow |= inj
            q_inj = (vals * sat[:, i]) / b_c
            frac = mob_c[i] / tot_mob
            q_prod = frac * vals / b_c
            q = q + q_inj * inj.astype(float) + q_prod * prod.astype(float)

        q_res.append(q)

    return q_res, cells, inflow


def source_reservoir_fluxes(model, state: State, src: Source):
    """Per-phase reservoir volumetric rate of each source (MRST getSourceFluxesAD)."""
    cells = np.asarray(src.cell, dtype=np.int64).ravel()
    rates = np.asarray(src.rate, dtype=float).ravel()
    n = cells.size
    nph = model.n_phases

    sat = np.atleast_2d(np.asarray(src.sat, dtype=float))
    if sat.shape[0] == 1 and n > 1:
        sat = np.tile(sat, (n, 1))

    mob = model.get_props(state, "Mobility")
    b = model.get_props(state, "ShrinkageFactors")
    mob_c = [m[cells] for m in mob]
    tot_mob = mob_c[0]
    for i in range(1, nph):
        tot_mob = tot_mob + mob_c[i]

    inj = rates > 0
    q_res = []
    for i in range(nph):
        b_c = b[i][cells]
        q_inj = (rates * sat[:, i]) / b_c
        q_prod = (mob_c[i] / tot_mob) * rates / b_c
        q_res.append(q_inj * inj.astype(float)
                     + q_prod * (~inj).astype(float))
    return q_res, cells, inj


def add_bc_and_sources(model, eqs: List[Any], state: State,
                       bc: Optional[BoundaryCondition] = None,
                       src: Optional[Source] = None) -> List[Any]:
    """Subtract boundary and source contributions from the component equations.

    The phase fluxes are converted to *component* contributions with the same
    densities the interior flux uses, so this works unchanged for dissolved gas
    - which is more than MRST's version claims for itself.
    """
    n_cells = model.G.cells.num
    rs = state["rs"] if model.fluid.disgas else None
    b = model.get_props(state, "ShrinkageFactors")

    for forces, fluxes in (
        (bc, boundary_reservoir_fluxes),
        (src, source_reservoir_fluxes),
    ):
        if forces is None or np.asarray(
                forces.face if hasattr(forces, "face") else forces.cell).size == 0:
            continue
        q_res, cells, _ = fluxes(model, state, forces)

        # density of component i in phase ph, on the upwind side. The cell's
        # values, as in MRST: a boundary does not carry its own PVT.
        for i, comp in enumerate(model.components):
            rho_i = comp.component_density(
                model, state,
                [bi[cells] for bi in b],
                None if rs is None else rs[cells],
            )
            contribution = None
            for ph in range(model.n_phases):
                if rho_i[ph] is None:
                    continue
                term = rho_i[ph] * q_res[ph]
                contribution = term if contribution is None else contribution + term
            if contribution is not None:
                # surface volumes in, mass out - see `ReservoirModel.mass_basis`
                contribution = model._basis_scale()[i] * contribution
                eqs[i] = _accumulate(eqs[i], contribution, cells, n_cells)
    return eqs


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_props.simple_adi_fluid import init_simple_adi_fluid
    from pymrst.params.rock import make_rock, pore_volume
    from pymrst.params.wells_and_bc import add_source, pside
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _model(nx=8, bc=None, src=None):
        """A model with **no wells at all** - which is the point."""
        g = compute_geometry(cart_grid([nx, 1, 1], [80.0, 10.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA,
        )
        m = GenericBlackOilModel(g, rock, fluid, [], bc=bc, src=src)
        st = State(pressure=np.full(g.cells.num, 200.0 * BARSA),
                   sW=np.full(g.cells.num, 0.2),
                   qWs=np.zeros(0), qOs=np.zeros(0), bhp=np.zeros(0))
        return g, m, st

    def _run(m, st, dt, n):
        solver = NonLinearSolver(max_iterations=40)
        for i in range(n):
            st, rep = solver.solve_ministep(m, st, st.copy(), dt)
            assert rep.converged, f"step {i}: {rep.residuals[-1]}"
        return st

    def test_no_forces_is_a_no_op():
        """Nothing applied must change nothing - the inert case, checked
        exactly, because a boundary term that fires when there is no boundary
        would corrupt every model in the port."""
        g, m, st = _model()
        from pymrst.utils.adi import init_variables_adi
        p, sw = init_variables_adi(st["pressure"], st["sW"])
        cur = State(pressure=p, sW=sw)
        a = m.conservation_equations(st, cur, 10 * DAY)
        b = add_bc_and_sources(m, list(a), cur, None, None)
        for x, y in zip(a, b):
            assert np.abs(value(x) - value(y)).max() == 0.0
        print("ok test_no_forces_is_a_no_op")

    def test_a_sealed_box_does_nothing():
        """No wells, no BCs, no sources: nothing may move at all."""
        g, m, st = _model()
        st1 = _run(m, st, 30 * DAY, 1)
        assert np.abs(st1["pressure"] - st["pressure"]).max() < 1e-6
        assert np.abs(st1["sW"] - st["sW"]).max() < 1e-12
        print("ok test_a_sealed_box_does_nothing")

    def test_pressure_bc_drives_a_model_with_no_wells():
        """The gap, closed: a model driven entirely by its boundary.

        Water in at 250 bar on the left, out at 150 bar on the right. Before
        this, an AD model could not be driven this way at all.
        """
        g, _, _ = _model()
        bc = pside(None, g, "xmin", 250 * BARSA, sat=np.array([1.0, 0.0]))
        bc = pside(bc, g, "xmax", 150 * BARSA, sat=np.array([0.0, 1.0]))
        g, m, st = _model(bc=bc)

        st = _run(m, st, 5 * DAY, 10)
        p = st["pressure"] / BARSA
        assert np.all(np.diff(p) < 0), "pressure must fall from inlet to outlet"
        assert 150 < p[-1] and p[0] < 250, "and stay between the two boundaries"
        assert st["sW"][0] > 0.6, "water floods in from the water boundary"
        assert np.all(np.diff(st["sW"]) < 0), (
            "and advances as a front - monotone from inlet to outlet"
        )
        print(f"ok test_pressure_bc_drives_a_model_with_no_wells "
              f"({p[0]:.1f} -> {p[-1]:.1f} bar, sW {st['sW'][0]:.2f} -> "
              f"{st['sW'][-1]:.2f})")

    def test_a_pressure_boundary_reaches_equilibrium():
        """Hold both ends at the same pressure and nothing must flow, however
        far from that pressure the field starts."""
        g, _, _ = _model()
        bc = pside(None, g, "xmin", 200 * BARSA, sat=np.array([0.2, 0.8]))
        bc = pside(bc, g, "xmax", 200 * BARSA, sat=np.array([0.2, 0.8]))
        g, m, st = _model(bc=bc)
        st["pressure"] = np.full(g.cells.num, 260.0 * BARSA)   # over-pressured

        st = _run(m, st, 20 * DAY, 6)
        p = st["pressure"] / BARSA
        assert np.allclose(p, 200.0, atol=0.5), (
            f"the field must bleed down to its boundary: {p}"
        )
        print(f"ok test_a_pressure_boundary_reaches_equilibrium "
              f"(260 bar bled down to {p.mean():.1f} bar)")

    def test_source_and_sink():
        """A source and an equal sink, no wells and no boundary."""
        rate = 1e-4
        g, _, _ = _model()
        src = add_source(None, np.array([0]), np.array([rate]),
                         sat=np.array([[1.0, 0.0]]))
        src = add_source(src, np.array([g.cells.num - 1]), np.array([-rate]),
                         sat=np.array([[0.0, 1.0]]))
        g, m, st = _model(src=src)

        pv = pore_volume(g, m.rock)
        w0 = float(np.sum(pv * value(m.get_props(st, "ShrinkageFactors")[0])
                          * st["sW"]))
        dt, n = 5 * DAY, 4
        st = _run(m, st, dt, n)
        w1 = float(np.sum(pv * value(m.get_props(st, "ShrinkageFactors")[0])
                          * st["sW"]))

        injected = rate * n * dt
        assert w1 > w0, "water must accumulate"
        assert w1 - w0 <= injected * (1 + 1e-9), (
            "and cannot accumulate more than was injected"
        )
        assert st["sW"][0] > st["sW"][-1], "the front moves away from the source"
        print(f"ok test_source_and_sink (injected {injected:.3e}, "
              f"stored {w1 - w0:.3e} m3)")

    test_no_forces_is_a_no_op()
    test_a_sealed_box_does_nothing()
    test_pressure_bc_drives_a_model_with_no_wells()
    test_a_pressure_boundary_reaches_equilibrium()
    test_source_and_sink()
    print("\nAll tests passed!")
