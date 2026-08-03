"""Explicit surface-volume conservation adapters for reservoir models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from pymrst.utils.adi import value


class ConservationAdapter(Protocol):
    component_names: tuple[str, ...]
    units: str
    sign_convention: str
    def inventory(self, state: Any) -> np.ndarray: ...
    def well_rates(self, state: Any) -> np.ndarray: ...


def _field(state, name):
    if isinstance(state, dict):
        return state[name]
    return getattr(state, name)


def _phase_pore_volume(model, state) -> np.ndarray:
    if ((isinstance(state, dict) and "pore_volume" in state)
            or hasattr(state, "pore_volume")):
        return np.asarray(_field(state, "pore_volume"), dtype=float).reshape(-1)
    if hasattr(model, "get_props"):
        try:
            return np.asarray(value(model.get_props(state, "PoreVolume")), dtype=float)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    for name in ("pv", "pv_ref"):
        if hasattr(model, name):
            pv = np.asarray(getattr(model, name), dtype=float).reshape(-1)
            fluid = getattr(model, "raw_fluid", getattr(model, "fluid", None))
            multiplier = getattr(fluid, "pv_mult", None)
            if name == "pv_ref" and multiplier is not None:
                pressure = np.asarray(_field(state, "pressure"), dtype=float)
                pv = pv*np.asarray(value(multiplier(pressure)), dtype=float)
            return pv
    raise TypeError("model must expose PoreVolume, pv, or pv_ref")


@dataclass(frozen=True, slots=True)
class ConservationReport:
    component_names: tuple[str, ...]
    units: str
    initial: np.ndarray
    final: np.ndarray
    change: np.ndarray
    wells: np.ndarray
    boundaries: np.ndarray
    sources: np.ndarray
    imbalance: np.ndarray
    normalized_imbalance: np.ndarray
    regional_imbalance: np.ndarray | None = None
    coupling_diagnostics: dict[str, float] | None = None

    @property
    def maximum_normalized_imbalance(self) -> float:
        return float(np.max(np.abs(self.normalized_imbalance), initial=0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_names": list(self.component_names), "units": self.units,
            "initial": self.initial.tolist(), "final": self.final.tolist(),
            "change": self.change.tolist(), "wells": self.wells.tolist(),
            "boundaries": self.boundaries.tolist(), "sources": self.sources.tolist(),
            "imbalance": self.imbalance.tolist(),
            "normalized_imbalance": self.normalized_imbalance.tolist(),
            "maximum_normalized_imbalance": self.maximum_normalized_imbalance,
            "regional_imbalance": (self.regional_imbalance.tolist()
                                   if self.regional_imbalance is not None else None),
            "coupling_diagnostics": self.coupling_diagnostics,
        }


def _water_shrinkage_factor(model, state, pressure) -> np.ndarray:
    """b for the water phase, from whichever source the model actually owns.

    The state-function graph is authoritative when present: it is what the
    residual is assembled from, so an inventory built any other way is
    measuring a different quantity than the equation conserves. `b_w` on a raw
    fluid is the fallback for models with no graph.

    Raises when the model exposes neither, rather than assuming an
    incompressible fluid. An incompressible fluid and a b that could not be
    found produce the same number, and only one of them is an answer - this
    function returned ones in that case until doing so put 5.25e-07 on the
    record as physics. Every sibling adapter already refuses a model it cannot
    read; no case in the characterization set reaches this branch.

    A genuinely incompressible model should say so by exposing a b of one,
    which costs its author a line and costs a reader nothing.
    """
    if hasattr(model, "get_props"):
        try:
            factors = model.get_props(state, "ShrinkageFactors")
            if isinstance(factors, tuple) and len(factors) == 1:
                factors = factors[0]
            first = factors[0] if isinstance(factors, (list, tuple)) else factors
            return np.asarray(value(first), dtype=float).reshape(-1)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            pass
    fluid = getattr(model, "raw_fluid", getattr(model, "fluid", None))
    b_function = getattr(fluid, "b_w", None)
    if b_function is None:
        raise AttributeError(
            "water model must expose a ShrinkageFactors state function or a "
            "fluid with b_w; refusing to assume an incompressible fluid"
        )
    return np.asarray(value(b_function(pressure)), dtype=float).reshape(-1)


class WaterConservationAdapter:
    """Conserve surface water.

    The shrinkage factor is taken from the model's own state-function graph
    first, and only then from a fluid's ``b_w``.  Order matters: a
    poromechanics model's ``PhaseFluid`` indexes b per phase and exposes no
    ``b_w``, so the previous lookup found nothing and quietly used ``b = 1``.
    The model's actual b is 1.0000001 - a hundred-nanoscale difference that
    looked like nothing and was not: it put the reported imbalance at 5.25e-07
    where the same run using the model's own b reports 1.74e-16.

    That figure was on record as evidence of poromechanical coupling, and the
    ten-order spread between model families was read as something a
    conservation budget would have to explain. It was an artifact of this
    fallback. A silent default is worse than a missing adapter, because it
    reports a plausible number computed from a property the model never
    supplied.
    """

    component_names = ("water",)
    units = "surface_m3"
    sign_convention = "positive_into_reservoir"

    def __init__(self, model):
        self.model = model

    def inventory(self, state) -> np.ndarray:
        pressure = np.asarray(_field(state, "pressure"), dtype=float)
        pv = _phase_pore_volume(self.model, state)
        return (pv*_water_shrinkage_factor(self.model, state, pressure))[:, None]

    def well_rates(self, state) -> np.ndarray:
        return np.array([np.sum(np.asarray(_field(state, "qWs"), dtype=float))])


class OilWaterConservationAdapter:
    """Conserve surface water and stock-tank oil.

    The two b-factors are resolved the same way: ``raw_fluid`` before
    ``fluid``.  A model may carry both - a poromechanics oil-water model has a
    ``PhaseFluid`` under ``fluid``, whose b-factors are indexed per phase, and
    an ``ADFluid`` under ``raw_fluid``, which is the one with ``b_w``.  Reading
    ``model.fluid`` directly picked the first of those and raised
    ``AttributeError`` on exactly the models this adapter exists to serve,
    while the oil half of the same inventory already resolved the fluid
    correctly through ``get_b_o_bo``.
    """

    component_names = ("water", "oil")
    units = "surface_m3"
    sign_convention = "positive_into_reservoir"

    def __init__(self, model):
        self.model = model

    def inventory(self, state) -> np.ndarray:
        from pymrst.modules.ad_blackoil.utils.get_b_o_bo import get_b_o_bo
        pressure = np.asarray(_field(state, "pressure"), dtype=float)
        saturation = np.asarray(_field(state, "sW"), dtype=float)
        pv = _phase_pore_volume(self.model, state)
        fluid = getattr(self.model, "raw_fluid", getattr(self.model, "fluid", None))
        b_water = getattr(fluid, "b_w", None)
        if b_water is None:
            # Not defaulted to one. `get_b_o_bo` raises when the oil b-factor
            # is missing, and a silent unit water b-factor would report a
            # plausible imbalance computed from a property the model never
            # supplied.
            raise AttributeError("oil-water model fluid must define b_w")
        bw = np.asarray(value(b_water(pressure)), dtype=float)
        bo = np.asarray(value(get_b_o_bo(self.model, pressure)), dtype=float)
        return np.column_stack((pv*bw*saturation, pv*bo*(1.0-saturation)))

    def well_rates(self, state) -> np.ndarray:
        return np.array([
            np.sum(np.asarray(_field(state, "qWs"), dtype=float)),
            np.sum(np.asarray(_field(state, "qOs"), dtype=float)),
        ])


def _blackoil_saturations(state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return water, oil and gas saturation from MRST-style state variants."""
    try:
        sw = np.asarray(value(_field(state, "sW")), dtype=float).reshape(-1)
        sg = np.asarray(value(_field(state, "sG")), dtype=float).reshape(-1)
        try:
            so = np.asarray(value(_field(state, "sO")), dtype=float).reshape(-1)
        except (AttributeError, KeyError):
            so = 1.0-sw-sg
        return sw, so, sg
    except (AttributeError, KeyError):
        saturation = np.asarray(value(_field(state, "s")), dtype=float)
        if saturation.ndim != 2 or saturation.shape[1] < 3:
            raise ValueError("black-oil state saturation must have W/O/G columns")
        return saturation[:, 0], saturation[:, 1], saturation[:, 2]


def _blackoil_shrinkage_factors(model, state, pressure, rs, sg):
    """Resolve bW/bO/bG through the generic graph or the direct fluid API."""
    if hasattr(model, "get_props"):
        try:
            factors = model.get_props(state, "ShrinkageFactors")
            if isinstance(factors, tuple) and len(factors) == 1:
                factors = factors[0]
            if len(factors) >= 3:
                return tuple(
                    np.asarray(value(factor), dtype=float).reshape(-1)
                    for factor in factors[:3]
                )
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    fluid = getattr(model, "raw_fluid", getattr(model, "fluid", None))
    if fluid is None:
        raise TypeError("black-oil model must expose fluid properties")
    bw = np.asarray(value(fluid.b_w(pressure)), dtype=float).reshape(-1)
    bg = np.asarray(value(fluid.b_g(pressure)), dtype=float).reshape(-1)
    try:
        bo_raw = fluid.b_o(pressure, rs, sg > 0.0)
    except TypeError:
        try:
            bo_raw = fluid.b_o(pressure, rs)
        except TypeError:
            bo_raw = fluid.b_o(pressure)
    bo = np.asarray(value(bo_raw), dtype=float).reshape(-1)
    return bw, bo, bg


class BlackOilConservationAdapter:
    """Conserve three black-oil *surface components*.

    The reported components are surface water, stock-tank oil, and total
    surface gas.  Total gas includes both free gas and gas dissolved in oil,
    so its cell inventory is ``PV*(bG*sG + Rs*bO*sO)``.  This is the same
    component basis used by PyMRST's black-oil residual and well unknowns;
    consequently ``qGs`` must not be augmented by ``Rs*qOs`` a second time.
    """

    component_names = ("water", "oil", "gas")
    units = "surface_m3"
    sign_convention = "positive_into_reservoir"

    def __init__(self, model):
        self.model = model

    def inventory(self, state) -> np.ndarray:
        pressure = np.asarray(value(_field(state, "pressure")), dtype=float).reshape(-1)
        sw, so, sg = _blackoil_saturations(state)
        rs = np.asarray(value(_field(state, "rs")), dtype=float).reshape(-1)
        pv = _phase_pore_volume(self.model, state)
        bw, bo, bg = _blackoil_shrinkage_factors(
            self.model, state, pressure, rs, sg,
        )
        sizes = {a.size for a in (pressure, sw, so, sg, rs, pv, bw, bo, bg)}
        if len(sizes) != 1:
            raise ValueError("black-oil state and property arrays must align by cell")
        return np.column_stack((
            pv*bw*sw,
            pv*bo*so,
            pv*(bg*sg + rs*bo*so),
        ))

    def well_rates(self, state) -> np.ndarray:
        return np.array([
            np.sum(np.asarray(value(_field(state, "qWs")), dtype=float)),
            np.sum(np.asarray(value(_field(state, "qOs")), dtype=float)),
            np.sum(np.asarray(value(_field(state, "qGs")), dtype=float)),
        ])


class CompositionalConservationAdapter:
    """Conserve EOS-component mass across liquid and vapour phases.

    This adapter deliberately uses the same mass basis as
    ``OverallCompositionModel.assemble``: for component ``i`` the cell
    inventory is ``PV*pvm*(rhoL*sL*xMass_i + rhoV*sV*yMass_i)``.  Phase split
    and phase compositions may therefore change without creating or destroying
    an overall component.  ``well_components``/``wellSol.components`` are
    already signed component mass rates in PyMRST.
    """

    units = "kg"
    sign_convention = "positive_into_reservoir"

    def __init__(self, model):
        self.model = model
        mixture = getattr(model, "mix", None)
        names = getattr(mixture, "names", None)
        ncomp = getattr(mixture, "n", None)
        if names is None and ncomp is None:
            raise TypeError("compositional model must expose mix.names or mix.n")
        self.component_names = tuple(
            str(name) for name in
            (names if names is not None else [f"Component {i + 1}" for i in range(ncomp)])
        )

    def inventory(self, state) -> np.ndarray:
        try:
            explicit = _field(state, "component_mass_inventory")
        except (AttributeError, KeyError):
            explicit = None
        if explicit is not None:
            result = np.asarray(value(explicit), dtype=float)
        else:
            flow_props = getattr(self.model, "_flow_props0", None)
            if flow_props is None:
                raise TypeError(
                    "compositional model must expose _flow_props0 or state "
                    "component_mass_inventory"
                )
            x_mass, y_mass, s_liquid, s_vapour, rho_liquid, rho_vapour = flow_props(state)
            x_mass = np.column_stack([
                np.asarray(value(column), dtype=float).reshape(-1) for column in x_mass
            ])
            y_mass = np.column_stack([
                np.asarray(value(column), dtype=float).reshape(-1) for column in y_mass
            ])
            s_liquid = np.asarray(value(s_liquid), dtype=float).reshape(-1)
            s_vapour = np.asarray(value(s_vapour), dtype=float).reshape(-1)
            if getattr(self.model, "water", False):
                hydrocarbon_fraction = 1.0-np.asarray(
                    value(_field(state, "sW")), dtype=float,
                ).reshape(-1)
                s_liquid = hydrocarbon_fraction*s_liquid
                s_vapour = hydrocarbon_fraction*s_vapour
            rho_liquid = np.asarray(value(rho_liquid), dtype=float).reshape(-1)
            rho_vapour = np.asarray(value(rho_vapour), dtype=float).reshape(-1)
            pv = _phase_pore_volume(self.model, state)
            if getattr(self.model, "rock_pv_mult", None) is not None:
                pressure = np.asarray(value(_field(state, "pressure")), dtype=float)
                pv = pv*np.asarray(
                    value(self.model.rock_pv_mult(pressure)), dtype=float,
                ).reshape(-1)
            result = pv[:, None]*(
                (rho_liquid*s_liquid)[:, None]*x_mass
                + (rho_vapour*s_vapour)[:, None]*y_mass
            )
        if result.ndim != 2 or result.shape[1] != len(self.component_names):
            raise ValueError("component inventory must be cell-by-EOS-component")
        return result

    def well_rates(self, state) -> np.ndarray:
        try:
            rates = _field(state, "well_components")
        except (AttributeError, KeyError):
            rates = None
        if rates is not None:
            array = np.asarray(value(rates), dtype=float)
            if array.size == 0:
                return np.zeros(len(self.component_names))
            return array.reshape(-1, len(self.component_names)).sum(axis=0)
        try:
            solutions = _field(state, "wellSol")
        except (AttributeError, KeyError):
            solutions = []
        total = np.zeros(len(self.component_names))
        for solution in solutions:
            components = (_field(solution, "components")
                          if solution is not None else None)
            if components is not None:
                total += np.asarray(value(components), dtype=float).reshape(
                    -1, len(self.component_names),
                ).sum(axis=0)
        return total


class BlackOilPseudoComponentConservationAdapter:
    """Conserve the two pseudo-components of a converted black-oil model.

    ``convertBlackOilModelToCompositionalModel`` produces a two-component
    mixture, ``PseudoOil``/``PseudoGas``, whose model object is a thin wrapper
    rather than an ``OverallCompositionModel``: it exposes ``mixture`` instead
    of ``mix`` and carries no ``_flow_props0``, so
    `CompositionalConservationAdapter` cannot read it.  The mass basis is the
    same one that adapter documents - for component ``i`` the cell inventory is
    ``PV*(rhoO*sO*x_i + rhoG*sG*y_i)`` - but the phase compositions and
    densities come from the state's ``blackoil_mass_fractions``.

    Well rates use the defining property of the conversion: the surface oil
    stream *is* ``PseudoOil`` and the surface gas stream *is* ``PseudoGas``, so
    the component mass rates are ``qOs*rhoOS`` and ``qGs*rhoGS``.  That is the
    same conversion `spe1_reference_pseudo_component_rates` applies to build
    the MRST reference well curves, and it is what makes the balance close:
    with the two components transposed the imbalance is order one rather than
    order ``1e-5``.
    """

    units = "kg"
    sign_convention = "positive_into_reservoir"
    component_names = ("PseudoOil", "PseudoGas")

    def __init__(self, model):
        self.model = model
        mixture = getattr(model, "mix", None)
        if mixture is None:
            mixture = getattr(model, "mixture", None)
        names = tuple(str(name) for name in (getattr(mixture, "names", None) or ()))
        if names != self.component_names:
            raise TypeError(
                "black-oil pseudo-component adapter expects the "
                f"{self.component_names} mixture, got {names or None}"
            )
        blackoil = getattr(model, "blackoil_model", None)
        if blackoil is None:
            raise TypeError("converted model must retain its blackoil_model")
        self._blackoil = blackoil
        density = np.asarray(
            getattr(getattr(blackoil, "fluid", None), "rho_s", None), dtype=float,
        ).reshape(-1)
        if density.size != 3:
            raise TypeError("black-oil fluid must expose three surface densities")
        self._surface_oil_density = float(density[1])
        self._surface_gas_density = float(density[2])

    def inventory(self, state) -> np.ndarray:
        fractions = _field(state, "blackoil_mass_fractions")
        pv = _phase_pore_volume(self._blackoil, state)
        s_oil = np.asarray(value(_field(state, "sO")), dtype=float).reshape(-1)
        s_gas = np.asarray(value(_field(state, "sG")), dtype=float).reshape(-1)
        rho_oil = np.asarray(value(fractions.rhoO), dtype=float).reshape(-1)
        rho_gas = np.asarray(value(fractions.rhoG), dtype=float).reshape(-1)
        liquid = np.column_stack([
            np.asarray(value(fractions.xo), dtype=float).reshape(-1),
            np.asarray(value(fractions.xg), dtype=float).reshape(-1),
        ])
        vapour = np.column_stack([
            np.asarray(value(fractions.yo), dtype=float).reshape(-1),
            np.asarray(value(fractions.yg), dtype=float).reshape(-1),
        ])
        sizes = {a.shape[0] for a in (pv, s_oil, s_gas, rho_oil, rho_gas, liquid, vapour)}
        if len(sizes) != 1:
            raise ValueError("black-oil pseudo-component arrays must align by cell")
        return pv[:, None]*(
            (rho_oil*s_oil)[:, None]*liquid + (rho_gas*s_gas)[:, None]*vapour
        )

    def well_rates(self, state) -> np.ndarray:
        return np.array([
            np.sum(np.asarray(value(_field(state, "qOs")), dtype=float))
            * self._surface_oil_density,
            np.sum(np.asarray(value(_field(state, "qGs")), dtype=float))
            * self._surface_gas_density,
        ])


class PoromechanicsConservationAdapter:
    """Fluid-component conservation with explicit poromechanical diagnostics.

    Mechanical equilibrium is a force balance, not an additional conserved
    mass.  This adapter therefore delegates inventory and well rates to the
    selected fluid-component adapter.  The coupled model's ``PoreVolume`` state
    function remains authoritative, so displacement/volumetric strain enters
    accumulation exactly as it does in the numerical equations.

    ``coupling_diagnostics`` separately reports displacement, volumetric-strain
    and effective-pore-volume changes.  These quantities are evidence about the
    coupling and are intentionally excluded from the component imbalance.
    """

    def __init__(self, model, fluid_basis: str = "auto"):
        self.model = model
        flow_model = getattr(model, "fluidModel", model)
        basis = fluid_basis.lower().replace("_", "-").replace(" ", "-")
        if basis == "auto":
            name = type(flow_model).__name__.lower()
            if "oilwater" in name or "oil-water" in name:
                basis = "oil-water"
            elif "blackoil" in name or "black-oil" in name:
                basis = "black-oil"
            elif "water" in name:
                basis = "water"
            else:
                raise ValueError(
                    "cannot infer poromechanics fluid basis; pass water, "
                    "oil-water, or black-oil"
                )
        adapters = {
            "water": WaterConservationAdapter,
            "oil-water": OilWaterConservationAdapter,
            "black-oil": BlackOilConservationAdapter,
            "blackoil": BlackOilConservationAdapter,
        }
        try:
            adapter_type = adapters[basis]
        except KeyError as error:
            raise ValueError(f"unsupported poromechanics fluid basis: {fluid_basis!r}") from error
        self.fluid_adapter = adapter_type(flow_model)
        self.component_names = self.fluid_adapter.component_names
        self.units = self.fluid_adapter.units
        self.sign_convention = self.fluid_adapter.sign_convention

    def inventory(self, state) -> np.ndarray:
        return self.fluid_adapter.inventory(state)

    def well_rates(self, state) -> np.ndarray:
        return self.fluid_adapter.well_rates(state)

    def coupling_diagnostics(self, initial_state, final_state) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        for label, state in (("initial", initial_state), ("final", final_state)):
            displacement = None
            displacement_field = None
            for field in ("u", "uu", "xd"):
                try:
                    displacement = np.asarray(value(_field(state, field)), dtype=float)
                    displacement_field = field
                    break
                except (AttributeError, KeyError):
                    pass
            if displacement is not None:
                diagnostics[f"{label}_displacement_l2"] = float(
                    np.linalg.norm(displacement.reshape(-1)),
                )
            strain = None
            for field in ("vdiv", "divu", "volumetric_strain"):
                try:
                    strain = np.asarray(value(_field(state, field)), dtype=float).reshape(-1)
                    break
                except (AttributeError, KeyError):
                    pass
            if strain is None and displacement is not None:
                mechanics = getattr(self.model, "mech", getattr(self.model, "mechModel", None))
                method = getattr(mechanics, "volumetric_strain_full", None)
                if method is not None and displacement_field != "xd":
                    strain = np.asarray(value(method(displacement)), dtype=float).reshape(-1)
            if strain is not None:
                diagnostics[f"{label}_volumetric_strain_l2"] = float(np.linalg.norm(strain))
                diagnostics[f"{label}_mean_volumetric_strain"] = float(np.mean(strain))
            try:
                pv = _phase_pore_volume(self.fluid_adapter.model, state)
            except (AttributeError, KeyError, TypeError, ValueError):
                pv = None
            if pv is not None:
                diagnostics[f"{label}_effective_pore_volume"] = float(np.sum(pv))
        if ("initial_effective_pore_volume" in diagnostics
                and "final_effective_pore_volume" in diagnostics):
            diagnostics["effective_pore_volume_change"] = (
                diagnostics["final_effective_pore_volume"]
                - diagnostics["initial_effective_pore_volume"]
            )
        return diagnostics


def aquifer_component_rates(
    adapter: ConservationAdapter,
    initial_state: Any,
    final_state: Any,
    dt: float,
) -> np.ndarray:
    """Recover signed analytic-aquifer rates from committed state changes.

    Fetkovich aquifers store remaining volume, whereas Carter-Tracy aquifers
    store cumulative influx. Both are converted to positive-into-reservoir
    water rates on the adapter's declared component basis. State differences
    are used deliberately: recomputing a pressure-driven flux with updated
    end-of-step aquifer history would not reproduce the accepted step.
    """
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    names = tuple(adapter.component_names)
    rates = np.zeros(len(names))
    try:
        water_index = names.index("water")
    except ValueError:
        return rates

    def optional(state, name):
        try:
            return np.asarray(value(_field(state, name)), dtype=float).reshape(-1)
        except (AttributeError, KeyError):
            return None

    v0 = optional(initial_state, "aquifer_V")
    v1 = optional(final_state, "aquifer_V")
    w0 = optional(initial_state, "aquifer_ct_W")
    w1 = optional(final_state, "aquifer_ct_W")
    delivered = 0.0
    if v0 is not None or v1 is not None:
        if v0 is None or v1 is None or v0.shape != v1.shape:
            raise ValueError("Fetkovich aquifer volume state is incomplete or misaligned")
        delivered += float(np.sum(v0-v1))
    if w0 is not None or w1 is not None:
        if w0 is None or w1 is None or w0.shape != w1.shape:
            raise ValueError("Carter-Tracy cumulative influx state is incomplete or misaligned")
        delivered += float(np.sum(w1-w0))
    rates[water_index] = delivered/dt
    return rates


def conservation_report(
    adapter: ConservationAdapter,
    initial_state: Any,
    final_state: Any,
    dt: float,
    *,
    boundary_rates=None,
    source_rates=None,
    regions=None,
) -> ConservationReport:
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    initial_cells = np.asarray(adapter.inventory(initial_state), dtype=float)
    final_cells = np.asarray(adapter.inventory(final_state), dtype=float)
    if initial_cells.shape != final_cells.shape or initial_cells.ndim != 2:
        raise ValueError("inventory must return matching cell-by-component arrays")
    ncomp = len(adapter.component_names)
    if initial_cells.shape[1] != ncomp:
        raise ValueError("inventory component count does not match adapter declaration")
    initial, final = initial_cells.sum(axis=0), final_cells.sum(axis=0)
    change = final-initial
    wells = dt*np.asarray(adapter.well_rates(final_state), dtype=float).reshape(ncomp)
    boundaries = dt*np.asarray(
        np.zeros(ncomp) if boundary_rates is None else boundary_rates,
        dtype=float,
    ).reshape(ncomp)
    sources = dt*np.asarray(
        np.zeros(ncomp) if source_rates is None else source_rates,
        dtype=float,
    ).reshape(ncomp)
    imbalance = change-wells-boundaries-sources
    scale = np.maximum.reduce((np.abs(initial), np.abs(final), np.abs(wells)+np.abs(boundaries)+np.abs(sources), np.ones(ncomp)))
    normalized = imbalance/scale
    regional = None
    if regions is not None:
        region_ids = np.asarray(regions).reshape(-1)
        if region_ids.size != initial_cells.shape[0]:
            raise ValueError("regions must contain one value per inventory cell")
        unique = np.unique(region_ids)
        regional = np.vstack([
            (final_cells[region_ids == region].sum(axis=0)
             - initial_cells[region_ids == region].sum(axis=0))
            for region in unique
        ])
    coupling = None
    diagnostics = getattr(adapter, "coupling_diagnostics", None)
    if diagnostics is not None:
        coupling = diagnostics(initial_state, final_state)
    return ConservationReport(
        tuple(adapter.component_names), adapter.units, initial, final, change,
        wells, boundaries, sources, imbalance, normalized, regional, coupling,
    )


__all__ = [
    "BlackOilConservationAdapter", "BlackOilPseudoComponentConservationAdapter",
    "CompositionalConservationAdapter",
    "ConservationAdapter", "ConservationReport", "OilWaterConservationAdapter",
    "PoromechanicsConservationAdapter", "WaterConservationAdapter",
    "aquifer_component_rates", "conservation_report",
]
