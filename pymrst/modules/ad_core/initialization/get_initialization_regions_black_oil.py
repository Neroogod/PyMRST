"""MRST ``ad-core/initialization/getInitializationRegionsBlackOil.m``.

An equilibration region for a black-oil model, over `cells` (all by
default) in the saturation and PVT region of the first cell: phase
densities `rho{i}(p, z)` from the fluid (water `bW*rhoWS`; live oil
`bO(p, rs, saturated)*(rhoOS + rs*rhoGS)` with `rs` capped at `rsSat` and
pressure at the model's minimum; gas likewise with `rv`), capillary curves,
signs and J-function scaling (`getEquilPC`), saturation bounds
(`getMinMaxPhaseSaturations`) and oil as the reference phase. `rs`/`rv` are
constants or functions of `(p, z)`, required for dissolved gas/vaporised
oil, and are stored on the region. Other options go to
`getInitializationRegionsBase`.

`get_initialization_regions_black_oil_ad`, below, is PyMRST's earlier
version, kept for its callers. Its notes follow.

`get_initialization_regions_base_ad` takes a per-phase density and builds the
region. This supplies the densities a *black-oil* fluid has, which is the whole
content of the function, and the three phases differ:

* **water** is simple - `b_w(p) * rhoWS`;
* **oil** carries dissolved gas. Its density is
  `b_o * (rhoOS + rs * rhoGS)`, so a live-oil column is heavier than the same
  oil would be dead, and the hydrostatic gradient the equilibrium integrates is
  correspondingly steeper. Omitting the `rs` term puts the contacts at the
  wrong depths;
* **gas** carries vaporised oil, `b_g * (rhoGS + rv * rhoOS)`, by the same
  argument in the other direction.

`rs` and `rv` may each be a constant or a function of `(p, z)`. A depth-varying
`rs` is the normal case for an undersaturated column, and accepting only a
constant would silently flatten it.

The reference phase is **oil** where the model has it. That is not arbitrary:
the equilibrium places every other phase against a reference by inverting
capillary pressure, and oil is the phase in contact with both of the others in
a three-phase column. Choosing water instead would need the gas-oil contact
expressed through two inversions rather than one.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.initialization.regions import (
    get_initialization_regions_base_ad, get_min_max_phase_saturations_ad,
)


def get_initialization_regions_black_oil(model: Any, contacts: Sequence[float], *,
                                         cells: Any = None, rs: Any = None, rv: Any = None,
                                         **args: Any):
    """MRST `region = getInitializationRegionsBlackOil(model, contacts, ...)`."""
    from copy import copy

    from pymrst.modules.ad_core.initialization.get_initialization_regions_base import (
        get_initialization_regions_base,
    )
    from pymrst.modules.ad_core.initialization.get_min_max_phase_saturations import (
        get_min_max_phase_saturations,
    )
    from pymrst.modules.ad_core.utils.get_equil_pc import get_equil_pc
    cells = np.arange(model.G.cells.num) if cells is None else np.asarray(cells, dtype=np.int64)
    model = copy(model).setup_state_function_groupings()
    nPh = int(np.sum(model.get_active_phases()))
    rho: List[Any] = [None]*nPh
    cc = cells[0]
    f = model.fluid
    satnum = pvtnum = 1
    regions = getattr(model.rock, "regions", None)
    if regions is not None and np.size(regions):
        saturation = _field(regions, "saturation")
        pvt = _field(regions, "pvt")
        if saturation is not None:
            satnum = int(np.asarray(saturation)[cc])
        if pvt is not None:
            pvtnum = int(np.asarray(pvt)[cc])
    pc, pc_sign, pc_scale = get_equil_pc(model, satnum - 1, cells)
    if model.water:
        ix = model.get_phase_index("W")
        bW = _get_function(f, "bW", pvtnum)
        rhoWS = _regional(f, "rhoWS", pvtnum)
        rho[ix] = lambda p, z, bW=bW, rhoWS=rhoWS: np.asarray(bW(p))*rhoWS
    if model.oil:
        ix = model.get_phase_index("O")
        rho[ix] = lambda p, z: _get_oil_density(model, p, z, rs, pvtnum)
    if model.gas:
        ix = model.get_phase_index("G")
        rho[ix] = lambda p, z: _get_gas_density(model, p, z, rv, pvtnum)
    ref_index = model.get_phase_index("O") if model.oil else 0
    s_min, s_max = get_min_max_phase_saturations(model, satnum, cells)
    region = get_initialization_regions_base(
        model, rho, contacts, rho=rho, cells=cells, reference_index=ref_index,
        pc_sign=pc_sign, pc_scale=pc_scale, s_min=s_min, s_max=s_max,
        saturation_region=satnum, pvt_region=pvtnum, pc_functions=pc, **args)
    region.rs = rs
    region.rv = rv
    return region


getInitializationRegionsBlackOil = get_initialization_regions_black_oil


def _get_oil_density(model: Any, p: Any, z: Any, rs: Any, pvtnum: int) -> np.ndarray:
    f = model.fluid
    bO = _get_function(f, "bO", pvtnum)
    rhoOS, rhoGS = _regional(f, "rhoOS", pvtnum), _regional(f, "rhoGS", pvtnum)
    p = np.atleast_1d(np.asarray(p, dtype=float))
    if model.disgas:
        p = np.maximum(p, model.minimumPressure)
        assert rs is not None, "Disgas model must have rs specified as optional argument"
        if callable(rs):
            rs = rs(p, z)
        rs = np.asarray(rs, dtype=float)*np.ones(p.shape)
        rsSat = np.asarray(_get_function(f, "rsSat", pvtnum)(p), dtype=float)
        rs = np.minimum(rs, rsSat)
        return np.asarray(bO(p, np.minimum(rs, rsSat), rs >= rsSat))*(rhoOS + rs*rhoGS)
    return np.asarray(bO(p))*rhoOS


def _get_gas_density(model: Any, p: Any, z: Any, rv: Any, pvtnum: int) -> np.ndarray:
    f = model.fluid
    bG = _get_function(f, "bG", pvtnum)
    rhoOS, rhoGS = _regional(f, "rhoOS", pvtnum), _regional(f, "rhoGS", pvtnum)
    p = np.atleast_1d(np.asarray(p, dtype=float))
    if model.vapoil:
        p = np.maximum(p, model.minimumPressure)
        assert rv is not None, "Vapoil model must have rv specified as optional argument"
        if callable(rv):
            rv = rv(p, z)
        rv = np.asarray(rv, dtype=float)*np.ones(p.shape)
        rvSat = np.asarray(_get_function(f, "rvSat", pvtnum)(p), dtype=float)
        rv = np.minimum(rv, rvSat)
        return np.asarray(bG(p, np.minimum(rv, rvSat), rv >= rvSat))*(rv*rhoOS + rhoGS)
    return np.asarray(bG(p))*rhoGS


def _get_function(fluid: Any, fld: str, reg: int) -> Any:
    """MRST `getFunction`: a fluid function, by PVT region if per region."""
    fn = _field(fluid, fld)
    if isinstance(fn, (list, tuple)):
        fn = fn[reg - 1]
    return fn


def _regional(fluid: Any, fld: str, reg: int) -> float:
    """MATLAB `f.(fld)(reg)`."""
    return float(np.ravel(np.asarray(_field(fluid, fld), dtype=float))[reg - 1])


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def get_initialization_regions_black_oil_ad(
    model: Any,
    contacts: Sequence[float],
    cells: Optional[np.ndarray] = None,
    rs: Any = None,
    rv: Any = None,
    **kwargs: Any,
):
    """Build the equilibrium region, with black-oil phase densities."""
    names = [str(n).upper() for n in model.get_phase_names()]
    fluid = getattr(model, "raw_fluid", None) or model.fluid
    rho_s = _surface_densities(fluid)

    disgas = bool(getattr(model, "disgas", getattr(fluid, "disgas", False)))
    vapoil = bool(getattr(model, "vapoil", getattr(fluid, "vapoil", False)))
    if disgas and rs is None:
        raise ValueError(
            "a model with dissolved gas needs `rs` (a value or a callable of "
            "(p, z)); without it the oil density is the dead-oil one and the "
            "contacts land at the wrong depths")
    if vapoil and rv is None:
        raise ValueError(
            "a model with vaporised oil needs `rv` (a value or a callable of "
            "(p, z))")

    densities: List[Callable] = [None]*len(names)
    if "W" in names:
        index = model.get_phase_index("W")
        densities[index] = _water_density(fluid, rho_s, index)
    if "O" in names:
        index = model.get_phase_index("O")
        densities[index] = _oil_density(model, fluid, rho_s, names, rs, disgas)
    if "G" in names:
        index = model.get_phase_index("G")
        densities[index] = _gas_density(model, fluid, rho_s, names, rv, vapoil)

    reference = model.get_phase_index("O") if "O" in names else 0
    kwargs.setdefault("reference_index", reference)
    kwargs.setdefault("pc_functions", _capillary_functions(fluid, names, model))
    if "s_min" not in kwargs or "s_max" not in kwargs:
        s_min, s_max = get_min_max_phase_saturations_ad(model)
        kwargs.setdefault("s_min", s_min)
        kwargs.setdefault("s_max", s_max)

    region = get_initialization_regions_base_ad(
        model, densities, contacts, cells=cells, **kwargs)
    # Carried on the region so the saturation fill can reach them, as MRST does.
    region.rs = rs
    region.rv = rv
    return region




def capillary_functions(fluid: Any, names: List[str], model: Any,
                        satnum: int = 1) -> Optional[List[Optional[Callable]]]:
    """The deck's capillary curves, per phase, for the saturation fill.

    Without these the equilibrium places a **sharp contact**: every cell above
    the water-oil contact gets connate water and every cell below it gets one.
    A real deck's SWOF has a `Pcow` column, and the transition zone it implies
    can be tens of metres thick - the aquifertest deck's is 1.24 bar at
    `Sw = 0.1`, which against a 200 kg/m3 density difference is 63 m of column.
    ECLIPSE reports `Sw = 0.111` at the top of that model where a sharp
    contact gives 0.

    `pc_functions[i]` is the capillary pressure of phase `i` *against the
    reference phase*, as a function of that phase's own saturation:
    `pcOW(sW) = p_o - p_w` and `pcOG(sG) = p_g - p_o`. Returning `None` when
    the deck states neither keeps the sharp-contact behaviour for the decks
    that genuinely have no capillary pressure.
    """
    functions: List[Optional[Callable]] = [None]*len(names)
    found = False
    if "W" in names:
        pc_ow = _region_curve(fluid, ("pc_ow", "pcOW"), satnum)
        if callable(pc_ow):
            # `pcOW` is `p_o - p_w`, and the fill wants each phase's excess
            # over the reference - so the water entry is negated.
            functions[model.get_phase_index("W")] = (
                lambda s, fn=pc_ow: -np.asarray(fn(s), dtype=float))
            found = True
    if "G" in names:
        pc_og = _region_curve(fluid, ("pc_og", "pcOG"), satnum)
        if callable(pc_og):
            functions[model.get_phase_index("G")] = (
                lambda s, fn=pc_og: np.asarray(fn(s), dtype=float))
            found = True
    return functions if found else None


def _region_curve(fluid: Any, names, satnum: int):
    """One saturation region's curve, however the fluid stores it.

    A multi-SATNUM deck gives `assign_swof` a list with one curve per region;
    a single-region deck gives a bare callable. Taking a list and calling it
    raises; taking `[0]` of a bare callable is not possible either, so both
    shapes are handled rather than assumed.
    """
    for name in names:
        found = getattr(fluid, name, None)
        if found is None:
            continue
        if callable(found):
            return found
        try:
            entries = list(found)
        except TypeError:
            continue
        if entries:
            return entries[min(max(int(satnum) - 1, 0), len(entries) - 1)]
    return None


_capillary_functions = capillary_functions


def _surface_densities(fluid: Any) -> np.ndarray:
    """The `[water, oil, gas]` surface densities, however the fluid spells it.

    `PhaseFluid` writes `rho_s`; `ADFluid` and `BlackOilFluid` - the two the
    *raw* fluid is ever one of - both write `rhoS`. Reading only `rho_s` meant
    this function worked against a stub and raised on every real model, which
    is why nothing caught it.
    """
    for name in ("rho_s", "rhoS"):
        values = getattr(fluid, name, None)
        if values is not None:
            return np.asarray(values, dtype=float).ravel()
    raise AttributeError(
        "the fluid carries no surface densities (rho_s / rhoS); the "
        "equilibrium column cannot be formed without them")


def _water_density(fluid: Any, rho_s: np.ndarray, index: int) -> Callable:
    b_w = _b_function(fluid, "w", index)
    return lambda p, z=None: np.asarray(b_w(p), dtype=float)*rho_s[index]


def _oil_density(model: Any, fluid: Any, rho_s: np.ndarray, names: List[str],
                 rs: Any, disgas: bool) -> Callable:
    oil = model.get_phase_index("O")
    b_o = _b_function(fluid, "o", oil)
    gas = model.get_phase_index("G") if "G" in names else None

    def density(p, z=None):
        p = np.asarray(p, dtype=float)
        if not disgas or gas is None:
            return np.asarray(b_o(p), dtype=float)*rho_s[oil]
        floor = float(getattr(model, "minimum_pressure", -np.inf))
        p = np.maximum(p, floor)
        ratio = _evaluate(rs, p, z)
        # Dissolved gas makes the oil column heavier; the gradient it produces
        # is what sets the contact depths.
        return np.asarray(b_o(p), float)*(rho_s[oil] + ratio*rho_s[gas])

    return density


def _gas_density(model: Any, fluid: Any, rho_s: np.ndarray, names: List[str],
                 rv: Any, vapoil: bool) -> Callable:
    gas = model.get_phase_index("G")
    b_g = _b_function(fluid, "g", gas)
    oil = model.get_phase_index("O") if "O" in names else None

    def density(p, z=None):
        p = np.asarray(p, dtype=float)
        if not vapoil or oil is None:
            return np.asarray(b_g(p), dtype=float)*rho_s[gas]
        ratio = _evaluate(rv, p, z)
        return np.asarray(b_g(p), float)*(rho_s[gas] + ratio*rho_s[oil])

    return density


def _b_function(fluid: Any, phase: str, index: int) -> Callable:
    """The phase's shrinkage factor, under whichever spelling the fluid uses."""
    for name in (f"b_{phase}", f"b{phase.upper()}"):
        found = getattr(fluid, name, None)
        if callable(found):
            return found
    vectorised = getattr(fluid, "b", None)
    if callable(vectorised):
        return lambda p: vectorised(p)[index]
    raise AttributeError(
        f"fluid has no shrinkage factor for phase {phase.upper()!r}; the "
        f"equilibrium density cannot be formed without it")


def _evaluate(ratio: Any, p: np.ndarray, z: Any) -> np.ndarray:
    """A dissolution ratio that may be constant or a function of `(p, z)`.

    A depth-varying `rs` is the normal case for an undersaturated column, so
    accepting only a constant would silently flatten it.
    """
    if callable(ratio):
        try:
            return np.asarray(ratio(p, z), dtype=float)
        except TypeError:
            return np.asarray(ratio(p), dtype=float)
    return np.asarray(ratio, dtype=float)


__all__ = ["getInitializationRegionsBlackOil", "get_initialization_regions_black_oil",
           "get_initialization_regions_black_oil_ad"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _model(phases=("W", "O", "G"), disgas=False, vapoil=False):
        names = list(phases)
        fluid = SimpleNamespace(
            rho_s=np.array([1000.0, 800.0, 1.2])[:len(names)],
            b_w=lambda p: np.ones_like(np.asarray(p, float)),
            b_o=lambda p: np.ones_like(np.asarray(p, float)),
            b_g=lambda p: np.ones_like(np.asarray(p, float)),
            swcon=0.1, disgas=disgas, vapoil=vapoil)
        return SimpleNamespace(
            fluid=fluid, raw_fluid=None, n_phases=len(names),
            disgas=disgas, vapoil=vapoil, minimum_pressure=1.0e5,
            water=("W" in names), oil=("O" in names), gas=("G" in names),
            get_phase_names=lambda: names,
            get_phase_index=lambda p: names.index(p),
            getNumberOfPhases=lambda: len(names))

    def test_dead_oil_density_is_just_b_times_surface():
        model = _model()
        reg = get_initialization_regions_black_oil_ad(model, [100.0, 200.0])
        p = np.array([2.0e7])
        assert np.allclose(reg.rho[1](p), 800.0)
        print("ok test_dead_oil_density_is_just_b_times_surface")

    def test_dissolved_gas_makes_the_oil_heavier():
        """The term whose omission puts the contacts at the wrong depth."""
        model = _model(disgas=True)
        reg = get_initialization_regions_black_oil_ad(model, [100.0, 200.0],
                                                   rs=50.0)
        p = np.array([2.0e7])
        heavy = float(np.ravel(reg.rho[1](p))[0])
        assert heavy > 800.0, heavy
        assert np.isclose(heavy, 800.0 + 50.0*1.2)
        print(f"ok test_dissolved_gas_makes_the_oil_heavier ({heavy:.1f})")

    def test_a_depth_varying_rs_is_accepted():
        """An undersaturated column has `rs` falling with depth; taking only a
        constant would flatten it."""
        model = _model(disgas=True)
        reg = get_initialization_regions_black_oil_ad(
            model, [100.0, 200.0], rs=lambda p, z: np.full_like(p, 20.0))
        p = np.array([2.0e7])
        assert np.isclose(float(np.ravel(reg.rho[1](p))[0]), 800.0 + 20.0*1.2)
        print("ok test_a_depth_varying_rs_is_accepted")

    def test_a_disgas_model_without_rs_is_refused():
        model = _model(disgas=True)
        try:
            get_initialization_regions_black_oil_ad(model, [100.0, 200.0])
        except ValueError as refused:
            assert "rs" in str(refused)
            print("ok test_a_disgas_model_without_rs_is_refused")
            return
        raise AssertionError("a live-oil model needs rs")

    def test_oil_is_the_reference_phase():
        """Oil is in contact with both other phases, so every other phase is
        placed against it by one capillary inversion."""
        reg = get_initialization_regions_black_oil_ad(_model(), [100.0, 200.0])
        assert reg.reference_index == 1
        print("ok test_oil_is_the_reference_phase")

    def test_a_fluid_that_spells_it_rhoS_is_accepted():
        """The stub below writes `rho_s`; `ADFluid` and `BlackOilFluid` - the
        only two the *raw* fluid is ever one of - both write `rhoS`. Reading
        only `rho_s` meant this function raised on every real model while its
        own tests passed."""
        model = _model()
        model.fluid.rhoS = model.fluid.rho_s
        del model.fluid.rho_s
        reg = get_initialization_regions_black_oil_ad(model, [100.0, 200.0])
        assert np.allclose(reg.rho[1](np.array([2.0e7])), 800.0)
        print("ok test_a_fluid_that_spells_it_rhoS_is_accepted")

    def test_a_fluid_with_neither_spelling_is_refused():
        model = _model()
        del model.fluid.rho_s
        try:
            get_initialization_regions_black_oil_ad(model, [100.0, 200.0])
        except AttributeError as refused:
            assert "surface densities" in str(refused)
            print("ok test_a_fluid_with_neither_spelling_is_refused")
            return
        raise AssertionError("the densities cannot be invented")

    test_dead_oil_density_is_just_b_times_surface()
    test_a_fluid_that_spells_it_rhoS_is_accepted()
    test_a_fluid_with_neither_spelling_is_refused()
    test_dissolved_gas_makes_the_oil_heavier()
    test_a_depth_varying_rs_is_accepted()
    test_a_disgas_model_without_rs_is_refused()
    test_oil_is_the_reference_phase()
    print("\nAll tests passed!")
