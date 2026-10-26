"""Simple AD-ready fluid, ported from MRST ``initSimpleADIFluid``.

Shrinkage factors b(p) = exp(c (p - p_ref)) give constant-compressibility
phases; relative permeabilities are power laws of each phase's own
saturation. All property functions accept NumPy arrays or ADI variables.

**One, two or three phases**, decided by the length of `mu`/`rho`/`n`/`c` -
MRST's `'phases', 'W'`, `'wo'` and `'wog'`. The three-phase form is what
`blackoilTutorialMexAcceleration` builds, and refusing it meant that example
could not be ported at all.

The shorter forms are padded to three slots internally so that every consumer
can index `b`, `mu` and `kr` by phase without asking how many there are. The
padding is never evaluated: a two-phase model never asks for phase 2. What is
*not* padded is `rhoS`, which keeps the caller's length, because a model reads
its phase count from it.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from pymrst.utils.adi import ad_exp, ad_max, ad_min


@dataclass
class ADFluid:
    """A water/oil/gas fluid for the AD-OO models, with one to three phases."""

    rhoS: np.ndarray          # surface densities, in [water, oil, gas] order
    mu: np.ndarray            # constant viscosities, same order
    b_w: Callable             # shrinkage factor of pressure
    b_o: Callable
    kr_w: Callable            # relperm of own saturation
    kr_o: Callable
    # pcOW(sW) = p_oil - p_water. None means the phases share one pressure,
    # which is what every PyMRST model assumed before capillarity existed.
    pc_ow: Optional[Callable] = None

    # -- the ad_core fluid contract ------------------------------------------
    #
    # Deliberately *not* dataclass fields (no annotation): a field would join
    # the generated `__init__`, `__eq__` and `__repr__`, and every existing
    # `ADFluid(...)` call and equality comparison would be answering a slightly
    # different question than before. As bare class attributes they are visible
    # to `hasattr` and read exactly like the `getattr(fluid, "disgas", False)`
    # default that every consumer already applies to this fluid - so nothing
    # that reads them can change its mind.
    #
    # Both are False and structurally so, *even with a gas phase*: this fluid
    # has power-law relperms and exponential b-factors and no dissolution
    # tables at all, so there is no gas dissolved in oil (disgas) and no oil
    # vaporised into gas (vapoil). A three-phase `initSimpleADIFluid` case is a
    # dead-oil, dry-gas case. They are stated rather than left absent because
    # `CPRSolver._weighting` reads `fluid.vapoil` directly, and an attribute
    # that is merely absent is a crash rather than a False.
    disgas = False
    vapoil = False

    #: Gas curves, present only when the caller asked for three phases.
    b_g: Optional[Callable] = None
    kr_g: Optional[Callable] = None
    #: pcOG(sG) = p_gas - p_oil.
    pc_og: Optional[Callable] = None
    #: Power-law exponents and compressibilities, kept so a caller can read
    #: back what it asked for - `blackoilTutorialMexAcceleration` checks them.
    n: Optional[np.ndarray] = None
    c: Optional[np.ndarray] = None

    def b(self, p, rs=None, sat=None, rv=None, oil_present=None) -> list:
        """Shrinkage factors per phase, in MRST's `[water, oil]` order.

        The signature is MRST's `fluid.b(p, rs, sat, rv, oil_present)` and the
        last four arguments are accepted and ignored: they carry the dissolved
        and vaporised ratios, which a fluid with `disgas = vapoil = False` has
        no use for. Accepting them is what lets a caller written against the
        black-oil fluid call this one without knowing which it holds.

        The order is load-bearing rather than conventional. CPR builds one
        weight per phase and multiplies it against a block row of the Jacobian,
        so `b` must be ordered the way the equations are assembled - water
        first, then oil, matching `assemble` and `combine_equations`. Reversed,
        the weights still have the right values, they are applied to the wrong
        equations, and the pressure system comes out subtly wrong rather than
        obviously broken.
        """
        out = [self.b_w(p), self.b_o(p)]
        if self.b_g is not None:
            out.append(self.b_g(p))
        return out[:int(getattr(self, "n_phases_declared", len(out)))]


def init_simple_adi_fluid(
    mu: Optional[Sequence[float]] = None,
    rho: Optional[Sequence[float]] = None,
    n: Sequence[float] = (1.0, 1.0),
    c: Sequence[float] = (0.0, 0.0),
    p_ref: float = 0.0,
    pc_ow: Optional[Callable] = None,
    pc_og: Optional[Callable] = None,
    phases: Optional[str] = None,
    b: Optional[Sequence[float]] = None,
    smin: Optional[Sequence[float]] = None,
    c_r: Optional[float] = None,
    blackoil: bool = True,
) -> ADFluid:
    """Water/oil/gas fluid with b = b0 exp(c (p - p_ref)) and kr = s^n.

    One, two or three phases, decided by the length of the arrays.

    `pc_ow(sW)` is MRST's `fluid.pcOW`: the capillary pressure `p_o - p_w`,
    a decreasing function of water saturation. Omit it and the phases share
    a single pressure, as before. `pc_og(sG)` is `p_g - p_o`.

    The remaining arguments are MRST's, and every one of them defaults to the
    behaviour this function had before they existed:

    `phases` names the phases the positional arrays describe - MRST's
    `'phases','GO'`. Defaulting it to None means `'WOG'[:len(mu)]`, which is
    what the length-based reading already assumed.

    `b` scales each shrinkage factor (MRST's `opt.b`); `smin` gives connate
    and critical saturations, which turn `kr = s^n` into MRST's scaled
    `max(0, min(1, (s-sl)/(su-sl)))^n` and populate `fluid.krPts`; `c_r` is
    rock compressibility, which adds `fluid.cR` and `fluid.pvMultR`; and
    `blackoil=False` builds relative permeabilities only, as in MRST.
    """
    if phases is None and (mu is None or rho is None):
        phases = "WOG"                           # MRST's default phases
    if mu is None and blackoil:
        # MRST: `opt.mu = [1, 1, 1]` with a warning.
        import warnings
        warnings.warn("Defaulted viscosity to unit value (1000 centipoise) for all phases.")
        mu = [1.0, 1.0, 1.0]
    elif mu is None:
        mu = [1.0, 1.0, 1.0]
    if rho is None:
        rho = [1.0, 1.0, 1.0]                    # MRST's default `rho`
    mu = np.asarray(mu, dtype=np.float64).ravel()
    rho = np.asarray(rho, dtype=np.float64).ravel()
    n = np.asarray(n, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()
    if phases is not None:
        # MRST: the phases string sets the phase count, and each option is
        # read per phase as `opt.x(i)`, so longer arrays (such as the default
        # `n = [1 1 1]` for `'phases', 'W'`) are cut to the phases named; a
        # shorter one fails as MATLAB's indexing would.
        n_phases = len(phases)
        if c.size and not np.any(c) and c.size < n_phases:
            c = np.zeros(n_phases)              # MRST's `c = []`: incompressible
        if n.size < n_phases and np.all(n == 1):
            n = np.ones(n_phases)               # MRST's default `n = [1 1 1]`
        for name, value in (("mu", mu), ("rho", rho), ("n", n), ("c", c)):
            if value.size < n_phases:
                raise ValueError(f"{name} has {value.size} entries for the "
                                 f"{n_phases} phases {phases!r}")
        mu, rho, n, c = (a[:n_phases] for a in (mu, rho, n, c))
    sizes = {mu.size, rho.size, n.size, c.size}
    if len(sizes) != 1 or sizes.pop() not in (1, 2, 3):
        raise ValueError(
            f"mu, rho, n and c must all have the same length, 1, 2 or 3; got "
            f"{mu.size}, {rho.size}, {n.size}, {c.size}")
    assert np.all(c >= 0), "compressibilities must be non-negative"
    n_phases = int(mu.size)

    # MRST validates the phase string before it uses it, and each of these
    # mistakes is otherwise silent: an unknown letter would create a phase no
    # consumer looks for, a repeat would overwrite the earlier one's curves,
    # and a length mismatch would pair `rho` with the wrong phase.
    names = (phases if phases is not None else "WOG"[:n_phases]).upper()
    if not set(names) <= set("WOG"):
        raise ValueError(f"phases must be some combination of W, O and G; "
                         f"got {names!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate phases in {names!r}")
    if len(names) != n_phases:
        raise ValueError(f"phases {names!r} names {len(names)} phases but "
                         f"mu, rho, n and c have {n_phases}")

    b0 = (np.ones(n_phases) if b is None
          else np.asarray(b, dtype=np.float64).ravel()[:n_phases])
    # Whether the caller named end points at all, which decides if `krPts` is
    # published. MRST always sets it, but in PyMRST `SaturationProperty` reads
    # `krPts.w` *in preference to* `fluid.swcon` - so an all-zero `krPts` from
    # a caller that never asked for one does not say "connate water is zero",
    # it silently overrides a `swcon` that was set deliberately.
    smin_given = smin is not None
    smin = (np.zeros(n_phases) if smin is None
            else np.asarray(smin, dtype=np.float64).ravel()[:n_phases])
    if smin.sum() >= 0.99:
        raise ValueError("sum of connate/critical saturations must be < 1; "
                         f"got {smin.sum()}")

    # Kept at the caller's length and in the caller's order: the padding below
    # rewrites `mu`/`n`/`c`, and the MRST-named view has to describe the phases
    # that were actually asked for.
    mu0, rho0, n0, c0 = mu.copy(), rho.copy(), n.copy(), c.copy()

    # A single-phase fluid is MRST's `initSimpleADIFluid('phases','W', ...)`,
    # and the one-phase tutorial needs it. The oil slot is filled with the
    # water curves rather than left absent, because every consumer indexes
    # `b`/`mu` by phase and a one-element list would raise on a model that
    # asks for two - while a *duplicated* curve is never reached, since a
    # one-phase model never evaluates phase 1.
    surface_densities = rho.copy()
    if n_phases < 3:
        # Pad the *curves* to three slots so every consumer can index by phase
        # without asking how many there are. Never evaluated: a two-phase model
        # does not ask for phase 2. `rhoS` below keeps the caller's length,
        # because a model reads its phase count from it.
        pad = 3 - n_phases
        mu, n, c = (np.concatenate([a, np.repeat(a[-1], pad)])
                    for a in (mu, n, c))
        rho = np.concatenate([rho, np.repeat(rho[-1], pad)])
        b_pad = np.concatenate([b0, np.repeat(b0[-1], pad)])
    else:
        b_pad = b0

    def make_b(ci: float, scale: float = 1.0) -> Callable:
        # MRST multiplies the exponential by `opt.b(i)`, and uses
        # `constantReciprocalFVF` - a flat 1 - when no compressibility is
        # given. `scale=1` and `ci=0` reproduce both.
        if ci == 0.0:
            return lambda p: p * 0.0 + scale
        return lambda p: scale * ad_exp(ci * (p - p_ref))

    def end_points(i: int, total: np.ndarray):
        """MRST's `[sl, su]`: `su` leaves room for the other phases' connates."""
        sl = float(total[i])
        return sl, 1.0 - float(total.sum()) + sl

    def make_kr(ni: float, sl: float, su: float) -> Callable:
        if sl == 0.0 and su == 1.0:
            return lambda s: s ** ni
        # MRST: max(0, min(1, (s-sl)/(su-sl))).^n. Clamped *before* the power,
        # so a saturation below the connate value gives exactly zero rather
        # than a negative base raised to a fractional exponent (a NaN).
        return lambda s: ad_max(ad_min((s - sl) / (su - sl), 1.0), 0.0) ** ni

    def make_mu(mui: float) -> Callable:
        """MRST's `constantViscosity`: a function of pressure, so that a caller
        written against a deck fluid can call `muW(p)` on this one."""
        return lambda p, *a: p * 0.0 + mui

    # End points come from the *unpadded* `smin`: padding repeats the last
    # connate value, and a repeat would enter `sum(smin)` and shrink `su` for
    # every phase - including the real ones.
    limits = [end_points(i, smin) for i in range(n_phases)]
    limits += [limits[-1]] * (3 - n_phases)

    fluid = ADFluid(
        rhoS=surface_densities,
        mu=mu[:n_phases] if n_phases > 1 else mu,
        b_w=make_b(c[0], b_pad[0]),
        b_o=make_b(c[1], b_pad[1]),
        kr_w=make_kr(n[0], *limits[0]),
        kr_o=make_kr(n[1], *limits[1]),
        pc_ow=pc_ow,
        b_g=make_b(c[2], b_pad[2]) if n_phases == 3 else None,
        kr_g=make_kr(n[2], *limits[2]) if n_phases == 3 else None,
        pc_og=pc_og,
        n=n[:n_phases].copy(),
        c=c[:n_phases].copy(),
    )
    #: How many phases the caller actually asked for. The curves are padded to
    #: three above so the per-phase indexing works, so the count has to be
    #: recorded rather than inferred from their length.
    fluid.n_phases_declared = n_phases

    # -- MRST's named view ----------------------------------------------------
    #
    # `initSimpleADIFluid` sets `rhoWS`, `bW`, `muW`, `krW` and so on by phase
    # *name*, not by position, which is what makes `'phases','GO'` meaningful:
    # the positional arrays describe `names[i]`, whatever letter that is. The
    # slots above stay positional, exactly as they were, so nothing that reads
    # this fluid today sees a different object; these attributes are added
    # beside them. Where the two disagree - a non-canonical `phases` string -
    # the named ones are the MRST answer, and `phase_names` records the order.
    fluid.phase_names = names
    fluid.krPts = {}
    for i, name in enumerate(names):
        sl, su = end_points(i, smin)
        kr = make_kr(n0[i], sl, su)
        setattr(fluid, f"rho{name}S", float(rho0[i]))
        setattr(fluid, f"kr{name}", kr)
        # Shape (1, 4), not (4,). MRST's `[sl, sl, su, 1]` is a *row* - one row
        # per saturation region - and consumers index it `krPts.w[regions, 0]`.
        # A flat four-element vector reads as four regions of one column and
        # raises on that index; this fluid always has exactly one region.
        pts = np.array([[sl, sl, su, 1.0]])
        if smin_given:
            fluid.krPts[name.lower()] = pts
        if blackoil:
            setattr(fluid, f"b{name}", make_b(c0[i], b0[i]))
            setattr(fluid, f"mu{name}", make_mu(mu0[i]))
        # Three-phase oil is reached from both the water and the gas side, and
        # MRST gives each its own handle so a caller can replace one of them.
        if name == "O" and len(names) > 2:
            fluid.krOW = fluid.krOG = kr
            if smin_given:
                fluid.krPts["ow"] = fluid.krPts["og"] = pts

    # MRST's capillary-pressure names. The MRST models test `isfield(fluid,
    # 'pcOW')`, so a curve supplied only under the snake-case slot is silently
    # ignored by them. Set beside the slots, and only when a curve was given:
    # an absent `pcOW` is how MRST says "no capillarity".
    if pc_ow is not None:
        fluid.pcOW = pc_ow
    if pc_og is not None:
        fluid.pcOG = pc_og

    if c_r is not None:
        if np.size(c_r) != 1:
            raise ValueError("rock compressibility must be a single number")
        c_r = float(np.asarray(c_r).ravel()[0])
        if c_r < 0:
            raise ValueError("rock compressibility must be non-negative")
        fluid.cR = c_r
        fluid.pvMultR = lambda p: 1.0 + c_r * (p - p_ref)

    return fluid


if __name__ == "__main__":
    from pymrst.utils.adi import init_variables_adi, value
    from pymrst.utils.units import BARSA

    def test_b_factors():
        """b(p_ref) = 1; slope equals compressibility."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                  n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA],
                                  p_ref=200 * BARSA)
        (p,) = init_variables_adi(np.array([200 * BARSA, 210 * BARSA]))
        bw = f.b_w(p)
        assert np.isclose(value(bw)[0], 1.0)
        assert np.isclose(bw.jac[0].toarray()[0, 0], 1e-6 / BARSA)
        assert np.isclose(value(f.b_o(p))[1],
                          np.exp(1e-4 / BARSA * 10 * BARSA))
        print("ok test_b_factors")

    def test_incompressible_b_is_one():
        """c = 0 gives b identically 1 (also through ADI)."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0])
        (p,) = init_variables_adi(np.array([1e7, 2e7]))
        b = f.b_w(p)
        assert np.allclose(value(b), 1.0)
        assert b.jac[0].count_nonzero() == 0
        print("ok test_incompressible_b_is_one")

    def test_relperms():
        """Power-law relperms of own saturation."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                  n=[2, 3])
        assert np.isclose(f.kr_w(np.array([0.5]))[0], 0.25)
        assert np.isclose(f.kr_o(np.array([0.5]))[0], 0.125)
        print("ok test_relperms")

    def test_three_phases_are_accepted():
        """MRST's `'phases','wog'`, and what `blackoilTutorialMexAcceleration`
        builds. Refusing it meant that example could not be ported."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3, 1e-4],
                                  rho=[1000.0, 700.0, 10.0],
                                  n=[2, 3, 1], c=[0.0, 1e-6, 1e-4])
        assert f.n_phases_declared == 3
        assert np.allclose(f.rhoS, [1000.0, 700.0, 10.0])
        assert f.b_g is not None and f.kr_g is not None
        assert len(f.b(np.array([1.0e7]))) == 3
        print("ok test_three_phases_are_accepted")

    def test_the_gas_curves_are_the_third_entries():
        """A gas relperm built from the oil exponent looks plausible and is a
        different fluid."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3, 1e-4],
                                  rho=[1000.0, 700.0, 10.0],
                                  n=[2, 3, 1], c=[0.0, 0.0, 0.0])
        s = np.array([0.5])
        assert np.allclose(f.kr_w(s), 0.5**2)
        assert np.allclose(f.kr_o(s), 0.5**3)
        assert np.allclose(f.kr_g(s), 0.5**1)
        print("ok test_the_gas_curves_are_the_third_entries")

    def test_a_two_phase_fluid_still_has_two_surface_densities():
        """`rhoS` keeps the caller's length - a model reads its phase count
        from it, and a padded third entry would make every two-phase model
        think it had gas."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0])
        assert f.rhoS.size == 2, f.rhoS
        assert f.n_phases_declared == 2
        assert f.b_g is None and f.kr_g is None
        assert len(f.b(np.array([1.0e7]))) == 2
        print("ok test_a_two_phase_fluid_still_has_two_surface_densities")

    def test_a_single_phase_fluid_is_unchanged():
        f = init_simple_adi_fluid(mu=[1e-3], rho=[1000.0], n=[1], c=[1e-9])
        assert f.n_phases_declared == 1 and f.rhoS.size == 1
        assert len(f.b(np.array([1.0e7]))) == 1
        print("ok test_a_single_phase_fluid_is_unchanged")

    def test_four_phases_are_refused():
        try:
            init_simple_adi_fluid(mu=[1]*4, rho=[1]*4, n=[1]*4, c=[0]*4)
        except ValueError as refused:
            assert "1, 2 or 3" in str(refused)
            print("ok test_four_phases_are_refused")
            return
        raise AssertionError("this fluid has at most three phases")

    # -- MRST options: phases, b, smin, cR, blackoil -------------------------

    def test_phases_names_the_phases_the_arrays_describe():
        """MRST's `'phases','GO'`: position 0 is gas, position 1 is oil."""
        f = init_simple_adi_fluid(mu=[1e-4, 5e-3], rho=[300.0, 500.0],
                                  phases="GO")
        assert f.phase_names == "GO"
        assert f.rhoGS == 300.0 and f.rhoOS == 500.0
        assert not hasattr(f, "rhoWS"), "a GO fluid has no water"
        print("ok test_phases_names_the_phases_the_arrays_describe")

    def test_phases_defaults_to_the_canonical_prefix():
        """Omitting it must keep the length-based reading it replaced."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0])
        assert f.phase_names == "WO"
        assert f.rhoWS == 1000.0 and f.rhoOS == 800.0
        print("ok test_phases_defaults_to_the_canonical_prefix")

    def test_smin_scales_the_relperm_curve():
        """kr = max(0,min(1,(s-sl)/(su-sl)))^n, with su leaving room for the
        other phase's connate saturation."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                  n=[2, 2], smin=[0.2, 0.0])
        # water: sl=0.2, su = 1 - 0.2 + 0.2 = 1.0 -> ((0.6-0.2)/0.8)^2
        assert np.isclose(f.kr_w(np.array([0.6]))[0], 0.25), f.kr_w(
            np.array([0.6]))
        # oil:   sl=0.0, su = 1 - 0.2 + 0.0 = 0.8 -> (0.4/0.8)^2
        assert np.isclose(f.kr_o(np.array([0.4]))[0], 0.25)
        print("ok test_smin_scales_the_relperm_curve")

    def test_smin_clamps_instead_of_going_negative():
        """Below the connate value the curve is exactly zero. Unclamped,
        (s-sl)/(su-sl) is negative and an odd power keeps the sign - a
        negative relative permeability that flows fluid backwards."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                  n=[3, 3], smin=[0.2, 0.0])
        assert f.kr_w(np.array([0.1]))[0] == 0.0
        assert f.kr_w(np.array([2.0]))[0] == 1.0, "and clamps above as well"
        print("ok test_smin_clamps_instead_of_going_negative")

    def test_kr_pts_records_the_end_points():
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                  smin=[0.2, 0.1])
        assert np.allclose(f.krPts["w"], [0.2, 0.2, 0.9, 1.0]), f.krPts
        assert np.allclose(f.krPts["o"], [0.1, 0.1, 0.8, 1.0]), f.krPts
        # One *row* per saturation region: `saturation_property` indexes this
        # as `krPts.w[regions, 0]`, which a flat (4,) vector cannot satisfy.
        assert f.krPts["w"].shape == (1, 4), f.krPts["w"].shape
        print("ok test_kr_pts_records_the_end_points")

    def test_no_smin_publishes_no_kr_pts():
        """The second regression this file caused. `SaturationProperty` reads
        `krPts.w` in preference to `fluid.swcon`, so publishing an all-zero
        `krPts` from a caller that never named end points does not report
        "connate water is zero" - it overrides a `swcon` that was set on
        purpose, and returns 0 where the caller asked for 0.17."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0])
        assert f.krPts == {}, f.krPts
        asked = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                      smin=[0.17, 0.0])
        assert np.isclose(asked.krPts["w"][0, 0], 0.17)
        print("ok test_no_smin_publishes_no_kr_pts")

    def test_kr_pts_is_shaped_the_way_its_consumer_indexes_it():
        """The regression this file caused: `ADFluid` had no `krPts` at all, so
        `get_connate_water` fell through to `swcon`. Adding one that is shaped
        wrong is worse than not adding it - the fall-through disappears and the
        index raises."""
        from pymrst.modules.ad_core.statefunctions.flowprops.virtual.\
            saturation_property import _region_indices
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                  smin=[0.2, 0.1])
        regions = _region_indices(None, np.arange(5), 5)
        assert np.allclose(np.asarray(f.krPts["w"], dtype=float)[regions, 0],
                           0.2)
        print("ok test_kr_pts_is_shaped_the_way_its_consumer_indexes_it")

    def test_b_scales_the_shrinkage_factor():
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                  b=[2.0, 3.0])
        assert np.allclose(f.b_w(np.array([1e7])), 2.0)
        assert np.allclose(f.b_o(np.array([1e7])), 3.0)
        assert np.allclose(f.bW(np.array([1e7])), 2.0)
        print("ok test_b_scales_the_shrinkage_factor")

    def test_c_r_adds_pv_mult_r():
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                  p_ref=100 * BARSA, c_r=1e-5 / BARSA)
        assert np.isclose(f.cR, 1e-5 / BARSA)
        assert np.allclose(f.pvMultR(np.array([110.0 * BARSA])), 1 + 1e-4)
        bare = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0])
        assert not hasattr(bare, "pvMultR"), "absent unless asked for"
        print("ok test_c_r_adds_pv_mult_r")

    def test_viscosity_is_a_function_of_pressure():
        """MRST's `constantViscosity`: `muW(p)`, not a bare number, so code
        written against a deck fluid can call it."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0])
        out = f.muW(np.array([1e7, 2e7]))
        assert np.allclose(out, 1e-3) and out.shape == (2,), out
        assert np.allclose(f.mu, [1e-3, 5e-3]), "the array view is untouched"
        print("ok test_viscosity_is_a_function_of_pressure")

    def test_blackoil_false_builds_only_relperms():
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                  blackoil=False)
        assert hasattr(f, "krW") and not hasattr(f, "bW")
        assert not hasattr(f, "muW")
        print("ok test_blackoil_false_builds_only_relperms")

    def test_three_phase_oil_is_reachable_from_both_sides():
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3, 1e-4],
                                  rho=[1000.0, 700.0, 10.0], n=[2, 3, 1],
                                  c=[0.0, 0.0, 0.0])
        assert f.krOW is f.krO and f.krOG is f.krO
        scaled = init_simple_adi_fluid(mu=[1e-3, 5e-3, 1e-4],
                                       rho=[1000.0, 700.0, 10.0],
                                       n=[2, 3, 1], c=[0.0, 0.0, 0.0],
                                       smin=[0.1, 0.0, 0.0])
        assert np.allclose(scaled.krPts["ow"], scaled.krPts["o"])
        two = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0])
        assert not hasattr(two, "krOW"), "only split when there are 3 phases"
        print("ok test_three_phase_oil_is_reachable_from_both_sides")

    def test_a_bad_phase_string_is_refused():
        for bad, why in [("WOX", "letter"), ("WW", "duplicate"),
                         ("WOG", "length")]:
            try:
                init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1.0, 1.0],
                                      phases=bad)
            except ValueError:
                continue
            raise AssertionError(f"a bad {why} must be refused: {bad}")
        print("ok test_a_bad_phase_string_is_refused")

    def test_connate_saturations_must_leave_room():
        try:
            init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1.0, 1.0],
                                  smin=[0.6, 0.5])
        except ValueError as refused:
            assert "< 1" in str(refused)
            print("ok test_connate_saturations_must_leave_room")
            return
        raise AssertionError("sum(smin) >= 1 leaves no mobile saturation")

    def test_smin_survives_adi():
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                  n=[2, 2], smin=[0.2, 0.0])
        (s,) = init_variables_adi(np.array([0.6]))
        kr = f.kr_w(s)
        assert np.isclose(value(kr)[0], 0.25)
        # d/ds ((s-0.2)/0.8)^2 = 2*0.5/0.8 = 1.25
        assert np.isclose(kr.jac[0].toarray()[0, 0], 1.25)
        print("ok test_smin_survives_adi")

    test_three_phases_are_accepted()
    test_the_gas_curves_are_the_third_entries()
    test_a_two_phase_fluid_still_has_two_surface_densities()
    test_a_single_phase_fluid_is_unchanged()
    test_four_phases_are_refused()
    test_b_factors()
    test_incompressible_b_is_one()
    test_relperms()
    test_phases_names_the_phases_the_arrays_describe()
    test_phases_defaults_to_the_canonical_prefix()
    test_smin_scales_the_relperm_curve()
    test_smin_clamps_instead_of_going_negative()
    test_smin_survives_adi()
    test_kr_pts_records_the_end_points()
    test_no_smin_publishes_no_kr_pts()
    test_kr_pts_is_shaped_the_way_its_consumer_indexes_it()
    test_b_scales_the_shrinkage_factor()
    test_c_r_adds_pv_mult_r()
    test_viscosity_is_a_function_of_pressure()
    test_blackoil_false_builds_only_relperms()
    test_three_phase_oil_is_reachable_from_both_sides()
    test_a_bad_phase_string_is_refused()
    test_connate_saturations_must_leave_room()

    def test_the_phases_string_sets_the_count_as_in_mrst():
        """`initSimpleADIFluid('phases','W','mu',1e-3,'rho',1000,'c',1e-9)`
        takes `n(1)` of the default `[1 1 1]`; a short option fails."""
        fluid = init_simple_adi_fluid(phases="W", mu=1e-3, rho=1000, c=1e-9)
        assert np.size(fluid.rhoS) == 1
        np.testing.assert_allclose(fluid.krW(np.array([0.5])), [0.5])
        fluid = init_simple_adi_fluid(phases="WOG", mu=[1e-3, 2e-3, 3e-4],
                                      rho=[1000, 800, 100])
        np.testing.assert_allclose(fluid.bG(np.array([1e7])), [1.0])
        try:
            init_simple_adi_fluid(phases="WO", mu=[1e-3], rho=[1000, 800])
        except ValueError as refused:
            assert "mu" in str(refused)
        else:
            raise AssertionError("a one-entry mu cannot serve two phases")
        print("ok test_the_phases_string_sets_the_count_as_in_mrst")

    test_the_phases_string_sets_the_count_as_in_mrst()

    def test_mu_and_rho_default_as_in_mrst():
        """`initSimpleADIFluid('phases', 'WO', 'n', [1 1])`: unit viscosity
        (1 Pa s, with a warning) and unit density."""
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fluid = init_simple_adi_fluid(phases="WO", n=[1, 1])
        assert any("1000 centipoise" in str(w.message) for w in caught)
        np.testing.assert_allclose(fluid.rhoS, [1.0, 1.0])
        np.testing.assert_allclose(fluid.muW(np.array([1e5])), [1.0])
        print("ok test_mu_and_rho_default_as_in_mrst")

    test_mu_and_rho_default_as_in_mrst()
    print("\nAll tests passed!")
