"""Build a rock from a full ECLIPSE deck.

Port of MRST ``initEclipseRock.m``.

Where ``grdecl2rock`` reads only a ``GRDECL`` structure, this reads the whole
deck and so can pick up everything a rock may carry: the permeability tensor,
porosity and net-to-gross, directional and fault transmissibility multipliers,
rock compressibility from ``PROPS.ROCK``, region numbers from ``REGIONS``, and
end-point scaling when ``RUNSPEC`` asks for it.

The permeability tensor logic is shared with ``grdecl2rock`` in MRST, which
carries two copies of ``tensorValues``/``getTensor``. The port reuses PyMRST's
single implementation rather than duplicating it, so a fix to the tensor rules
reaches both entry points. That includes ECLIPSE's default that ``PERMX`` alone
implies an isotropic tensor.

Returned shapes follow MRST: ``perm`` has one column when the tensor is
isotropic, three when it is diagonal, and six when it is full and symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pymrst.params.rock.grdecl2rock import _consistent, _get_tensor
from pymrst.params.rock.rock import Rock

_MULTIPLIER_KEYWORDS = ("MULTX", "MULTY", "MULTZ",
                        "MULTX-", "MULTY-", "MULTZ-",
                        "MULTX_", "MULTY_", "MULTZ_")


@dataclass
class EclipseRock:
    """A rock plus the optional deck-derived data MRST attaches to it.

    `Rock` itself carries only `perm`, `poro` and `ntg`, so the extra fields
    live here rather than being forced into it. `as_rock()` returns the plain
    object for callers that want just the core three.
    """

    perm: np.ndarray
    poro: np.ndarray | None = None
    ntg: np.ndarray | None = None
    multipliers: dict[str, np.ndarray] = field(default_factory=dict)
    faultdata: dict[str, Any] | None = None
    cr: np.ndarray | None = None
    pref: np.ndarray | None = None
    regions: dict[str, np.ndarray] = field(default_factory=dict)
    krscale: Any = None
    sw: np.ndarray | None = None

    def as_rock(self) -> Rock:
        return Rock(perm=self.perm, poro=self.poro, ntg=self.ntg)


def init_eclipse_rock(deck: dict) -> EclipseRock:
    """Assemble the rock a deck describes."""
    grid_sec = _section(deck, "GRID")
    if not _consistent(grid_sec):
        raise ValueError("permeability tensor in the deck is not structurally "
                         "consistent; an off-diagonal term was given without "
                         "its diagonal")

    rock = EclipseRock(perm=_get_tensor(grid_sec))

    for name in ("PORO", "NTG"):
        value = _get(grid_sec, name)
        if value is not None:
            setattr(rock, name.lower(),
                    np.asarray(value, dtype=float).ravel())

    _assign_multipliers(rock, grid_sec)
    _assign_fault_multipliers(rock, grid_sec)
    _assign_compressibility(rock, _section(deck, "PROPS"))
    _assign_regions(rock, _section(deck, "REGIONS"))
    _assign_scaling(rock, deck)
    return rock


def _assign_multipliers(rock: EclipseRock, grid_sec) -> None:
    """Directional transmissibility multipliers, keyed by direction.

    ECLIPSE names them `MULTX` and `MULTX-`, the second applying to the face on
    the low side of a cell. MRST also accepts `MULTX_`. The key kept here is
    everything after `MULT`, lowercased, so `MULTX-` becomes `x-`.
    """
    import warnings
    for keyword in _MULTIPLIER_KEYWORDS:
        value = _get(grid_sec, keyword)
        if value is None:
            continue
        mult = np.asarray(value, dtype=float).ravel()
        bad = int(np.count_nonzero(~np.isfinite(mult)))
        if bad:
            warnings.warn(f"{bad} non-finite values in multiplier {keyword!r}",
                          RuntimeWarning, stacklevel=3)
        rock.multipliers[keyword[4:].lower()] = mult


def _assign_fault_multipliers(rock: EclipseRock, grid_sec) -> None:
    faults, multflt = _get(grid_sec, "FAULTS"), _get(grid_sec, "MULTFLT")
    if faults is not None and multflt is not None:
        rock.faultdata = {"faults": faults, "multflt": multflt}


def _assign_compressibility(rock: EclipseRock, props) -> None:
    """`PROPS.ROCK` gives reference pressure and compressibility per region.

    A missing compressibility defaults to zero, which is E100's behaviour and
    means an incompressible rock rather than an error.
    """
    table = _get(props, "ROCK")
    if table is None:
        return
    arr = np.atleast_2d(np.asarray(table, dtype=float))
    rock.pref = arr[:, 0].copy()
    cr = arr[:, 1].copy()
    cr[~np.isfinite(cr)] = 0.0
    rock.cr = cr


def _assign_regions(rock: EclipseRock, regions) -> None:
    """Region numbers, kept only when they actually partition the model.

    MRST records `PVTNUM`, `SATNUM` and `IMBNUM` only when their maximum
    exceeds one, since an all-ones array is the trivial single-region case and
    carrying it would imply a partition that is not there. `SURFNUM` is kept
    whenever present, and pulls `SATNUM` along with it.
    """
    out: dict[str, np.ndarray] = {}
    for key, name in (("PVTNUM", "pvt"), ("SATNUM", "saturation"),
                      ("IMBNUM", "imbibition")):
        value = _get(regions, key)
        if value is not None and np.max(np.asarray(value)) > 1:
            out[name] = np.asarray(value)

    surf = _get(regions, "SURFNUM")
    if surf is not None:
        out["surfactant"] = np.asarray(surf)
        satnum = _get(regions, "SATNUM")
        if "saturation" not in out and satnum is not None:
            out["saturation"] = np.asarray(satnum)

    if out:
        rock.regions = out


def _assign_scaling(rock: EclipseRock, deck) -> None:
    """End-point scaling, active only when `RUNSPEC` declares `ENDSCALE`."""
    runspec = _section(deck, "RUNSPEC")
    if _get(runspec, "ENDSCALE") is None or rock.poro is None:
        return
    from pymrst.modules.deckformat.params.rock.init_relperm_scaling import (
        init_relperm_scaling,
    )
    n_cells = int(np.asarray(rock.poro).shape[0])
    rock.krscale = init_relperm_scaling(deck, n_cells)

    swatinit = _get(_section(deck, "PROPS"), "SWATINIT")
    if swatinit is not None:
        drainage_w = getattr(getattr(rock.krscale, "drainage", None), "w", None)
        sw = np.asarray(swatinit, dtype=float).ravel()
        if drainage_w is not None:
            sw = np.maximum(sw, np.asarray(drainage_w, dtype=float)[:, 0])
        rock.sw = sw


def _section(deck, name):
    if hasattr(deck, name):
        return getattr(deck, name)
    if isinstance(deck, dict):
        for key in (name, name.lower()):
            if key in deck:
                return deck[key]
    return {}


def _get(section, key):
    if section is None:
        return None
    for k in (key, key.lower()):
        if isinstance(section, dict) and k in section:
            return section[k]
        if hasattr(section, k):
            return getattr(section, k)
    return None


__all__ = ["EclipseRock", "init_eclipse_rock"]


if __name__ == "__main__":
    import warnings

    def _deck(**grid):
        base = {"PERMX": np.full(6, 100.0), "PORO": np.full(6, 0.3)}
        base.update(grid)
        return {"GRID": base, "PROPS": {}, "REGIONS": {}, "RUNSPEC": {}}

    def test_permx_alone_gives_an_isotropic_tensor():
        """ECLIPSE's default: PERMX with no PERMY/PERMZ means all three equal.

        The tensor collapses to a single column, which is how MRST signals
        isotropy downstream.
        """
        rock = init_eclipse_rock(_deck())
        assert rock.perm.shape == (6, 1), rock.perm.shape
        assert np.allclose(rock.perm, 100.0)
        assert np.allclose(rock.poro, 0.3)
        print("ok test_permx_alone_gives_an_isotropic_tensor")

    def test_three_distinct_diagonals_give_three_columns():
        rock = init_eclipse_rock(_deck(PERMY=np.full(6, 200.0),
                                       PERMZ=np.full(6, 10.0)))
        assert rock.perm.shape == (6, 3), rock.perm.shape
        assert np.allclose(rock.perm[:, 0], 100.0)
        assert np.allclose(rock.perm[:, 1], 200.0)
        assert np.allclose(rock.perm[:, 2], 10.0)
        print("ok test_three_distinct_diagonals_give_three_columns")

    def test_full_tensor_gives_six_columns():
        rock = init_eclipse_rock(_deck(
            PERMY=np.full(6, 200.0), PERMZ=np.full(6, 10.0),
            PERMXY=np.full(6, 1.0), PERMXZ=np.full(6, 2.0),
            PERMYZ=np.full(6, 3.0)))
        assert rock.perm.shape == (6, 6), rock.perm.shape
        print("ok test_full_tensor_gives_six_columns")

    def test_off_diagonal_without_its_diagonal_is_rejected():
        """`PERMXY` with no `PERMX` is not a consistent tensor."""
        deck = {"GRID": {"PERMXY": np.full(6, 1.0), "PERMY": np.full(6, 1.0),
                         "PERMZ": np.full(6, 1.0), "PORO": np.full(6, 0.3)},
                "PROPS": {}, "REGIONS": {}, "RUNSPEC": {}}
        try:
            init_eclipse_rock(deck)
        except ValueError as exc:
            assert "consistent" in str(exc), exc
        else:
            raise AssertionError("expected an inconsistent tensor to be rejected")
        print("ok test_off_diagonal_without_its_diagonal_is_rejected")

    def test_ntg_and_multipliers_are_carried():
        rock = init_eclipse_rock(_deck(NTG=np.full(6, 0.8),
                                       MULTZ=np.full(6, 0.1),
                                       MULTX=np.full(6, 2.0)))
        assert np.allclose(rock.ntg, 0.8)
        assert set(rock.multipliers) == {"z", "x"}, set(rock.multipliers)
        assert np.allclose(rock.multipliers["z"], 0.1)
        print("ok test_ntg_and_multipliers_are_carried")

    def test_non_finite_multiplier_warns_rather_than_failing():
        mult = np.full(6, 0.5); mult[2] = np.nan
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rock = init_eclipse_rock(_deck(MULTZ=mult))
            assert any("non-finite" in str(x.message) for x in w), [str(x.message) for x in w]
        assert rock.multipliers["z"].shape == (6,)
        print("ok test_non_finite_multiplier_warns_rather_than_failing")

    def test_rock_compressibility_defaults_missing_values_to_zero():
        """A NaN compressibility means incompressible in E100, not an error."""
        deck = _deck()
        deck["PROPS"] = {"ROCK": np.array([[300e5, 1e-6], [400e5, np.nan]])}
        rock = init_eclipse_rock(deck)
        assert np.allclose(rock.pref, [300e5, 400e5]), rock.pref
        assert np.allclose(rock.cr, [1e-6, 0.0]), rock.cr
        print("ok test_rock_compressibility_defaults_missing_values_to_zero")

    def test_trivial_regions_are_not_recorded():
        """An all-ones SATNUM is a single region, so recording it would mislead."""
        deck = _deck()
        deck["REGIONS"] = {"SATNUM": np.ones(6, dtype=int)}
        assert init_eclipse_rock(deck).regions == {}
        deck["REGIONS"] = {"SATNUM": np.array([1, 1, 2, 2, 1, 2])}
        out = init_eclipse_rock(deck)
        assert set(out.regions) == {"saturation"}, set(out.regions)
        print("ok test_trivial_regions_are_not_recorded")

    def test_surfnum_pulls_satnum_along():
        deck = _deck()
        deck["REGIONS"] = {"SURFNUM": np.ones(6, dtype=int),
                           "SATNUM": np.ones(6, dtype=int)}
        out = init_eclipse_rock(deck)
        assert set(out.regions) == {"surfactant", "saturation"}, set(out.regions)
        print("ok test_surfnum_pulls_satnum_along")

    def test_fault_data_needs_both_keywords():
        deck = _deck(FAULTS=[("F1", 1, 1, 1, 1, 1, 1, "X")])
        assert init_eclipse_rock(deck).faultdata is None
        deck = _deck(FAULTS=[("F1", 1, 1, 1, 1, 1, 1, "X")],
                     MULTFLT=[("F1", 0.01)])
        assert init_eclipse_rock(deck).faultdata is not None
        print("ok test_fault_data_needs_both_keywords")

    def test_as_rock_returns_the_plain_three_fields():
        rock = init_eclipse_rock(_deck(NTG=np.full(6, 0.8)))
        plain = rock.as_rock()
        assert isinstance(plain, Rock)
        assert np.allclose(plain.poro, 0.3) and np.allclose(plain.ntg, 0.8)
        print("ok test_as_rock_returns_the_plain_three_fields")

    test_permx_alone_gives_an_isotropic_tensor()
    test_three_distinct_diagonals_give_three_columns()
    test_full_tensor_gives_six_columns()
    test_off_diagonal_without_its_diagonal_is_rejected()
    test_ntg_and_multipliers_are_carried()
    test_non_finite_multiplier_warns_rather_than_failing()
    test_rock_compressibility_defaults_missing_values_to_zero()
    test_trivial_regions_are_not_recorded()
    test_surfnum_pulls_satnum_along()
    test_fault_data_needs_both_keywords()
    test_as_rock_returns_the_plain_three_fields()
    print("\nAll tests passed!")
