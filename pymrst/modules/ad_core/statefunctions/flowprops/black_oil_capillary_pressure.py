"""Black-oil capillary pressure with J-function scaling.

Ports MRST `ad-core/statefunctions/flowprops/BlackOilCapillaryPressure.m`.
PyMRST already had a `CapillaryPressure` state function that evaluates the
fluid's single `pc` curve. This adds the two things that class does not have,
and both are physics rather than plumbing.

**Per-pair curves and their signs.** A three-phase model carries `pcOW` and
`pcOG` separately, and the water entry is stored *negated* - MRST's
`pc{water} = -pcow`. That sign is not a convention detail: capillary pressure
is defined as the non-wetting minus the wetting phase pressure, so water, being
wetting, sits below the reference. Getting it backwards moves the contact the
wrong way and still produces a plausible saturation profile.

**Leverett J-function scaling.** A single measured `pc` curve belongs to the
rock sample it came from. Applying it unscaled across a heterogeneous field
gives every cell the same entry pressure regardless of its permeability, which
is wrong in the direction that matters: tight rock holds water far more
strongly than good rock. The J-function restores that dependence,

    pc_cell = sigma * (phi^a / k^b) * pc_table(s)

with `a`, `b` and the permeability direction configurable. Classic Leverett is
`a = b = 0.5`, giving `sqrt(phi/k)`; MRST leaves the exponents free because
published correlations differ, and this port does the same rather than
hard-coding one.

Scaling is off unless surface tension is set for a phase pair, so a model that
never asks for it behaves exactly as before.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE, StateFunction,
)

#: Which permeability column each direction letter selects.
_DIRECTIONS = {"x": 0, "y": 1, "z": 2}


class BlackOilCapillaryPressure(StateFunction):
    """Per-phase capillary offsets, optionally J-function scaled."""

    def __init__(self, model: Any = None, label: str = "p_c"):
        super().__init__(model, label=label)
        self.depends_on("s", STATE)
        self.surface_tension_ow: Optional[float] = None
        self.surface_tension_og: Optional[float] = None
        self.porosity_exponent: float = 0.5
        self.permeability_exponent: float = 0.5
        self.permeability_direction: Sequence[int] = (0, 1, 2)
        self.pressure_unit: float = 1.0
        self.cell_subset: Optional[Any] = None

    # -- what the fluid offers ---------------------------------------------

    @staticmethod
    def pc_present(model: Any) -> bool:
        """Whether the fluid carries any capillary curve at all.

        Checked before the machinery runs: a model with no `pcOW`, `pcOG` or
        `pcWG` has zero capillary offsets, and evaluating anything would be
        inventing a curve.
        """
        fluid = getattr(model, "raw_fluid", None) or getattr(model, "fluid", None)
        return any(_curve(fluid, name) is not None
                   for name in ("pcOW", "pcOG", "pcWG"))

    pcPresent = pc_present

    # -- J-function configuration ------------------------------------------

    def set_surface_tension(self, value: float, phase_pair: str
                            ) -> "BlackOilCapillaryPressure":
        """Set the interfacial tension for `'ow'` or `'og'`.

        Setting it is what *enables* scaling for that pair, so a model that
        never calls this keeps the unscaled curve.
        """
        pair = str(phase_pair).lower()
        if pair == "ow":
            self.surface_tension_ow = float(value)
        elif pair == "og":
            self.surface_tension_og = float(value)
        else:
            raise ValueError(
                f"unsupported phase pair {phase_pair!r}; expected 'ow' or 'og'")
        return self

    setSurfaceTension = set_surface_tension

    def get_surface_tension(self, phase_pair: str) -> Optional[float]:
        pair = str(phase_pair).lower()
        if pair == "ow":
            return self.surface_tension_ow
        if pair == "og":
            return self.surface_tension_og
        raise ValueError(
            f"unsupported phase pair {phase_pair!r}; expected 'ow' or 'og'")

    getSurfaceTension = get_surface_tension

    def has_j_function_scaler(self, phase_pair: str) -> bool:
        """Scaling applies only when a surface tension was supplied."""
        return self.get_surface_tension(phase_pair) is not None

    hasJFunctionScaler = has_j_function_scaler

    def set_j_function_constants(self, porosity_exponent: float,
                                 permeability_exponent: float,
                                 permeability_direction="xyz",
                                 pressure_unit: Optional[float] = None
                                 ) -> "BlackOilCapillaryPressure":
        """Set the exponents and the permeability direction.

        `permeability_direction` may be a string like `"xyz"` or `"z"`, or
        column indices. A string is the readable form and the one MRST takes;
        an unknown letter is refused rather than silently dropped, because a
        typo would quietly change which permeability the scaling reads.
        """
        if pressure_unit is not None:
            self.pressure_unit = float(pressure_unit)
        self.porosity_exponent = float(porosity_exponent)
        self.permeability_exponent = float(permeability_exponent)

        if isinstance(permeability_direction, str):
            columns = []
            for letter in permeability_direction.lower():
                if letter not in _DIRECTIONS:
                    raise ValueError(
                        f"unknown permeability direction {letter!r}; expected "
                        f"some of 'xyz'")
                columns.append(_DIRECTIONS[letter])
            self.permeability_direction = tuple(columns)
        else:
            self.permeability_direction = tuple(
                int(c) for c in np.atleast_1d(permeability_direction))
        return self

    setJFunctionConstants = set_j_function_constants

    def get_j_function_static_ratio(self, model: Any) -> np.ndarray:
        """`phi^a / k^b`, per cell, divided by the table's pressure unit.

        The permeability is averaged over the selected directions *before* the
        exponent is applied - averaging afterwards would give a different
        number, and the mean of a power is not the power of a mean.
        """
        rock = model.rock
        poro = np.asarray(rock.poro, dtype=float).ravel()
        perm = np.atleast_2d(np.asarray(rock.perm, dtype=float))
        if perm.shape[0] == 1 and poro.size > 1:
            perm = np.repeat(perm, poro.size, axis=0)

        columns = [c for c in self.permeability_direction if c < perm.shape[1]]
        if not columns:
            raise ValueError(
                f"permeability has {perm.shape[1]} column(s); directions "
                f"{self.permeability_direction} select none of them")

        if self.cell_subset is not None:
            poro = poro[self.cell_subset]
            perm = perm[self.cell_subset]

        k = perm[:, columns].sum(axis=1)/len(columns)
        ratio = (poro**self.porosity_exponent)/(k**self.permeability_exponent)
        return ratio/self.pressure_unit

    getJFunctionStaticRatio = get_j_function_static_ratio

    def subset(self, cell_subset) -> "BlackOilCapillaryPressure":
        out = super().subset(cell_subset)
        out.cell_subset = cell_subset
        return out

    # -- evaluation ---------------------------------------------------------

    def evaluate_on_domain(self, model: Any, state: Any):
        """One capillary offset per active phase, zero where there is none.

        The water entry is negated. Capillary pressure is the non-wetting minus
        the wetting phase pressure, and water is the wetting phase, so its
        offset from the reference is negative.
        """
        fluid = getattr(model, "raw_fluid", None) or model.fluid
        names = list(model.get_phase_names())
        offsets: list[Any] = [None]*len(names)

        ratio = None
        if self.has_j_function_scaler("ow") or self.has_j_function_scaler("og"):
            ratio = self.get_j_function_static_ratio(model)

        pc_ow = _curve(fluid, "pcOW")
        if "W" in names and "O" in names and pc_ow is not None:
            value = pc_ow(model.get_prop(state, "sW"))
            if self.has_j_function_scaler("ow"):
                value = self.surface_tension_ow*ratio*value
            offsets[names.index("W")] = -value

        pc_og = _curve(fluid, "pcOG")
        if "G" in names and "O" in names and pc_og is not None:
            value = pc_og(model.get_prop(state, "sG"))
            if self.has_j_function_scaler("og"):
                value = self.surface_tension_og*ratio*value
            offsets[names.index("G")] = value

        pc_wg = _curve(fluid, "pcWG")
        if "O" not in names and "G" in names and pc_wg is not None:
            offsets[names.index("G")] = pc_wg(model.get_prop(state, "sG"))

        present = [o for o in offsets if o is not None]
        if not present:
            return offsets
        n = max(np.size(_value(o)) for o in present)
        return [np.zeros(n) if o is None else o for o in offsets]


def _curve(fluid: Any, name: str):
    """A capillary curve off a fluid, under either spelling."""
    if fluid is None:
        return None
    snake = "pc_" + name[2:].lower()
    return getattr(fluid, name, None) or getattr(fluid, snake, None)


def _value(x):
    return getattr(x, "val", x)


__all__ = ["BlackOilCapillaryPressure"]


if __name__ == "__main__":
    from types import SimpleNamespace

    class _Fluid:
        @staticmethod
        def pcOW(sw):
            return 1.0e5*(1.0 - np.asarray(sw))

    def _model(poro=0.2, perm=1e-13, phases=("water", "oil")):
        n = 3
        return SimpleNamespace(
            fluid=SimpleNamespace(phases=list(phases)),
            raw_fluid=_Fluid,
            rock=SimpleNamespace(poro=np.full(n, poro),
                                 perm=np.full((n, 3), perm)),
            get_phase_names=lambda: ["W", "O"],
            get_prop=lambda state, name: state[name],
        )

    def test_water_offset_is_negative():
        prop = BlackOilCapillaryPressure()
        out = prop.evaluate_on_domain(_model(), {"sW": np.full(3, 0.5)})
        assert np.all(np.asarray(out[0]) < 0), "water is the wetting phase"
        print("ok test_water_offset_is_negative")

    def test_leverett_ratio_is_sqrt_phi_over_k():
        prop = BlackOilCapillaryPressure()
        prop.set_j_function_constants(0.5, 0.5, "xyz")
        ratio = prop.get_j_function_static_ratio(_model(0.25, 1e-12))
        assert np.allclose(ratio, np.sqrt(0.25/1e-12))
        print("ok test_leverett_ratio_is_sqrt_phi_over_k")

    def test_scaling_is_off_without_surface_tension():
        prop = BlackOilCapillaryPressure()
        assert not prop.has_j_function_scaler("ow")
        print("ok test_scaling_is_off_without_surface_tension")

    test_water_offset_is_negative()
    test_leverett_ratio_is_sqrt_phi_over_k()
    test_scaling_is_off_without_surface_tension()
    print("\nAll tests passed!")
