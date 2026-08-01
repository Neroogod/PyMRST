"""MRST JFUNC capillary-pressure scaling."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np


def j_function_ratio(rock, porosity_exponent: float, permeability_exponent: float,
                     permeability_direction: str | Sequence[int] = "xyz",
                     pressure_unit: float = 1.0):
    """Return MRST's static J-function ratio ``phi^a / k^b / pressure_unit``."""
    phi = np.asarray(rock.poro, dtype=float)
    perm = np.asarray(rock.perm, dtype=float)
    if perm.ndim == 1:
        perm = perm[:, None]
    if isinstance(permeability_direction, str):
        dirs = ["xyz".index(ch.lower()) for ch in permeability_direction]
    else:
        dirs = [int(d) for d in permeability_direction]
    dirs = [d for d in dirs if d < perm.shape[1]]
    if not dirs:
        raise ValueError("JFUNC permeability direction is outside rock.perm")
    k = np.mean(perm[:, dirs], axis=1)
    return phi**float(porosity_exponent) / k**float(permeability_exponent) / float(pressure_unit)


def _scaled_function(fn, ratio, surface_tension):
    ratio = np.asarray(ratio, dtype=float)
    sigma = float(surface_tension)

    def scaled(s):
        return sigma * ratio * fn(s)

    return scaled


def apply_j_function_scaling(fluid, rock, jfunc, pressure_unit: float = 1.0):
    """Return a copy of ``fluid`` with MRST JFUNC-scaled pc functions."""
    if not jfunc:
        return fluid
    jtype = str(jfunc[0]).strip().lower()
    ratio = j_function_ratio(rock, jfunc[3], jfunc[4], jfunc[5], pressure_unit)
    updates = {}
    if jtype in ("both", "water") and getattr(fluid, "pc_ow", None) is not None:
        updates["pc_ow"] = _scaled_function(fluid.pc_ow, ratio, jfunc[1])
    if jtype in ("both", "gas") and getattr(fluid, "pc_og", None) is not None:
        updates["pc_og"] = _scaled_function(fluid.pc_og, ratio, jfunc[2])
    return replace(fluid, **updates) if updates else fluid


__all__ = ["apply_j_function_scaling", "j_function_ratio"]


if __name__ == "__main__":
    from pymrst.modules.ad_props.props.init_deck_adi_fluid import BlackOilFluid
    from pymrst.params.rock import Rock

    def test_j_function_ratio_and_scaling():
        rock = Rock(perm=np.array([[4.0], [16.0]]), poro=np.array([0.25, 0.5]))
        ratio = j_function_ratio(rock, 1.0, 0.5, "x", 2.0)
        assert np.allclose(ratio, [0.0625, 0.0625])
        f = BlackOilFluid(
            rhoS=np.ones(3),
            b_w=lambda p: p, mu_w=lambda p: p,
            b_o=lambda p, rs, flag: p, mu_o=lambda p, rs, flag: p,
            rs_sat=lambda p: p,
            b_g=lambda p, rv=None, flag=None: p,
            mu_g=lambda p, rv=None, flag=None: p,
            kr_w=lambda s: s, kr_g=lambda s: s,
            kr_ow=lambda s: s, kr_og=lambda s: s,
            pv_mult=lambda p: p, swcon=0.1,
            pc_ow=lambda s: 2.0 + 0.0 * s,
        )
        fs = apply_j_function_scaling(
            f, rock, ["water", 3.0, 0.0, 1.0, 0.5, "x"], 2.0
        )
        assert np.allclose(fs.pc_ow(np.array([0.2, 0.3])), 3.0 * ratio * 2.0)
        print("ok test_j_function_ratio_and_scaling")

    test_j_function_ratio_and_scaling()
