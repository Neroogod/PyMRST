from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

from pymrst.modules.ad_mechanics.utils.equations_poro_mechanics import equations_poro_mechanics
from pymrst.utils.adi import init_variables_adi


FIXTURE = (Path(__file__).resolve().parents[2] / "scripts" / "matlab" /
           "ad_mechanics" / "ref_equations_poro_mechanics.mat")


def test_residual_and_both_jacobian_blocks_match_mrst():
    ref = loadmat(FIXTURE, squeeze_me=True, struct_as_record=False)
    m = ref["model"]
    mech = object.__new__(__import__(
        "pymrst.modules.ad_mechanics.mechanics", fromlist=["MechanicModel"]
    ).MechanicModel)
    mech.G = type("Grid", (), {"cells": type("Cells", (), {
        "volumes": np.asarray(m.G.cells.volumes, dtype=float).reshape(-1)
    })()})()
    mech.alpha = np.asarray(m.rock.alpha, dtype=float).reshape(-1)
    mech.operators = type("Operators", (), {
        "A": sp.csr_array(m.operators.A),
        "gradP": sp.csr_array(m.operators.gradP),
        "rhs": np.asarray(m.operators.rhs, dtype=float).reshape(-1),
    })()
    mech.res_scale = 1.0 / (1e6 * np.mean(mech.G.cells.volumes))
    xd, p = init_variables_adi(np.atleast_1d(ref["xd0"]), np.atleast_1d(ref["p0"]))
    eq = equations_poro_mechanics(mech, xd, p)
    np.testing.assert_allclose(eq.val, np.atleast_1d(ref["residual"]), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(eq.jac[0].toarray(), np.atleast_2d(ref["jac_xd"]), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(eq.jac[1].toarray(), np.atleast_2d(ref["jac_p"]), rtol=1e-13, atol=1e-13)
