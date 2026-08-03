from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_mechanics.utils.setup_operators_vem import setup_operators_vem
from pymrst.modules.vemmech.vem import ElasticBC, create_augmented_grid, enu2c_2d


FIXTURE = (Path(__file__).resolve().parents[2] / "scripts" / "matlab" /
           "ad_mechanics" / "ref_custom_vem_stabilization.mat")


def test_custom_stabilization_operators_match_mrst():
    ref = loadmat(FIXTURE, squeeze_me=True)
    grid = create_augmented_grid(compute_geometry(cart_grid([1, 1], [1.0, 1.0])))
    nodes = np.arange(grid.nodes.num)
    bc = ElasticBC(
        disp_nodes=nodes,
        disp_mask=np.tile([True, False], (nodes.size, 1)),
        disp_vals=np.zeros((nodes.size, 2)),
    )
    C = enu2c_2d(np.array([1e9]), np.array([0.25]))
    S = np.eye(grid.griddim * grid.nodes.num) * 7.5e8
    op = setup_operators_vem(
        grid, C, 1.0, bc,
        load=lambda x: np.tile([2.0, -3.0], (x.shape[0], 1)), S=S,
    )
    np.testing.assert_allclose(op.A.toarray(), np.atleast_2d(ref["A"]), rtol=2e-13, atol=1e-7)
    np.testing.assert_allclose(op.rhs, np.atleast_1d(ref["rhs"]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(op.strain[:, ~op.is_dir].toarray(), np.atleast_2d(ref["strain"]), rtol=2e-13, atol=1e-13)
    np.testing.assert_allclose(op.stress[:, ~op.is_dir].toarray(), np.atleast_2d(ref["stress"]), rtol=2e-13, atol=1e-7)
    np.testing.assert_array_equal(op.is_dir, np.asarray(ref["isdirdofs"], dtype=bool).reshape(-1))
