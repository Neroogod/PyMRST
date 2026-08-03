"""Grid/rock/state tensor export (F0.4 risk list).

The claim this module makes in its own docstring is the one worth testing:
`float64` is the default **to preserve MRST parity**, and `float32` is opt-in.
A silent single-precision default would not fail anything - it would make the
port disagree with MRST in the sixth digit, which is exactly the kind of drift
the characterization gates exist to catch and the hardest to attribute.

The second is quieter: connectivity is indices, not measurements. If
`faces.neighbors` or `cells.faces` follow the requested float dtype, a grid
exported at `float32` silently loses cell identity past 2^24 cells - a size
real field models reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.ml.tensors import grid_to_tensors, rock_to_tensors, state_to_tensors
from pymrst.params.rock import make_rock

INTEGER_KEYS = ("cells.facePos", "cells.faces", "cells.indexMap",
                "faces.nodePos", "faces.nodes", "faces.neighbors")


def _grid():
    return compute_geometry(cart_grid([3, 2, 2], [30.0, 20.0, 20.0]))


def _rock(grid):
    return make_rock(grid, np.full(grid.cells.num, 1e-13),
                     np.full(grid.cells.num, 0.2))


def test_the_default_dtype_is_float64_for_mrst_parity():
    grid = _grid()
    tensors = grid_to_tensors(grid)
    for key in ("nodes.coords", "cells.volumes", "cells.centroids",
                "faces.areas", "faces.normals", "faces.centroids"):
        assert tensors[key].dtype == np.float64, f"{key} is {tensors[key].dtype}"

    rock_tensors = rock_to_tensors(_rock(grid))
    assert rock_tensors["perm"].dtype == np.float64
    assert rock_tensors["poro"].dtype == np.float64


def test_float64_export_is_exact_not_merely_close():
    """Parity means the same numbers, so the tolerance is zero."""
    grid = _grid()
    tensors = grid_to_tensors(grid)
    np.testing.assert_array_equal(tensors["nodes.coords"], grid.nodes.coords)
    np.testing.assert_array_equal(tensors["cells.volumes"], grid.cells.volumes)
    np.testing.assert_array_equal(tensors["faces.normals"], grid.faces.normals)

    rock = _rock(grid)
    exported = rock_to_tensors(rock)
    np.testing.assert_array_equal(exported["perm"], rock.perm)
    np.testing.assert_array_equal(exported["poro"], rock.poro)


def test_float32_is_opt_in_and_applies_only_to_measurements():
    """Connectivity must not follow the requested float dtype.

    Indices are identity, not measurement: at float32 a cell index above 2^24
    is no longer representable exactly, and real field models are that size.
    """
    grid = _grid()
    tensors = grid_to_tensors(grid, dtype="float32")
    assert tensors["nodes.coords"].dtype == np.float32
    assert tensors["cells.volumes"].dtype == np.float32
    for key in INTEGER_KEYS:
        assert tensors[key].dtype == np.int64, f"{key} became {tensors[key].dtype}"


def test_absent_optional_fields_are_omitted_rather_than_invented():
    """A zero-filled `cells.volumes` would train a model on a fiction."""
    grid = cart_grid([3, 2, 2], [30.0, 20.0, 20.0])       # no compute_geometry
    tensors = grid_to_tensors(grid)
    for key in ("cells.volumes", "cells.centroids", "faces.areas",
                "faces.normals", "faces.centroids"):
        assert key not in tensors, f"{key} was invented for a grid without geometry"
    assert "nodes.coords" in tensors


def test_a_rock_without_net_to_gross_omits_it():
    grid = _grid()
    rock = _rock(grid)
    if hasattr(rock, "ntg"):
        rock.ntg = None
    assert "ntg" not in rock_to_tensors(rock)


def test_state_export_keeps_numeric_fields_and_drops_the_rest():
    grid = _grid()
    state = {"pressure": np.full(grid.cells.num, 200e5),
             "sW": np.full(grid.cells.num, 0.3),
             "wellSol": [{"bhp": 1.0}],
             "name": "step-3",
             "cube": np.zeros((2, 2, 2))}
    tensors = state_to_tensors(state)
    assert set(tensors) == {"pressure", "sW"}
    np.testing.assert_array_equal(tensors["pressure"], state["pressure"])


def test_integer_state_fields_stay_integers():
    tensors = state_to_tensors({"region": np.array([1, 2, 3]),
                                "pressure": np.array([1.0, 2.0, 3.0])},
                               dtype="float32")
    assert tensors["region"].dtype == np.int64
    assert tensors["pressure"].dtype == np.float32


def test_an_unknown_backend_is_refused_by_name():
    grid = _grid()
    with pytest.raises(ValueError, match="jax"):
        grid_to_tensors(grid, backend="jax")


def test_the_numpy_backend_is_the_default_and_accepts_its_aliases():
    grid = _grid()
    for backend in ("numpy", "np", "NumPy"):
        assert isinstance(grid_to_tensors(grid, backend=backend)["nodes.coords"],
                          np.ndarray)


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("torch") is None,
    reason="torch is optional")
def test_the_torch_backend_preserves_values_and_integer_dtypes():
    import torch
    grid = _grid()
    tensors = grid_to_tensors(grid, backend="torch")
    assert tensors["nodes.coords"].dtype is torch.float64
    np.testing.assert_array_equal(tensors["nodes.coords"].numpy(),
                                  grid.nodes.coords)
    for key in INTEGER_KEYS:
        assert not tensors[key].is_floating_point(), f"{key} became floating"

    single = grid_to_tensors(grid, backend="torch", dtype="float32")
    assert single["nodes.coords"].dtype is torch.float32
    for key in INTEGER_KEYS:
        assert not single[key].is_floating_point()
