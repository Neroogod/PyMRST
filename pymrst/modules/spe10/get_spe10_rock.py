"""Rock properties for SPE10 model 2 (MRST ``getSPE10rock.m``).

The model is a 60 x 220 x 85 box, 1122000 cells, with a permeability field
spanning eight orders of magnitude. It is the standard hard case for upscaling
and for linear solvers.

Three call forms, matching MRST:

    get_spe10_rock()             # the whole dataset
    get_spe10_rock(layers)       # selected layers, K only
    get_spe10_rock(i, j, k)      # a logical box

Indices are **1-based and inclusive**, as MRST writes them, because the layer
numbers in the SPE10 literature are 1-based and quoting them differently would
invite the wrong subset. They are converted internally.

The data ships as ``spe10_rock.mat``, already in SI: permeability in m2 and
porosity dimensionless. No unit conversion happens here, which matters because
applying one twice is invisible in a field that already spans eight decades.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPE10_CART_DIMS = (60, 220, 85)
SPE10_NUM_CELLS = 60 * 220 * 85


@dataclass
class SPE10Rock:
    """Permeability and porosity for a subset of SPE10 model 2."""

    perm: np.ndarray   # (n, 3) in m2
    poro: np.ndarray   # (n,)

    @property
    def num_cells(self) -> int:
        return int(self.poro.shape[0])


def spe10_rock_file(root=None):
    """Locate ``spe10_rock.mat``, or return ``None``. Never downloads."""
    if root is not None:
        candidate = Path(root) / "spe10_rock.mat"
        return candidate if candidate.exists() else None

    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    try:
        base = get_dataset_path("spe10", download=False)
        if base is not None:
            candidate = Path(base) / "spe10_rock.mat"
            if candidate.exists():
                return candidate
    except Exception:
        pass

    for parent in Path(__file__).resolve().parents:
        candidate = (parent.parent
                     / "mrst-2024a-Geomech/examples/data/SPE10/spe10_rock.mat")
        if candidate.exists():
            return candidate
    return None


def get_spe10_rock(*args, data_root=None) -> SPE10Rock | None:
    """Load SPE10 model 2 rock, optionally restricted to layers or a box.

    Returns ``None`` when the dataset is not installed.
    """
    path = spe10_rock_file(data_root)
    if path is None:
        return None

    import scipy.io
    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    rock = data["rock"]
    perm = np.asarray(rock.perm, dtype=float)
    poro = np.asarray(rock.poro, dtype=float).ravel()
    if poro.shape[0] != SPE10_NUM_CELLS:
        raise ValueError(
            f"{path.name} holds {poro.shape[0]} cells, expected {SPE10_NUM_CELLS}")

    index = _subset_indices(*args)
    return SPE10Rock(perm=perm[index, :].copy(), poro=poro[index].copy())


def _subset_indices(*args) -> np.ndarray:
    """Linear 0-based indices for MRST's three call forms."""
    nx, ny, nz = SPE10_CART_DIMS
    i = np.arange(1, nx + 1)
    j = np.arange(1, ny + 1)
    k = np.arange(1, nz + 1)

    if len(args) == 1:
        k = np.atleast_1d(np.asarray(args[0], dtype=np.int64)).ravel()
    elif len(args) == 3:
        i, j, k = (np.atleast_1d(np.asarray(a, dtype=np.int64)).ravel()
                   for a in args)
    elif args:
        raise TypeError(
            "call get_spe10_rock(), get_spe10_rock(layers) or "
            "get_spe10_rock(i, j, k)")

    for name, values, n in (("I", i, nx), ("J", j, ny), ("K", k, nz)):
        if values.size == 0 or np.any((values < 1) | (values > n)):
            raise IndexError(
                f"{name} outside the valid range 1:{n}; SPE10 indices are "
                f"1-based and inclusive")

    # Fortran order, so the linear index runs fastest in I.
    ii, jj, kk = np.meshgrid(i - 1, j - 1, k - 1, indexing="ij")
    return (ii + nx * (jj + ny * kk)).ravel(order="F")


__all__ = ["SPE10Rock", "SPE10_CART_DIMS", "SPE10_NUM_CELLS", "get_spe10_rock",
           "spe10_rock_file"]


if __name__ == "__main__":
    def test_dataset_is_available():
        assert spe10_rock_file() is not None, "spe10_rock.mat not found"
        print("ok test_dataset_is_available")

    def test_full_dataset_has_the_published_size():
        rock = get_spe10_rock()
        assert rock is not None, "SPE10 dataset not found"
        assert rock.num_cells == SPE10_NUM_CELLS, rock.num_cells
        assert rock.perm.shape == (SPE10_NUM_CELLS, 3), rock.perm.shape
        print(f"ok test_full_dataset_has_the_published_size "
              f"({rock.num_cells} cells)")

    def test_permeability_spans_eight_orders_of_magnitude():
        """SPE10's difficulty is its contrast, so the range is worth asserting.

        A field that has been unit-converted twice, or read in the wrong order,
        usually loses that spread while still looking like permeability.
        """
        rock = get_spe10_rock()
        assert rock is not None
        kx = rock.perm[:, 0]
        assert kx.min() > 0.0, kx.min()
        decades = np.log10(kx.max() / kx.min())
        assert decades > 7.0, decades
        print(f"ok test_permeability_spans_eight_orders_of_magnitude "
              f"({decades:.1f} decades)")

    def test_values_are_already_si():
        """The file ships in m2 and needs no conversion.

        1 mD is about 9.87e-16 m2, so a field in m2 sits near 1e-16 to 1e-11.
        Values near 1 to 1e4 would mean millidarcy was never converted; values
        near 1e-31 would mean it was converted twice.
        """
        rock = get_spe10_rock(1)
        assert rock is not None
        kx = rock.perm[:, 0]
        assert 1e-20 < kx.max() < 1e-9, kx.max()
        assert np.all((rock.poro >= 0.0) & (rock.poro <= 1.0)), (
            rock.poro.min(), rock.poro.max())
        print(f"ok test_values_are_already_si "
              f"(kx max {kx.max():.3e} m2 = {kx.max() / 9.869233e-16:.0f} mD)")

    def test_a_single_layer_is_one_slice():
        rock = get_spe10_rock(1)
        assert rock is not None
        assert rock.num_cells == 60 * 220, rock.num_cells
        print("ok test_a_single_layer_is_one_slice")

    def test_layer_selection_matches_a_direct_slice():
        """Layer 36 through the subset must equal layer 36 sliced by hand.

        The linear index runs fastest in I, so a C-order mistake here returns
        an array of the right length holding entirely different cells.
        """
        full = get_spe10_rock()
        assert full is not None
        layer = get_spe10_rock(36)
        start = (36 - 1) * 60 * 220
        expect = full.poro[start:start + 60 * 220]
        assert np.allclose(layer.poro, expect), "layer slice does not match"
        assert np.allclose(layer.perm, full.perm[start:start + 60 * 220, :])
        print("ok test_layer_selection_matches_a_direct_slice")

    def test_a_logical_box_selects_its_cells():
        rock = get_spe10_rock([1, 2, 3], [10, 11], [5])
        assert rock is not None
        assert rock.num_cells == 3 * 2 * 1, rock.num_cells
        print("ok test_a_logical_box_selects_its_cells")

    def test_out_of_range_indices_are_rejected():
        """1-based inclusive, so 0 and 86 are both outside 1:85."""
        for bad in (0, 86):
            try:
                get_spe10_rock(bad)
            except IndexError as exc:
                assert "1-based" in str(exc), exc
            else:
                raise AssertionError(f"expected layer {bad} to be rejected")
        try:
            get_spe10_rock(1, 2)
        except TypeError:
            pass
        else:
            raise AssertionError("expected two positional arguments to be rejected")
        print("ok test_out_of_range_indices_are_rejected")

    test_dataset_is_available()
    test_full_dataset_has_the_published_size()
    test_permeability_spans_eight_orders_of_magnitude()
    test_values_are_already_si()
    test_a_single_layer_is_one_slice()
    test_layer_selection_matches_a_direct_slice()
    test_a_logical_box_selects_its_cells()
    test_out_of_range_indices_are_rejected()
    print("\nAll tests passed!")
