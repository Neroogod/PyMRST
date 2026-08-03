"""Model 2 of the 10th SPE Comparative Solution Project.

Port of MRST ``examples/datasets/showSPE10.m``.

SPE10 model 2 is a 60 x 220 x 85 box of 1122000 cells whose permeability spans
more than seven orders of magnitude. It exists to be hard: upscaling methods and
linear solvers are judged on it precisely because the contrast defeats naive
approaches.

MRST's script loads the rock and draws four slice plots, of porosity and of
log10 permeability in each direction, converting permeability from m2 to
millidarcy along the way:

    Kx = reshape(log10(rock.perm(:,1)), 60, 220, 85) - log10(milli*darcy);

Subtracting `log10(milli*darcy)` in log space is dividing by it in linear
space, which is the conversion to millidarcy. The port keeps the same
arrangement and makes plotting optional, so the reshaping and the unit handling
are what get checked rather than the pictures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.modules.spe10.get_spe10_rock import (
    SPE10_CART_DIMS,
    SPE10_NUM_CELLS,
    get_spe10_rock,
    spe10_rock_file,
)
from pymrst.utils.units import DARCY, MILLI


@dataclass
class SPE10Fields:
    """Porosity and log10 permeability, shaped as the 60 x 220 x 85 box."""

    poro: np.ndarray            # (60, 220, 85)
    log10_kx: np.ndarray        # log10 of permeability in mD
    log10_ky: np.ndarray
    log10_kz: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.poro.shape


def load_spe10_fields(layers=None) -> SPE10Fields | None:
    """Load SPE10 rock and reshape it into the logical box.

    ``layers`` restricts to selected 1-based layers, which keeps the memory
    manageable: the full field is 1122000 cells across four arrays.

    Returns ``None`` when the dataset is absent.
    """
    rock = get_spe10_rock() if layers is None else get_spe10_rock(layers)
    if rock is None:
        return None

    nx, ny, _ = SPE10_CART_DIMS
    nz = rock.num_cells // (nx * ny)
    shape = (nx, ny, nz)

    # Fortran order: the linear index runs fastest in I, as MRST's `reshape`
    # assumes. C order here would transpose the model without changing its
    # size, and every slice plot would still look like a reservoir.
    poro = rock.poro.reshape(shape, order="F")

    mdarcy = MILLI * DARCY
    logs = []
    for column in range(3):
        k = rock.perm[:, column].reshape(shape, order="F")
        with np.errstate(divide="ignore"):
            logs.append(np.log10(k) - np.log10(mdarcy))
    return SPE10Fields(poro=poro, log10_kx=logs[0], log10_ky=logs[1],
                       log10_kz=logs[2])


def plot_spe10_slices(fields: SPE10Fields | None = None):
    """Slice plots of porosity and log permeability, when a backend exists."""
    fields = load_spe10_fields() if fields is None else fields
    if fields is None:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    figures = []
    for name, data in (("porosity", fields.poro),
                       ("log10 Kx [mD]", fields.log10_kx),
                       ("log10 Ky [mD]", fields.log10_ky),
                       ("log10 Kz [mD]", fields.log10_kz)):
        fig, ax = plt.subplots()
        ax.imshow(data[:, :, 0].T, aspect="equal")
        ax.set_title(name)
        figures.append(fig)
    return figures


if __name__ == "__main__":
    def test_dataset_is_available():
        assert spe10_rock_file() is not None, "spe10_rock.mat not found"
        print("ok test_dataset_is_available")

    def test_fields_reshape_to_the_published_box():
        fields = load_spe10_fields(layers=list(range(1, 6)))
        assert fields is not None, "SPE10 dataset not found"
        assert fields.shape == (60, 220, 5), fields.shape
        assert fields.log10_kx.shape == fields.poro.shape
        print(f"ok test_fields_reshape_to_the_published_box {fields.shape}")

    def test_reshape_uses_fortran_order():
        """MRST's `reshape` is column-major, and C order transposes the model.

        Checked by recomputing one cell's position by hand: logical `(i, j, k)`
        sits at linear index `i + nx*(j + ny*k)`. A C-order reshape puts a
        different value there while keeping the array's shape, so nothing
        downstream complains.
        """
        rock = get_spe10_rock(list(range(1, 4)))
        fields = load_spe10_fields(layers=list(range(1, 4)))
        assert rock is not None and fields is not None
        nx, ny = 60, 220
        for (i, j, k) in ((0, 0, 0), (7, 3, 1), (59, 219, 2), (30, 100, 1)):
            linear = i + nx * (j + ny * k)
            assert np.isclose(fields.poro[i, j, k], rock.poro[linear]), (i, j, k)
        print("ok test_reshape_uses_fortran_order")

    def test_log_permeability_is_in_millidarcy():
        """Subtracting log10(mD) converts m2 to mD in log space.

        SPE10's permeability runs from well under 1 mD to a few thousand, so
        log10 values should land roughly in [-4, 4]. A missing conversion would
        put them near -15, and a doubled one near +15.
        """
        fields = load_spe10_fields(layers=list(range(1, 11)))
        assert fields is not None
        finite = fields.log10_kx[np.isfinite(fields.log10_kx)]
        assert finite.size > 0
        assert -6.0 < finite.min() and finite.max() < 6.0, (
            finite.min(), finite.max())
        # Cross-check one value against a direct conversion.
        rock = get_spe10_rock(list(range(1, 11)))
        kx_md = rock.perm[0, 0] / (MILLI * DARCY)
        assert np.isclose(fields.log10_kx[0, 0, 0], np.log10(kx_md)), (
            fields.log10_kx[0, 0, 0], np.log10(kx_md))
        print(f"ok test_log_permeability_is_in_millidarcy "
              f"({finite.min():.2f} to {finite.max():.2f} log10 mD)")

    def test_vertical_permeability_is_lower_than_horizontal():
        """SPE10's Kz is systematically below Kx, as layered rock should be."""
        fields = load_spe10_fields(layers=list(range(1, 11)))
        assert fields is not None
        kx = fields.log10_kx[np.isfinite(fields.log10_kx)]
        kz = fields.log10_kz[np.isfinite(fields.log10_kz)]
        assert kz.mean() < kx.mean(), (kz.mean(), kx.mean())
        print(f"ok test_vertical_permeability_is_lower_than_horizontal "
              f"(mean log10 Kx {kx.mean():.2f}, Kz {kz.mean():.2f})")

    def test_porosity_is_a_fraction():
        fields = load_spe10_fields(layers=[1, 40, 85])
        assert fields is not None
        assert np.all((fields.poro >= 0.0) & (fields.poro <= 1.0)), (
            fields.poro.min(), fields.poro.max())
        print(f"ok test_porosity_is_a_fraction "
              f"({fields.poro.min():.4f} to {fields.poro.max():.4f})")

    def test_the_two_named_formations_differ():
        """SPE10 is a Tarbert top (layers 1-35) over an Upper Ness base.

        The two formations have visibly different character, so their mean log
        permeabilities should not coincide. Equal means would suggest the layer
        indexing collapsed.
        """
        tarbert = load_spe10_fields(layers=list(range(1, 36)))
        ness = load_spe10_fields(layers=list(range(36, 86)))
        assert tarbert is not None and ness is not None
        a = tarbert.log10_kx[np.isfinite(tarbert.log10_kx)].mean()
        b = ness.log10_kx[np.isfinite(ness.log10_kx)].mean()
        assert not np.isclose(a, b, atol=0.05), (a, b)
        print(f"ok test_the_two_named_formations_differ "
              f"(Tarbert {a:.2f}, Upper Ness {b:.2f} log10 mD)")

    def test_full_model_size():
        assert SPE10_NUM_CELLS == 60 * 220 * 85 == 1122000
        assert SPE10_CART_DIMS == (60, 220, 85)
        print("ok test_full_model_size")

    test_dataset_is_available()
    test_fields_reshape_to_the_published_box()
    test_reshape_uses_fortran_order()
    test_log_permeability_is_in_millidarcy()
    test_vertical_permeability_is_lower_than_horizontal()
    test_porosity_is_a_fraction()
    test_the_two_named_formations_differ()
    test_full_model_size()
    print("\nAll tests passed!")
