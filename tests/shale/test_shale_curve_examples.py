import numpy as np
import pytest

from pymrst.modules.shale import gangi_plot, langmuir_plot


def test_gangi_plot_reproduces_mrst_reference_curve_without_renderer():
    result = gangi_plot(plot=False)
    expected = (1 - ((38e6 - .5 * result.pressure) / 180e6) ** .5) ** 3
    assert result.pressure.shape == result.multiplier.shape == (50,)
    assert result.pressure[[0, -1]].tolist() == [0, 300e5]
    assert np.allclose(result.multiplier, expected)
    assert result.figure is None


def test_langmuir_plot_reproduces_mrst_reference_curve_without_renderer():
    result = langmuir_plot(plot=False)
    expected = result.pressure * 3 / (1.077e7 + result.pressure)
    assert result.pressure.shape == result.sorbed_density.shape == (100,)
    assert result.sorbed_density[0] == 0
    assert np.allclose(result.sorbed_density, expected)
    assert result.figure is None


@pytest.mark.parametrize("function", [gangi_plot, langmuir_plot])
def test_curve_examples_reject_degenerate_sampling(function):
    with pytest.raises(ValueError, match="at least two"):
        function(samples=1)
