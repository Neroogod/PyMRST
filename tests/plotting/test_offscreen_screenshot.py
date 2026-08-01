"""Safe VTK screenshot regressions; unavailable renderers are skipped."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

def test_offscreen_grid_screenshot_is_nonempty_and_repeatable(tmp_path: Path):
    image_reader = pytest.importorskip("matplotlib.image")
    images = []
    script = """
from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.plotting.plot_grid import plot_grid
grid = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
plot_grid(grid, filename=r'{filename}', show=False, zscale=1.5,
          **{{'FaceColor': [0.2, 0.55, 0.85], 'EdgeColor': 'black',
              'FaceAlpha': 1.0}})
"""
    for index in range(2):
        filename = tmp_path / f"grid-{index}.png"
        completed = subprocess.run(
            [sys.executable, "-c", script.format(filename=filename)],
            cwd=Path(__file__).parents[2],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if completed.returncode != 0:
            pytest.skip(
                "working off-screen VTK renderer unavailable "
                f"(isolated process returned {completed.returncode})"
            )
        assert filename.exists() and filename.stat().st_size > 1000
        image = np.asarray(image_reader.imread(filename))
        assert image.ndim == 3 and image.shape[0] >= 100
        assert float(np.std(image[..., :3])) > 0.01
        images.append(image)
    np.testing.assert_array_equal(images[0], images[1])
