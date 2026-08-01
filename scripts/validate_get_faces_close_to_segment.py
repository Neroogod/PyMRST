"""Check `get_faces_close_to_segment` against MRST (scripts/matlab/ref_faces_close.mat).

Regenerate the reference from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_faces_close_reference.m')"

Faces are compared by **centroid**, since MRST and PyMRST number the faces of
the same `cartGrid` differently. The sets must match exactly - this is a
deterministic geometric predicate, not a heuristic with room to differ, so
"a superset is fine" would be the wrong standard here.

Cases cover the parts most likely to drift: the `fac` slack at both a smaller
and a larger value than the default, the SVD line fit for a polyline, the
canonical-axis projection, and a segment far outside the grid that must come
back empty.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing import cart_grid, compute_geometry               # noqa: E402
from pymrst.utils.gridtools.add_bounding_box_fields import (                # noqa: E402
    add_bounding_box_fields,
)
from pymrst.utils.gridtools.get_faces_close_to_segment import (             # noqa: E402
    get_faces_close_to_segment,
)

DIAG = np.array([[0.3, 0.4, 0.5], [5.7, 5.6, 3.5]])
AXIS_X = np.array([[0.5, 0.5, 0.5], [5.5, 0.5, 0.5]])
AXIS_Y = np.array([[2.5, 0.2, 1.5], [2.5, 5.8, 1.5]])
POLY = np.array([[0.5, 3.0, 2.0], [3.0, 4.2, 2.0], [5.5, 3.0, 2.0]])


def sorted_rows(a, width):
    """Rows in a canonical order, so two orderings of the same set compare."""
    a = np.asarray(a, dtype=float).reshape(-1, width) if np.asarray(a).size \
        else np.zeros((0, width))
    return a[np.lexsort(tuple(a[:, d] for d in range(width)))]


def main() -> int:
    reference = loadmat(Path(__file__).parent / "matlab" / "ref_faces_close.mat",
                        squeeze_me=True, struct_as_record=False)
    g = add_bounding_box_fields(
        compute_geometry(cart_grid([6, 6, 4], [6.0, 6.0, 4.0])))
    g2 = add_bounding_box_fields(
        compute_geometry(cart_grid([6, 6], [6.0, 6.0])))
    centroids = np.asarray(g.faces.centroids, dtype=float)
    centroids2 = np.asarray(g2.faces.centroids, dtype=float)

    def near(seg, **kw):
        return centroids[get_faces_close_to_segment(g, seg, **kw)]

    cases = [
        ("diag", near(DIAG)),
        ("axis_x", near(AXIS_X)),
        ("axis_y", near(AXIS_Y)),
        ("poly", near(POLY)),
        ("fac_half", near(AXIS_X, fac=0.5)),
        ("fac_two", near(AXIS_X, fac=2.0)),
        ("projz", near(np.array([[1.5, 1.5, 0.2], [4.5, 4.5, 0.4]]),
                       proj_vec=[0.0, 0.0, 1.0])),
        ("far", near(np.array([[100.0, 100.0, 100.0], [200.0, 200.0, 200.0]]))),
        ("twod", centroids2[get_faces_close_to_segment(
            g2, np.array([[0.5, 0.5], [5.5, 5.5]]))]),
    ]

    failures = []
    for name, mine in cases:
        width = 2 if name == "twod" else 3
        a = sorted_rows(mine, width)
        b = sorted_rows(reference[name], width)
        ok = a.shape == b.shape and (a.size == 0 or np.allclose(a, b))
        note = " (empty, as expected)" if b.shape[0] == 0 else ""
        print(f"  {'ok ' if ok else 'FAIL'} {name}: {a.shape[0]} vs "
              f"{b.shape[0]} faces{note}")
        if not ok:
            failures.append(name)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("get_faces_close_to_segment matches MRST's getFacesCloseToSegment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
