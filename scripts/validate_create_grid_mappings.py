"""Check `create_grid_mappings` against MRST (scripts/matlab/ref_grid_mappings.mat).

Regenerate the reference from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_grid_mappings_reference.m')"

**Comparison is numbering-independent in the faces**, because MRST and PyMRST
number the faces of the same `cartGrid` differently. Comparing the raw table
would fail on that alone and say nothing about the mapping. Each row is instead
fingerprinted as

    (cell, node, sorted tuple of the face's nodes)

which identifies the same incidence in either numbering, and the two multisets
of fingerprints must be equal. Cell and node numbering *are* asserted to agree
first, so the fingerprint is meaningful; the script fails rather than
proceeding if they do not.

The half-face column is checked structurally: it must partition the rows the
same way in both, since its absolute values are cell-face row indices and
inherit the face numbering.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing import cart_grid, compute_geometry            # noqa: E402
from pymrst.utils.gridtools.create_grid_mappings import (                # noqa: E402
    create_grid_mappings,
)

CASES = {"c322": ([3, 2, 2], [3.0, 2.0, 2.0]), "c33": ([3, 3], [3.0, 3.0])}


def fingerprints(table, face_nodes, node_pos):
    """One hashable key per row, independent of how faces are numbered."""
    signature = [tuple(sorted(face_nodes[node_pos[f]:node_pos[f + 1]]))
                 for f in range(len(node_pos) - 1)]
    return Counter((int(c), int(n), signature[int(f)])
                   for c, n, _, f in table)


def partition(table):
    """Sizes of the half-face groups, as a multiset."""
    return Counter(Counter(table[:, 2].tolist()).values())


def main() -> int:
    reference = loadmat(Path(__file__).parent / "matlab" / "ref_grid_mappings.mat",
                        squeeze_me=True, struct_as_record=False)
    failures = []
    for name, (dims, sizes) in CASES.items():
        ref = reference[name]
        g = compute_geometry(cart_grid(list(dims), list(sizes)))

        aligned = np.allclose(np.asarray(g.nodes.coords, dtype=float),
                              np.asarray(ref.coords, dtype=float))
        print(f"  {'ok ' if aligned else 'FAIL'} {name}: node numbering agrees")
        if not aligned:
            failures.append(f"{name}: node numbering")
            continue

        mine = create_grid_mappings(g)
        theirs = np.asarray(ref.w, dtype=np.int64) - 1        # MATLAB is 1-based

        for label, ok in (
            ("row count", mine.shape == theirs.shape),
            ("incidences", fingerprints(
                mine, np.asarray(g.faces.nodes, dtype=np.int64),
                np.asarray(g.faces.nodePos, dtype=np.int64))
             == fingerprints(
                theirs, np.asarray(ref.faceNodes, dtype=np.int64).ravel() - 1,
                np.asarray(ref.nodePos, dtype=np.int64).ravel() - 1)),
            ("half-face partition", partition(mine) == partition(theirs)),
        ):
            print(f"  {'ok ' if ok else 'FAIL'} {name}: {label}")
            if not ok:
                failures.append(f"{name}: {label}")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("create_grid_mappings matches MRST's createGridMappings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
