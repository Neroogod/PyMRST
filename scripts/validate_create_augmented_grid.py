"""Check `create_augmented_grid` against MRST (scripts/matlab/ref_augmented_grid.mat).

Regenerate from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_augmented_grid_reference.m')"

**Which fields can be compared entry-wise, and which cannot.** Cell and node
numbering agree between the two `cartGrid`s - asserted first, via the
coordinates - so `cells.nodes`, `cells.nodePos`, `nodes.cells` and
`nodes.cellPos` are compared directly. Face numbering does *not* agree, so
`nodes.faces` is compared by its per-node counts rather than its values, and
`faces.edges`/`faces.edgeSign` are not compared at all: both are packed in face
order and would differ for that reason alone.

`edges.nodes` *is* compared directly. Edges are numbered by a canonical sort of
their node pairs, so agreeing node numbering forces agreeing edge numbering -
which makes the edge count a real check rather than a coincidence.

The last section checks the **orphan-node divergence** rather than glossing
over it. MRST's `map2Pos` builds its position arrays with
`find(diff(keys) == 1)`, assuming no key is skipped; a node belonging to no
cell skips one, and the array comes back one entry short. The reference records
what MRST actually produced, and this asserts both that MRST is short and that
the port is not.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing import cart_grid, compute_geometry               # noqa: E402
from pymrst.utils.gridtools.create_augmented_grid import (                  # noqa: E402
    create_augmented_grid,
)

CASES = {"c322": ([3, 2, 2], [3.0, 2.0, 2.0]), "c33": ([3, 3], [3.0, 3.0]),
         "c222": ([2, 2, 2], [2.0, 2.0, 2.0])}


def main() -> int:
    reference = loadmat(
        Path(__file__).parent / "matlab" / "ref_augmented_grid.mat",
        squeeze_me=True, struct_as_record=False)
    failures = []

    for name, (dims, sizes) in CASES.items():
        ref = reference[name]
        g = create_augmented_grid(compute_geometry(cart_grid(dims, sizes)))

        aligned = np.allclose(np.asarray(g.nodes.coords, dtype=float),
                              np.asarray(ref.coords, dtype=float))
        print(f"  {'ok ' if aligned else 'FAIL'} {name}: node numbering agrees")
        if not aligned:
            failures.append(f"{name} numbering")
            continue

        checks = [
            ("cells.nodePos", np.array_equal(
                np.asarray(g.cells.nodePos, np.int64),
                np.asarray(ref.cellNodePos, np.int64).ravel() - 1), ""),
            ("cells.nodes", np.array_equal(
                np.asarray(g.cells.nodes, np.int64),
                np.asarray(ref.cellNodes, np.int64).ravel() - 1), ""),
            ("nodes.cellPos", np.array_equal(
                np.asarray(g.nodes.cellPos, np.int64),
                np.asarray(ref.nodeCellPos, np.int64).ravel() - 1), ""),
            ("nodes.cells", np.array_equal(
                np.asarray(g.nodes.cells, np.int64),
                np.asarray(ref.nodeCells, np.int64).ravel() - 1), ""),
            # Face numbering differs; the counts do not.
            ("nodes.faces (counts)", np.array_equal(
                np.diff(np.asarray(g.nodes.facePos, np.int64)),
                np.diff(np.asarray(ref.nodeFacePos, np.int64).ravel())), ""),
        ]
        if int(ref.numEdges):
            checks += [
                ("edges.num", g.edges.num == int(ref.numEdges),
                 f"{g.edges.num} vs {int(ref.numEdges)}"),
                ("edges.nodes", np.array_equal(
                    np.asarray(g.edges.nodes, np.int64),
                    np.asarray(ref.edgeNodes, np.int64).ravel() - 1), ""),
            ]

        for label, ok, detail in checks:
            print(f"  {'ok ' if ok else 'FAIL'} {name}: {label}"
                  f"{'  ' + detail if detail else ''}")
            if not ok:
                failures.append(f"{name} {label}")

    # The orphan-node divergence.
    orphan = reference["orphan"]
    if int(orphan.ran):
        nodes, got = int(orphan.numNodes), int(orphan.lenCellPos)
        short = got == nodes and got != nodes + 1
        print(f"  {'ok ' if short else 'FAIL'} orphan: MRST returns a "
              f"{got}-entry cellPos for {nodes} nodes (needs {nodes + 1})")
        if not short:
            failures.append("orphan: expected MRST to be short")

        g = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
        g.nodes.coords = np.vstack([np.asarray(g.nodes.coords),
                                    [[99.0, 99.0, 99.0]]])
        g.nodes.num += 1
        a = create_augmented_grid(g)
        ok = (np.asarray(a.nodes.cellPos).size == a.nodes.num + 1
              and np.asarray(a.nodes.facePos).size == a.nodes.num + 1)
        print(f"  {'ok ' if ok else 'FAIL'} orphan: the port returns "
              f"{np.asarray(a.nodes.cellPos).size} for {a.nodes.num} nodes")
        if not ok:
            failures.append("orphan: port length")
    else:
        print(f"  ..  orphan: MRST errored -> {orphan.msg}")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("create_augmented_grid matches MRST's createAugmentedGrid, and is "
          "correct where MRST's position arrays are not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
