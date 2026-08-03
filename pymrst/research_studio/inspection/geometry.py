"""Twisted cells and face orientation (M2.11).

The two remaining items of M2.11's required list, and neither is a matter of
taste - both have a named consequence in the discretization.

**Twisted cells.** `compute_trans` forms the half-face transmissibility from
`c . (K n)`, where `c` runs from the cell centroid to the face centroid and `n`
is the outward normal. When that product is negative the cell centroid lies on
the wrong side of its own face - the cell is twisted or non-convex there - and
TPFA is no longer monotone across it. MRST's response, faithfully ported, is
`fix_negative=True`: the value is **replaced by its absolute value** and a
warning goes to stderr. Measured on a deliberately spiked corner-point cell:
2 of 110 half-transmissibilities negative, the returned array exactly `abs()`
of the raw one. So by the time a user sees results the defect has been
repaired, the warning is somewhere in a long log, and the grid looks fine.
This reports it against a cell index instead.

The geometric test is that product with `K = I`, which is why it belongs to
the grid rather than to the rock: a twisted cell is twisted whatever
permeability is later assigned to it.

**Face orientation.** PyMRST's normals point from `neighbors[:, 0]` to
`neighbors[:, 1]`, and every flux sign in the code depends on it. A reversed
normal does not crash anything - it inverts the flow direction through one
face, which reads downstream as a plausible saturation pattern that is wrong.

**Thresholds are set so a healthy real grid is silent.** Norne mini's most
skewed half-face sits at `cos = 0.489` (about 61 degrees), which is legitimate
corner-point geometry, so warning anywhere near it would report dozens of
findings on a correct grid and teach the reader to ignore the check - the
mistake the transmissibility check already had to correct. The error condition
is the sign change, which is exact and needs no threshold; the warning is
reserved for `cos < 0.2`, near-degenerate by any standard and comfortably
clear of anything real geometry produces.

Read-only, per the plan's rule: nothing here modifies the grid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.research_studio.inspection.grid_report import Finding, Severity

# Below this cosine the face is so oblique to the centroid-to-face vector that
# the two-point approximation is barely meaningful. Set clear of Norne mini's
# worst legitimate value (0.489) on purpose.
SKEW_COSINE = 0.2


def _orphan_half_faces(grid) -> np.ndarray:
    """Half-faces whose face does not list the cell that owns it.

    `compute_trans` decides the outward direction with
    `neighbors[face, 0] == cell`, taking the `else` branch to mean "cell is
    neighbour 1". When the face lists neither, that branch is still taken and
    the normal is silently flipped - so a grid whose `cells.faces` and
    `faces.neighbors` disagree yields transmissibilities with inverted signs
    and no complaint. That is a broken grid rather than a twisted one, and
    reporting it as a twist would send the reader looking at geometry that is
    fine.
    """
    cell_no = grid.cell_no()
    cell_faces = grid.cells.faces
    neighbors = np.asarray(grid.faces.neighbors)[cell_faces]
    return np.flatnonzero((neighbors[:, 0] != cell_no)
                          & (neighbors[:, 1] != cell_no))


def _half_face_projection(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(cell of each half-face, c . n_outward, cos between c and n_outward)`.

    This is `compute_trans` with the permeability removed, deliberately: the
    same quantity, so a finding here predicts the negative transmissibility
    rather than merely resembling it.
    """
    cell_no = grid.cell_no()
    cell_faces = grid.cells.faces
    c = grid.faces.centroids[cell_faces] - grid.cells.centroids[cell_no]
    sign = np.where(grid.faces.neighbors[cell_faces, 0] == cell_no, 1.0, -1.0)
    outward = sign[:, None]*np.asarray(grid.faces.normals)[cell_faces]

    projection = np.sum(c*outward, axis=1)
    # `normals` are area-weighted in MRST's convention, so the norm of the
    # outward normal is the face area; dividing by it and by |c| gives a cosine.
    scale = np.linalg.norm(c, axis=1)*np.linalg.norm(outward, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.where(scale > 0.0, projection/scale, 0.0)
    return cell_no, projection, cosine


def twist_findings(grid: Any) -> list[Finding]:
    """Cells whose centroid falls on the wrong side of one of their faces."""
    if getattr(grid.cells, "centroids", None) is None:
        return [Finding(Severity.WARNING, "geometry",
                        "grid has no centroids; run compute_geometry")]
    cell_no, projection, cosine = _half_face_projection(grid)
    out: list[Finding] = []

    # Checked first: on an inconsistent grid the projections below are computed
    # against flipped normals, so reporting them as twists would be wrong.
    orphans = _orphan_half_faces(grid)
    if orphans.size:
        cells = sorted({int(c) for c in cell_no[orphans]})
        return [Finding(
            Severity.ERROR, "geometry",
            f"{orphans.size} half-face(s) across {len(cells)} cell(s) reference "
            f"a face that does not list them as a neighbour; cells.faces and "
            f"faces.neighbors disagree, and compute_trans reads this as "
            f"'neighbour 1' and flips the normal. Geometry checks are skipped "
            f"because their signs cannot be trusted on this grid",
            cells=tuple(cells))]

    inverted = np.flatnonzero(projection <= 0.0)
    if inverted.size:
        cells = sorted({int(c) for c in cell_no[inverted]})
        out.append(Finding(
            Severity.ERROR, "geometry",
            f"{inverted.size} half-face(s) across {len(cells)} cell(s) have the "
            f"cell centroid on the wrong side of the face; these produce "
            f"negative transmissibilities, which compute_trans replaces with "
            f"their absolute value by default",
            cells=tuple(cells)))

    skewed = np.flatnonzero((projection > 0.0) & (cosine < SKEW_COSINE))
    if skewed.size:
        cells = sorted({int(c) for c in cell_no[skewed]})
        out.append(Finding(
            Severity.WARNING, "geometry",
            f"{skewed.size} half-face(s) across {len(cells)} cell(s) are more "
            f"than {np.degrees(np.arccos(SKEW_COSINE)):.0f} degrees oblique "
            f"(worst cos {cosine[skewed].min():.3f}); two-point flux is a poor "
            f"approximation here",
            cells=tuple(cells)))
    return out


def orientation_findings(grid: Any) -> list[Finding]:
    """Interior faces whose normal does not point from neighbour 0 to 1."""
    centroids = getattr(grid.cells, "centroids", None)
    normals = getattr(grid.faces, "normals", None)
    if centroids is None or normals is None:
        return []

    neighbors = np.asarray(grid.faces.neighbors)
    interior = np.flatnonzero((neighbors[:, 0] >= 0) & (neighbors[:, 1] >= 0))
    if interior.size == 0:
        return []

    separation = (np.asarray(centroids)[neighbors[interior, 1]]
                  - np.asarray(centroids)[neighbors[interior, 0]])
    aligned = np.sum(separation*np.asarray(normals)[interior], axis=1)
    reversed_ = interior[aligned <= 0.0]
    if reversed_.size == 0:
        return []
    return [Finding(
        Severity.ERROR, "faces",
        f"{reversed_.size} interior face normal(s) do not point from "
        f"neighbour 0 to neighbour 1; every flux sign in the discretization "
        f"assumes they do, so flow through these faces runs backwards",
        faces=tuple(int(f) for f in reversed_))]


__all__ = ["SKEW_COSINE", "orientation_findings", "twist_findings"]
