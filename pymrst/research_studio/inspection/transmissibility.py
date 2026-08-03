"""Transmissibility and fault findings (M2.11, remaining checks).

The design here was settled by measurement rather than by picking a threshold.
On a *healthy* Norne mini the interior face transmissibility spans 5e-301 to
3.6e-11, and 493 interior faces sit more than four orders below the median.
Those are the sealed kz of the layer-4 shale - geology the case setup
announces in its own header. A plain ratio threshold reports all 493, and a
check that cries wolf 493 times on a correct model is worse than no check,
because the reader learns to skip it.

So the separation is the whole design:

* **sealed exactly** - zero transmissibility - is a permeability decision.
  Reported once, as INFO.
* **small but open** - orders below its neighbours while still conducting -
  is usually a geometry or upscaling accident, and is the thing worth a look.

A fault is likewise a feature, not a defect: the count is reported, the faces
are not flagged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.research_studio.inspection.grid_report import Finding, Severity


def face_transmissibility(grid: Any, rock: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-face transmissibility and a sealed mask.

    `compute_trans` returns one value per *half*-face, so a face's
    transmissibility is the harmonic average of its two - using the half-face
    values directly would report numbers twice the size they should be.

    A sealed half-face has zero transmissibility, which is infinite
    resistance. That is handled explicitly rather than with a numerical floor,
    because a floor turns a deliberate seal into 1e-300 and makes it
    indistinguishable from an anomalously small face - which is exactly the
    distinction this module exists to keep.
    """
    from pymrst.solvers.compute_trans import compute_trans

    half = np.asarray(compute_trans(grid, rock), dtype=float).ravel()
    cell_faces = np.asarray(grid.cells.faces, dtype=np.int64).ravel()
    count = int(grid.faces.num)

    resistance = np.zeros(count)
    sealed = np.zeros(count, dtype=bool)
    conducting = half > 0.0
    np.add.at(resistance, cell_faces[conducting], 1.0/half[conducting])
    np.logical_or.at(sealed, cell_faces[~conducting], True)

    transmissibility = np.zeros(count)
    live = (~sealed) & (resistance > 0.0)
    transmissibility[live] = 1.0/resistance[live]
    return transmissibility, sealed


def transmissibility_findings(grid: Any, rock: Any, *,
                              orders: float = 4.0) -> list[Finding]:
    """Sealed faces as information, anomalously small ones as a warning."""
    transmissibility, sealed = face_transmissibility(grid, rock)
    neighbors = np.asarray(grid.faces.neighbors, dtype=np.int64)
    interior = np.all(neighbors >= 0, axis=1)

    out: list[Finding] = []
    sealed_interior = np.flatnonzero(interior & sealed)
    if sealed_interior.size:
        out.append(Finding(
            Severity.INFO, "transmissibility",
            f"{sealed_interior.size} interior face(s) are sealed - zero "
            f"transmissibility. This is a permeability decision, not a defect",
            faces=tuple(int(f) for f in sealed_interior[:32]),
        ))

    open_faces = interior & ~sealed & (transmissibility > 0.0)
    if int(open_faces.sum()) >= 2:
        median = float(np.median(transmissibility[open_faces]))
        # Compared against the median of the *open* faces only. Including the
        # sealed ones would drag the median to zero and silence the check.
        gap = np.log10(median/np.maximum(transmissibility, 1e-300))
        anomalous = np.flatnonzero(open_faces & (gap > orders))
        if anomalous.size:
            out.append(Finding(
                Severity.WARNING, "transmissibility",
                f"{anomalous.size} open interior face(s) more than {orders:g} "
                f"orders below the median transmissibility; these conduct, but "
                f"barely, which is usually geometry rather than geology",
                cells=tuple(sorted({int(c) for f in anomalous[:16]
                                    for c in neighbors[f] if c >= 0})),
                faces=tuple(int(f) for f in anomalous[:32]),
            ))
    return out


def fault_findings(grid: Any) -> list[Finding]:
    """How many fault faces there are, and how many actually connect cells.

    A fault is a feature. Counting them is useful - a model with none where
    the geologist expected some is worth knowing about - but flagging them is
    not, so this never rises above INFO.
    """
    tag = getattr(grid.faces, "tag", None)
    if tag is None:
        return []
    tag = np.asarray(tag, dtype=np.int64).ravel()
    if tag.size != int(grid.faces.num):
        return []
    neighbors = np.asarray(grid.faces.neighbors, dtype=np.int64)
    interior = np.all(neighbors >= 0, axis=1)
    faulted = tag == 1
    if not faulted.any():
        return []
    return [Finding(
        Severity.INFO, "faults",
        f"{int(faulted.sum())} fault face(s), of which "
        f"{int((faulted & interior).sum())} connect two active cells",
        faces=tuple(int(f) for f in np.flatnonzero(faulted)[:32]),
    )]


__all__ = ["face_transmissibility", "fault_findings", "transmissibility_findings"]
