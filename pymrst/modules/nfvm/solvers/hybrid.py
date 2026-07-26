"""Hybrid facewise discretization selector (MRST ``Hybrid.m``)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Hybrid:
    """Select gradients from multiple discretizations on face blocks."""

    G: object
    discretizations: list
    faceblocks: list

    def gradient(self, pressure):
        if not self.discretizations:
            raise ValueError("Hybrid needs at least one discretization")
        internal_faces = np.flatnonzero(
            np.all(self.G.faces.neighbors >= 0, axis=1)
        )
        gradient = self.discretizations[0].gradient(pressure).copy()
        for discretization, block in zip(
            self.discretizations[1:], self.faceblocks[1:]
        ):
            candidate = discretization.gradient(pressure)
            selected = np.asarray(block)
            if selected.dtype == bool:
                if selected.size != self.G.faces.num:
                    raise ValueError("boolean face block must cover all grid faces")
                mask = selected[internal_faces]
            else:
                mask = np.isin(internal_faces, selected.astype(np.int64))
            gradient[mask] = candidate[mask]
        return gradient

