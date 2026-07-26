"""Expose PyMRST MPFA matrices as re-mpfa's ``F``, ``boundF`` and ``div``."""

import numpy as np
import scipy.sparse as sp

from pymrst.utils.adi import ADI


class HydraulicHeadMPFAOperators:
    """Face-flux operators consumed by the original re-mpfa algorithms."""

    def __init__(self, g, discretization, bc):
        self.G = g
        self.discretization = discretization
        self.bc = bc
        self.nc = g.cells.num
        self._face_of_subface = np.asarray(
            discretization["d1"].argmax(axis=1)
        ).ravel()
        self._boundary_rows = np.flatnonzero(discretization["sb"])
        self._boundary_faces = self._face_of_subface[self._boundary_rows]
        weights = np.ones(g.faces.num)
        internal = np.all(np.asarray(g.faces.neighbors) >= 0, axis=1)
        weights[internal] = 0.5
        self._aggregate = (
            sp.diags_array(weights)
            @ discretization["d1"].T
        ).tocsr()
        neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
        rows, cols, data = [], [], []
        for face, (n0, n1) in enumerate(neighbors):
            if n0 >= 0:
                rows.append(int(n0)); cols.append(face); data.append(1.0)
            if n1 >= 0:
                rows.append(int(n1)); cols.append(face); data.append(-1.0)
        self._div = sp.csr_array(
            (data, (rows, cols)), shape=(g.cells.num, g.faces.num)
        )
        self._flux_faces = np.array(
            [int(face) for face, kind in zip(bc.face, bc.type)
             if kind == "flux"],
            dtype=np.int64,
        )

    def F(self, hydraulic_head):
        cell_operator = self.discretization["cf_trans"][:, :self.nc]
        if isinstance(hydraulic_head, ADI):
            subflux = hydraulic_head.__rmatmul__(cell_operator)
            result = subflux.__rmatmul__(self._aggregate)
        else:
            result = self._aggregate @ (cell_operator @ hydraulic_head)
        if self._flux_faces.size:
            result[self._flux_faces] = 0.0
        return result

    def boundF(self, boundary_values):
        values = np.asarray(boundary_values, dtype=float).ravel()
        sub_values = values[self._boundary_faces]
        boundary_operator = self.discretization["cf_trans"][:, self.nc:]
        result = np.asarray(
            self._aggregate @ (boundary_operator @ sub_values)
        ).ravel()
        for face, kind, value in zip(
            self.bc.face, self.bc.type, self.bc.value
        ):
            if kind == "flux":
                result[int(face)] = float(value)
        return result

    def div(self, face_flux):
        if isinstance(face_flux, ADI):
            return face_flux.__rmatmul__(self._div)
        return np.asarray(self._div @ np.asarray(face_flux).ravel()).ravel()


__all__ = ["HydraulicHeadMPFAOperators"]
