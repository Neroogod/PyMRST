"""Richards equation model, corresponding to MRST ``modelRE.m``."""

from dataclasses import dataclass

import numpy as np

from pymrst.modules.fv_unsat.utility._parameters import get_parameter
from pymrst.modules.fv_unsat.utility.arithmetic_average_mpfa import (
    arithmetic_average_mpfa,
)
from pymrst.modules.fv_unsat.utility.vgm_theta import vgm_theta


@dataclass
class RichardsModel:
    """Cell-centered backward-Euler Richards model using MPFA fluxes."""

    G: object
    phys: object
    mpfa_discr: dict
    bc: object
    gravity: bool = True
    relperm_method: str = "arithmetic"

    def __post_init__(self):
        self.theta, self.krw, self.C_theta = vgm_theta(self.phys)
        self.gamma = float(get_parameter(self.phys, "gamma"))
        self.mu = float(get_parameter(self.phys, "mu"))
        self._prepare_boundary_mapping()

    def _prepare_boundary_mapping(self):
        T = self.mpfa_discr
        boundary_subfaces = np.flatnonzero(T["sb"])
        face_of_subface = np.asarray(T["d1"].argmax(axis=1)).ravel()
        self.boundary_faces = face_of_subface[boundary_subfaces]
        values = {}
        if self.bc is not None:
            values = {
                int(face): float(value)
                for face, kind, value in zip(
                    self.bc.face, self.bc.type, self.bc.value
                )
                if kind == "pressure"
            }
        missing = set(map(int, self.boundary_faces)) - set(values)
        if missing:
            raise NotImplementedError(
                "the current Richards slice requires pressure data on every "
                f"boundary face; missing {sorted(missing)}"
            )
        self.boundary_values = np.array(
            [values[int(face)] for face in self.boundary_faces], dtype=float
        )

    def _total_head(self, psi, boundary=False):
        values = np.asarray(psi, dtype=float)
        if not self.gravity:
            return values
        zmax = float(np.max(self.G.faces.centroids[:, -1]))
        if boundary:
            elevation = zmax - self.G.faces.centroids[self.boundary_faces, -1]
        else:
            elevation = zmax - self.G.cells.centroids[:, -1]
        return values + elevation

    def flux(self, psi):
        """Return global face flux and MPFA subface flux."""
        psi = np.asarray(psi, dtype=float).ravel()
        cell_head = self._total_head(psi)
        boundary_head = self._total_head(self.boundary_values, boundary=True)
        all_pressure = np.r_[cell_head, boundary_head]
        base_subflux = np.asarray(
            self.mpfa_discr["cf_trans"] @ all_pressure
        ).ravel()
        if self.relperm_method != "arithmetic":
            raise NotImplementedError("upstream Richards weighting lands next")
        face_kr = arithmetic_average_mpfa(self.G, self.krw, self.bc, psi)
        subface_kr = np.asarray(self.mpfa_discr["d1"] @ face_kr).ravel()
        subflux = (self.gamma/self.mu)*subface_kr*base_subflux
        face_flux = np.asarray(self.mpfa_discr["d1"].T @ subflux).ravel()
        internal = np.all(self.G.faces.neighbors >= 0, axis=1)
        face_flux[internal] *= 0.5
        return face_flux, subflux

    def Q(self, psi, psi_m=None):
        """MRST ``model.Q`` compatibility wrapper.

        MRST's modified-Picard formulation evaluates relative permeability at
        ``psi_m`` while applying the pressure operator to ``psi``.  The fully
        implicit PyMRST path uses the same value for both when ``psi_m`` is
        omitted.
        """
        psi = np.asarray(psi, dtype=float).ravel()
        mobility_head = psi if psi_m is None else np.asarray(psi_m, dtype=float).ravel()
        cell_head = self._total_head(psi)
        boundary_head = self._total_head(self.boundary_values, boundary=True)
        base_subflux = np.asarray(
            self.mpfa_discr["cf_trans"] @ np.r_[cell_head, boundary_head]
        ).ravel()
        face_kr = arithmetic_average_mpfa(
            self.G, self.krw, self.bc, mobility_head
        )
        subface_kr = np.asarray(self.mpfa_discr["d1"] @ face_kr).ravel()
        return (self.gamma/self.mu)*subface_kr*base_subflux

    def residual(self, psi, psi_n, dt, source=None):
        """Backward-Euler water balance in every cell."""
        psi = np.asarray(psi, dtype=float).ravel()
        psi_n = np.asarray(psi_n, dtype=float).ravel()
        source = (
            np.zeros(self.G.cells.num)
            if source is None else np.asarray(source, dtype=float).ravel()
        )
        _, subflux = self.flux(psi)
        divergence = np.asarray(
            self.mpfa_discr["e_div"] @ subflux
        ).ravel()[:self.G.cells.num]
        volume = np.asarray(self.G.cells.volumes, dtype=float)
        return (
            volume*(self.theta(psi) - self.theta(psi_n))/dt
            + divergence - volume*source
        )

    def psiEq(self, psi, psi_n, psi_m, tau, source):
        """MRST modified-Picard mass equation at iteration ``psi_m``."""
        psi = np.asarray(psi, dtype=float).ravel()
        psi_n = np.asarray(psi_n, dtype=float).ravel()
        psi_m = np.asarray(psi_m, dtype=float).ravel()
        source = np.asarray(source, dtype=float).ravel()
        subflux = self.Q(psi, psi_m)
        divergence = np.asarray(self.mpfa_discr["e_div"] @ subflux).ravel()
        volume = np.asarray(self.G.cells.volumes, dtype=float)
        storage = self.theta(psi_m) + self.C_theta(psi_m)*(psi - psi_m)
        return (
            (volume/tau)*(storage - self.theta(psi_n))
            + divergence[:self.G.cells.num] - volume*source
        )


def model_re(g, phys, mpfa_discr, bc, bc_values=None,
             relperm_method="arithmetic", gravity="on"):
    """MRST-style constructor for :class:`RichardsModel`."""
    del bc_values  # values already live on PyMRST's BoundaryCondition
    return RichardsModel(
        g, phys, mpfa_discr, bc,
        gravity=str(gravity).lower() == "on",
        relperm_method=relperm_method,
    )


modelRE = model_re

__all__ = ["RichardsModel", "model_re", "modelRE"]
