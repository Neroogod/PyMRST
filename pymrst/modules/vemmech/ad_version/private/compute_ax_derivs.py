"""MRST ``compute_Ax_derivs.m`` material derivative entry point."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pymrst.modules.vemmech.ad_version import _concat_jac, _value_vec
from pymrst.utils.adi import ADI


def compute_ax_derivs(g, u, extra, E, nu, gamma: float = 1.0,
                      dirdofs=None):
    """Return ``d(S(E,nu) * u) / dcontrols`` for the AD VEM system."""
    E_val = _value_vec(E)
    nu_val = _value_vec(nu)
    jac = _concat_jac(E, nu)
    if not jac:
        nfree = g.griddim * g.nodes.num if dirdofs is None else np.count_nonzero(~dirdofs)
        return sp.csr_array((nfree, 0))
    ejac, nujac = jac
    if not isinstance(E, ADI):
        ejac = sp.csr_array((E_val.size, nujac.shape[1]))
    if not isinstance(nu, ADI):
        nujac = sp.csr_array((nu_val.size, ejac.shape[1]))
    num_ders = ejac.shape[1]
    if num_ders == 0:
        nfree = g.griddim * g.nodes.num if dirdofs is None else np.count_nonzero(~dirdofs)
        return sp.csr_array((nfree, 0))

    N = g.cells.num
    nlin = 3 if g.griddim == 2 else 6
    dim = g.griddim
    vols = np.asarray(g.cells.volumes, dtype=float).ravel()
    repNlin = sp.csr_array(
        (np.ones(nlin * N),
         (np.arange(nlin * N),
          np.repeat(np.arange(N), nlin))),
        shape=(nlin * N, N),
    )
    nlc = np.diff(g.cells.nodePos)
    nlocal = dim * int(np.sum(nlc))
    repX = sp.csr_array(
        (np.ones(nlocal),
         (np.arange(nlocal),
          np.repeat(np.arange(N), nlc * dim))),
        shape=(nlocal, N),
    )

    repE = repNlin @ E_val
    D_dE = extra.D @ sp.diags(1.0 / repE, format="csr")

    Ds = E_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    repDs = repNlin @ Ds
    Dm = extra.D @ sp.diags(1.0 / repDs, format="csr")
    Ds_dnu = E_val * (1.0 + 4.0 * nu_val) / (
        ((1.0 + nu_val) * (1.0 - 2.0 * nu_val)) ** 2
    )
    Dm_dnu = extra.D.copy().astype(float)
    Dm_dnu.eliminate_zeros()
    Dm_dnu.data = np.ones_like(Dm_dnu.data)
    diag_pattern = [-1.0, -1.0, -4.0] if dim == 2 else [
        -1.0, -1.0, -1.0, -4.0, -4.0, -4.0
    ]
    diag = np.tile(diag_pattern, N)
    Dm_dnu = Dm_dnu.tolil()
    Dm_dnu[np.arange(diag.size), np.arange(diag.size)] = diag
    Dm_dnu = Dm_dnu.tocsr()
    D_dnu = (
        Dm @ sp.diags(repNlin @ Ds_dnu, format="csr")
        + Dm_dnu @ sp.diags(repNlin @ Ds, format="csr")
    )

    DNC = np.asarray((extra.NC.T @ extra.NC).diagonal(), dtype=float)
    trDNC = DNC.reshape(N, nlin).sum(axis=1)
    c = gamma / trDNC * vols
    trD = np.asarray(extra.D.diagonal()).reshape(N, nlin).sum(axis=1)
    trD_dnu = np.asarray(D_dnu.diagonal()).reshape(N, nlin).sum(axis=1)
    alpha_dE = c * trD / E_val
    alpha_dnu = c * trD_dnu
    S_dE = sp.diags(repX @ alpha_dE, format="csr")
    S_dnu = sp.diags(repX @ alpha_dnu, format="csr")
    ImPP = extra.I - extra.PP
    local_u = extra.assemb.T @ np.asarray(u, dtype=float).ravel()

    EJ = ejac.multiply(vols[:, None])
    NJ = nujac.multiply(vols[:, None])
    ax_E = extra.assemb @ (extra.WC @ D_dE @ extra.WC.T) @ (
        (repX @ EJ).multiply(local_u[:, None])
    )
    ax_nu = extra.assemb @ (extra.WC @ D_dnu @ extra.WC.T) @ (
        (repX @ NJ).multiply(local_u[:, None])
    )
    ax_stab_E = extra.assemb @ (ImPP.T @ S_dE @ ImPP) @ (
        (repX @ ejac).multiply(local_u[:, None])
    )
    ax_stab_nu = extra.assemb @ (ImPP.T @ S_dnu @ ImPP) @ (
        (repX @ nujac).multiply(local_u[:, None])
    )
    ax = (ax_E + ax_nu + ax_stab_E + ax_stab_nu).tocsr()
    if dirdofs is None:
        return ax
    return ax[~np.asarray(dirdofs, dtype=bool), :]


__all__ = ["compute_ax_derivs"]
