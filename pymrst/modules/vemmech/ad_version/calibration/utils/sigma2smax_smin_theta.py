"""MRST ``Sigma2SmaxSminTheta.m`` principal-stress transform."""

from __future__ import annotations

import numpy as np


def sigma_to_smax_smin_theta(o_xx, o_yy=None, o_xy=None, gdim=3):
    """Convert stress components to maximum/minimum principal stress and angle.

    The MRST calibration code passes ``o_xx, o_yy, o_xy`` separately.  For
    compatibility with older PyMRST layout calls, a single stress array may
    also be supplied; its first three columns are interpreted as xx, yy, xy.
    """
    if o_yy is None and o_xy is None:
        sigma = np.asarray(o_xx, dtype=float)
        if sigma.ndim == 1:
            if sigma.size < 3:
                raise ValueError("sigma must contain at least xx, yy and xy")
            o_xx, o_yy, o_xy = sigma[0], sigma[1], sigma[2]
        else:
            if sigma.shape[1] < 3:
                raise ValueError("sigma must contain at least xx, yy and xy")
            o_xx, o_yy, o_xy = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    elif o_yy is None or o_xy is None:
        raise ValueError("pass either sigma or all of o_xx, o_yy and o_xy")

    o_xx = np.asarray(o_xx, dtype=float)
    o_yy = np.asarray(o_yy, dtype=float)
    o_xy = np.asarray(o_xy, dtype=float)
    sgn = -1.0
    s = o_xx + o_yy
    d = o_xx - o_yy
    g = 2.0 * o_xy
    a = np.sqrt(d * d + g * g)
    Smax = 0.5 * (s + sgn * a)
    Smin = 0.5 * (s - sgn * a)

    d = sgn * d
    g = sgn * g
    acos_ix = np.abs(d) < np.abs(g)
    theta2 = np.zeros_like(Smax, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        tmp = np.arccos(np.clip(d[acos_ix] / a[acos_ix], -1.0, 1.0))
    tmp_g = g[acos_ix]
    loc = np.flatnonzero(acos_ix)
    theta2[loc[tmp_g >= 0]] = tmp[tmp_g >= 0]
    theta2[loc[tmp_g < 0]] = -tmp[tmp_g < 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        tmp = np.arcsin(np.clip(g[~acos_ix] / a[~acos_ix], -1.0, 1.0))
    tmp = np.where(np.isnan(tmp), 0.0, tmp)
    tmp_d = d[~acos_ix]
    loc = np.flatnonzero(~acos_ix)
    theta2[loc[tmp_d >= 0]] = tmp[tmp_d >= 0]
    theta2[loc[tmp_d < 0]] = np.pi - tmp[tmp_d < 0]
    theta2 = np.where(theta2 < 0, theta2 + 2.0 * np.pi, theta2)
    return Smax, Smin, theta2 / 2.0


__all__ = ["sigma_to_smax_smin_theta"]
