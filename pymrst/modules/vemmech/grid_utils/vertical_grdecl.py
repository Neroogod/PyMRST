"""MRST ``verticalGrdecl.m`` pillar-straightening helper."""

from typing import Dict

import numpy as np

from pymrst.modules.vemmech.grid_utils import _flat, grdecl_xyz


def vertical_grdecl(grdecl: Dict, method: str = "all") -> Dict:
    """Straighten GRDECL pillars as MRST ``verticalGrdecl``."""
    out = dict(grdecl)
    xyz, _ = grdecl_xyz(grdecl)
    top = [0, 1]
    bot = [3, 4]
    if method == "sides":
        xyz[np.ix_(bot, [0], range(xyz.shape[2]))] = \
            xyz[np.ix_(top, [0], range(xyz.shape[2]))]
        xyz[np.ix_(bot, [xyz.shape[1] - 1], range(xyz.shape[2]))] = \
            xyz[np.ix_(top, [xyz.shape[1] - 1], range(xyz.shape[2]))]
        xyz[np.ix_(bot, range(xyz.shape[1]), [0])] = \
            xyz[np.ix_(top, range(xyz.shape[1]), [0])]
        xyz[np.ix_(bot, range(xyz.shape[1]), [xyz.shape[2] - 1])] = \
            xyz[np.ix_(top, range(xyz.shape[1]), [xyz.shape[2] - 1])]
    elif method == "all":
        xyz[top, :, :] = xyz[bot, :, :]
    else:
        raise ValueError(f"unknown vertical_grdecl method {method!r}")
    out["COORD"] = _flat(xyz)
    return out

__all__ = ["vertical_grdecl"]
