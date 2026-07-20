"""MRST `ad-blackoil/utils/getCellStatusVO.m`."""

import numpy as np

ETOL: float = np.sqrt(np.finfo(float).eps)


def get_cell_status_vo(sw, so, sg, disgas: bool = True,
                       vapoil: bool = False) -> np.ndarray:
    """Return MRST-style black-oil hydrocarbon status flags.

    Status is encoded as `oil_present + 2*gas_present`:
    1 for undersaturated oil, 2 for undersaturated gas, and 3 when both
    hydrocarbon phases are present.
    """
    sw, so, sg = (np.asarray(x, dtype=float) for x in (sw, so, sg))
    wat_only = sw > 1.0 - ETOL
    oil_present = ((so > 0) | wat_only) if vapoil else np.ones_like(wat_only)
    gas_present = ((sg > 0) | wat_only) if disgas else np.ones_like(wat_only)
    return (
        oil_present.astype(np.int64)
        + 2 * gas_present.astype(np.int64)
    )


blackoil_status = get_cell_status_vo

__all__ = ["get_cell_status_vo", "blackoil_status"]
