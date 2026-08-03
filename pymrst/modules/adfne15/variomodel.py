"""Native port of ADFNE15 ``Variomodel.m``."""

import numpy as np


def _model_dict(model):
    if model is None: return {"name": "sph", "nugget": 0, "sill": 4, "range": 40}
    if isinstance(model, dict): return model
    values = list(model)
    return {str(values[i]): values[i+1] for i in range(0, len(values), 2)}


def variomodel(model, lags):
    model = _model_dict(model)
    lags = np.asarray(lags, dtype=float)
    name = str(model["name"]).lower(); nugget = float(model.get("nugget", 0))
    if name in {"spherical", "sph"}:
        sill, span = float(model["sill"]), float(model["range"])
        ratio = lags/span
        return nugget + np.where(lags <= span,
                                 sill*(1.5*ratio-.5*ratio**3), sill)
    if name in {"exponential", "exp"}:
        return nugget + float(model["sill"])*(1-np.exp(-3*lags/float(model["range"])))
    if name in {"linear", "lin"}:
        return nugget + float(model["slope"])*lags
    raise ValueError(f"unknown variogram model: {model['name']}")
