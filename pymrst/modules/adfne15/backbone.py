"""Port of ADFNE15 ``Backbone.m``."""
from dataclasses import dataclass
import numpy as np
from .isolated import isolated
from .pipe import PipeModel
@dataclass(frozen=True, slots=True)
class BackboneModel:
    pipe_model: PipeModel; lines: np.ndarray; mask: np.ndarray
def backbone(model, tolerance=1e-14):
    if not model.percolating: return BackboneModel(model,np.empty((0,6)),np.zeros(len(model.pipes),bool))
    removed=np.zeros(len(model.pipes),bool); boundary=model.types!=0
    while True:
        ids=np.flatnonzero(~removed|boundary); leaves=isolated(model.pipes[ids],tolerance)
        new=ids[leaves & ~boundary[ids]]
        if not len(new): break
        removed[new]=True
    mask=~removed
    return BackboneModel(model,model.pipes[mask],mask)
