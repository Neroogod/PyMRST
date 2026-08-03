"""Port of ADFNE15 ``Stack.m``."""
import numpy as np
def stack(values):
    values=list(values)
    return np.empty((0,0)) if not values else np.concatenate(values,axis=0)
