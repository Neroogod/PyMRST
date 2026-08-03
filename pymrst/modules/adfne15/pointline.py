"""Port of ADFNE15 ``Pointline.m``."""
import numpy as np
def pointline(points,indices):
    p=np.asarray(points); i=np.asarray(indices,int); return np.hstack((p[i[:,0]],p[i[:,1]]))
