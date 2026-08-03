"""Port of ADFNE15 ``Net.m``."""
import numpy as np
def net(x,y,z=None):
    if z is None:
        xx,yy=np.meshgrid(x,y); return np.column_stack((xx.ravel(order="F"),yy.ravel(order="F")))
    xx,yy,zz=np.meshgrid(x,y,z,indexing="xy"); return np.column_stack((xx.ravel(order="F"),yy.ravel(order="F"),zz.ravel(order="F")))
