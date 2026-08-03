"""Port of ADFNE15 ``Project.m``."""
import numpy as np
def project(points,plane_definition=None,*,return_mask=False):
    points=np.asarray(points,float)
    if plane_definition is None:
        dim=int(np.argmin(np.ptp(points,axis=0))); mask=np.ones(points.shape[1],bool); mask[dim]=False; out=points[:,mask]
        return (out,mask) if return_mask else out
    p=np.asarray(plane_definition,float); first=p[3:6]/np.linalg.norm(p[3:6]); second=p[6:9]/np.linalg.norm(p[6:9]); rel=points-p[:3]; out=np.column_stack((rel@first,rel@second))
    return (out,None) if return_mask else out
