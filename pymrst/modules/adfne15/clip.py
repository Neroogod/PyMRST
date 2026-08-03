"""Port of line and box-polygon modes from ADFNE15 ``Clip.m``."""
import numpy as np
def clip_lines(lines,box=(0,0,1,1),*,return_mask=False):
    lines=np.atleast_2d(np.asarray(lines,float)); box=np.asarray(box,float); lower,upper=box[:2],box[2:]; accepted=[]; mask=np.zeros(len(lines),bool)
    for index,row in enumerate(lines):
        p0,p1=row[:2],row[2:]; delta=p1-p0; low,high=0.,1.
        for axis in range(2):
            if abs(delta[axis])<=1e-15:
                if p0[axis]<lower[axis] or p0[axis]>upper[axis]: low,high=1,0; break
            else:
                a=(lower[axis]-p0[axis])/delta[axis]; b=(upper[axis]-p0[axis])/delta[axis]; low=max(low,min(a,b)); high=min(high,max(a,b))
        if low<=high: accepted.append(np.r_[p0+low*delta,p0+high*delta]); mask[index]=True
    out=np.asarray(accepted,float).reshape(-1,4); return (out,mask) if return_mask else out
def _half(poly,axis,boundary,above):
    if not poly.size:return poly.reshape(0,3)
    inside=lambda p:p[axis]>=boundary-1e-14 if above else p[axis]<=boundary+1e-14; out=[]; previous=poly[-1]; was=inside(previous)
    for current in poly:
        now=inside(current)
        if now!=was:
            fraction=(boundary-previous[axis])/(current[axis]-previous[axis]); out.append(previous+fraction*(current-previous))
        if now: out.append(current)
        previous,was=current,now
    return np.asarray(out,float).reshape(-1,3)
def clip_polygons(polygons,box=(0,0,0,1,1,1),*,return_mask=False):
    box=np.asarray(box,float); output=[]; mask=np.zeros(len(polygons),bool)
    for index,p in enumerate(polygons):
        clipped=np.asarray(p,float)
        for axis in range(3): clipped=_half(clipped,axis,box[axis],True); clipped=_half(clipped,axis,box[axis+3],False)
        if len(clipped)>=3: output.append(clipped); mask[index]=True
    return (output,mask) if return_mask else output
