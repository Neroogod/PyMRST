"""Native port of ADFNE15 ``Grid.m``."""
import numpy as np
from .clip import clip_lines
from .rotate import rotate
def grid(count, *, center=(.5,.5), angle=0, box=(0,0,1,1)):
    count=int(count)
    if count<2: return np.empty((0,4))
    s=1/count; p=np.arange(1,count)*s
    lines=np.vstack((np.column_stack((p,np.zeros_like(p),p,np.ones_like(p))),
                     np.column_stack((np.zeros_like(p),p,np.ones_like(p),p))))
    return clip_lines(rotate(lines,angle=angle,center=center),box)
