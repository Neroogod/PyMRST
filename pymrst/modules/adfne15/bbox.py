"""Port of ADFNE15 ``Bbox.m``."""
import numpy as np
def bounding_box(items,*,per_item=False,point_format=True,intersections=False):
    if isinstance(items,(list,tuple)): minimum=np.vstack([np.min(p,axis=0) for p in items]); maximum=np.vstack([np.max(p,axis=0) for p in items])
    else:
        v=np.atleast_2d(np.asarray(items,float)); c=v.shape[1]
        if c>2 and c%2==0: h=c//2; minimum=np.minimum(v[:,:h],v[:,h:]); maximum=np.maximum(v[:,:h],v[:,h:])
        else: minimum=np.min(v,axis=0)[None]; maximum=np.max(v,axis=0)[None]
    boxes=np.hstack((minimum,maximum))
    if intersections:
        pairs=[]; d=boxes.shape[1]//2
        for i in range(len(boxes)-1):
            for j in range(i+1,len(boxes)):
                if np.all(boxes[i,:d]<=boxes[j,d:]) and np.all(boxes[j,:d]<=boxes[i,d:]): pairs.append((i,j))
        return np.asarray(pairs,int).reshape(-1,2)
    if not per_item: boxes=np.hstack((np.min(minimum,axis=0),np.max(maximum,axis=0)))[None]
    if point_format: return boxes[0] if len(boxes)==1 else boxes
    d=boxes.shape[1]//2; return boxes[:,np.ravel(np.column_stack((np.arange(d),np.arange(d,2*d))))]
