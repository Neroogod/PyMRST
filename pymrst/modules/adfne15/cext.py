"""Native NumPy port of geometric helpers in ADFNE15 ``CExt.m``."""
import numpy as np
from .polyclip import _clip_halfspace
class CExt:
    @staticmethod
    def line_x_edge(line, edge, tolerance=1e-14):
        line=np.asarray(line,float).reshape(-1); edge=np.asarray(edge,float).reshape(-1)
        origin=line[:2]; direction=line[2:4]; start=edge[:2]; delta=edge[2:4]-start
        cross=direction[0]*delta[1]-direction[1]*delta[0]
        if abs(cross)<=tolerance:
            return np.array([np.inf,np.inf]) if abs(np.cross(start-origin,direction))<=tolerance else np.array([np.nan,np.nan])
        t=np.cross(start-origin,delta)/cross; u=np.cross(start-origin,direction)/cross
        return origin+t*direction if -tolerance<=u<=1+tolerance else np.array([np.nan,np.nan])
    @staticmethod
    def plane_normal(definition):
        p=np.asarray(definition,float); return np.cross(p[...,3:6],p[...,6:9])
    @staticmethod
    def vector_cross_3d(a,b): return np.cross(a,b)
    @staticmethod
    def vector_norm_3d(v): return np.linalg.norm(v,axis=-1)
    @staticmethod
    def normalize_vector_3d(v):
        v=np.asarray(v,float); return v/np.linalg.norm(v,axis=-1,keepdims=True)
    @staticmethod
    def line_position_3d(points,lines):
        points=np.asarray(points,float); lines=np.asarray(lines,float); direction=lines[...,3:6]
        denominator=np.sum(direction**2,axis=-1); return np.divide(np.sum((points-lines[...,:3])*direction,axis=-1),denominator,out=np.zeros_like(denominator),where=denominator>np.finfo(float).eps)
    @staticmethod
    def is_below_plane(points,definition):
        definition=np.asarray(definition,float); line=np.r_[definition[:3],CExt.plane_normal(definition)]; return CExt.line_position_3d(points,line)<=0
    @staticmethod
    def intersect_line_plane(lines,definition):
        lines=np.atleast_2d(np.asarray(lines,float)); definition=np.asarray(definition,float); n=CExt.plane_normal(definition); direction=lines[:,3:6]
        denominator=direction@n; t=((definition[:3]-lines[:,:3])@n)/denominator
        out=lines[:,:3]+t[:,None]*direction; out[np.abs(denominator)<1e-14]=np.nan; return out
    @staticmethod
    def point_on_plane(points,definition):
        points=np.atleast_2d(np.asarray(points,float)); definition=np.asarray(definition,float); n=CExt.plane_normal(definition); return points+(((definition[:3]-points)@n)/(n@n))[:,None]*n
    @staticmethod
    def clip_polygon_half_plane(polygon,definition):
        origin=definition[:3]; a=definition[3:6]; b=definition[6:9]
        boundary=np.vstack((origin,origin+a,origin+a+b,origin+b)); return _clip_halfspace(np.asarray(polygon,float),boundary)
    @staticmethod
    def is_parallel(a,b,tolerance=1e-14): return np.linalg.norm(np.cross(a,b))<=tolerance if np.size(a)==3 else abs(np.cross(a,b))<=tolerance
    @staticmethod
    def edge_position(point,edge):
        edge=np.asarray(edge,float); d=edge[len(edge)//2:]-edge[:len(edge)//2]; return np.dot(np.asarray(point)-edge[:len(d)],d)/np.dot(d,d)
    @staticmethod
    def plane_from_normal(point,normal):
        point=np.asarray(point,float); normal=CExt.normalize_vector_3d(np.asarray(normal,float).reshape(1,3))[0]; helper=np.array([1.,0,0]) if abs(normal[0])<1-1e-14 else np.array([0.,1,0]); first=CExt.normalize_vector_3d(np.cross(normal,helper).reshape(1,3))[0]; second=-CExt.normalize_vector_3d(np.cross(first,normal).reshape(1,3))[0]; return np.r_[point,first,second]
    @staticmethod
    def clip_line(lines,box,*,return_mask=False):
        from .clip import clip_lines
        return clip_lines(lines,box,return_mask=return_mask)
    @staticmethod
    def intersect_edges(first,second,tolerance=1e-14):
        first=np.asarray(first,float).reshape(-1); second=np.asarray(second,float).reshape(-1); p=first[:2]; r=first[2:]-p; q=second[:2]; s=second[2:]-q; cross=np.cross(r,s)
        if abs(cross)<=tolerance:
            if abs(np.cross(q-p,r))>tolerance:return np.array([np.nan,np.nan])
            candidates=[]
            for point in (p,first[2:],q,second[2:]):
                if -tolerance<=CExt.edge_position(point,first)<=1+tolerance and -tolerance<=CExt.edge_position(point,second)<=1+tolerance:candidates.append(point)
            return np.mean(candidates,axis=0) if candidates else np.array([np.nan,np.nan])
        t=np.cross(q-p,s)/cross; u=np.cross(q-p,r)/cross
        return p+t*r if -tolerance<=t<=1+tolerance and -tolerance<=u<=1+tolerance else np.array([np.nan,np.nan])
