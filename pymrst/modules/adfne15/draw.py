"""Native Matplotlib renderer for ADFNE15 ``Draw.m`` primitives."""
import numpy as np
def draw(kind,values=None,*,axis=None,**kwargs):
    import matplotlib.pyplot as plt
    name=str(kind).lower(); data=values
    if values is None and not isinstance(kind,str): name='poly' if isinstance(kind,(list,tuple)) else 'line'; data=kind
    first=np.asarray(data[0]) if isinstance(data,(list,tuple)) and len(data) else None
    dimension=3 if first is not None and first.ndim==2 and first.shape[1]==3 else None
    if dimension is None and data is not None:
        a=np.asarray(data); dimension=3 if a.ndim==2 and a.shape[1] in (3,6) else 2
    axis=axis or plt.figure().add_subplot(111,projection='3d' if dimension==3 else None)
    if name in {'line','lin'}:
        for row in np.atleast_2d(data):
            d=row.size//2; axis.plot(*[row[[i,i+d]] for i in range(d)],**kwargs)
    elif name in {'point','pnt'}:
        pts=np.atleast_2d(data); axis.scatter(*[pts[:,i] for i in range(pts.shape[1])],**kwargs)
    elif name in {'poly','ply'}:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        polygons=list(data) if isinstance(data,(list,tuple)) else [data]
        if np.asarray(polygons[0]).shape[1]==3: axis.add_collection3d(Poly3DCollection(polygons,**kwargs))
        else:
            from matplotlib.collections import PolyCollection
            axis.add_collection(PolyCollection(polygons,**kwargs)); axis.autoscale_view()
    elif name in {'cube','cub'}:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        for box in np.atleast_2d(data):
            lo,hi=box[:3],box[3:]; x0,y0,z0=lo; x1,y1,z1=hi
            faces=[[(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],[(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],[(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],[(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],[(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],[(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]]
            axis.add_collection3d(Poly3DCollection(faces,**kwargs))
    elif name in {'sphere','sph'}:
        from .geometry import sphere
        row=np.asarray(data,float).reshape(-1); center=row[:3]; radius=row[3] if len(row)>3 else 1
        mesh=sphere(center=center,radius=radius,resolution=24); axis.plot_surface(mesh.x,mesh.y,mesh.z,**kwargs)
    elif name in {'cylinder','cyl','ellipsoid','els'}:
        row=np.asarray(data,float).reshape(-1); start,end=row[:3],row[3:6]; radius=float(kwargs.pop('radius',.5)); resolution=int(kwargs.pop('resolution',24))
        direction=end-start; length=np.linalg.norm(direction); unit=direction/length; helper=np.array([1.,0,0]) if abs(unit[0])<.9 else np.array([0.,1,0]); first=np.cross(unit,helper); first/=np.linalg.norm(first); second=np.cross(unit,first)
        theta=np.linspace(0,2*np.pi,resolution); axial=np.linspace(0,length,resolution); tt,aa=np.meshgrid(theta,axial)
        radial=radius*(np.sqrt(np.maximum(0,1-(2*aa/length-1)**2)) if name in {'ellipsoid','els'} else np.ones_like(aa))
        xyz=start+aa[...,None]*unit+radial[...,None]*(np.cos(tt)[...,None]*first+np.sin(tt)[...,None]*second)
        axis.plot_surface(xyz[...,0],xyz[...,1],xyz[...,2],**kwargs)
    else: raise ValueError(f'unsupported Draw primitive: {kind}')
    return axis
