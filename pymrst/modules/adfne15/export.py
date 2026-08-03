"""Native file exporters corresponding to ADFNE15 ``Export.m``."""
from pathlib import Path
import json
import numpy as np
def _polygons(values): return [np.asarray(v,float) for v in values] if isinstance(values,(list,tuple)) else [np.asarray(values,float)]
def export(values,filename,**options):
    path=Path(filename); kind=path.suffix.lower().lstrip('.')
    if kind in {'png','pdf'}:
        figure=values.figure if hasattr(values,'figure') else values; figure.savefig(path,**options); return path
    if kind=='mat':
        from scipy.io import savemat; savemat(path,values if isinstance(values,dict) else {'data':values}); return path
    if kind=='xyz': np.savetxt(path,np.asarray(values).reshape(-1,np.asarray(values).shape[-1])); return path
    if kind=='txt':
        with path.open('w') as stream:
            for i,polygon in enumerate(_polygons(values),1):
                stream.write(f'# polygon {i}\n'); np.savetxt(stream,polygon)
        return path
    if kind=='svg':
        lines=np.atleast_2d(np.asarray(values,float)); box=np.r_[lines[:,:2].min(0),lines[:,:2].max(0)]
        body=''.join(f'<line x1="{r[0]}" y1="{r[1]}" x2="{r[2]}" y2="{r[3]}" />' for r in lines)
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{box[0]} {box[1]} {box[2]-box[0]} {box[3]-box[1]}"><g stroke="black">{body}</g></svg>')
        return path
    if kind=='inp':
        points,cells=values; points=np.asarray(points,float); cells=np.asarray(cells,int)
        with path.open('w') as s:
            s.write('*Heading\n** ADFNE15 export\n*Node\n')
            for i,p in enumerate(points,1): s.write(f'{i}, '+', '.join(map(str,p))+'\n')
            s.write('*Element, type=S3\n')
            offset=0 if cells.min(initial=0)==0 else 1
            for i,cell in enumerate(cells,1): s.write(f'{i}, '+', '.join(map(str,cell+1-offset))+'\n')
        return path
    polygons=_polygons(values); points=np.vstack(polygons); offsets=np.cumsum([0]+[len(p) for p in polygons])
    if kind=='vtk':
        with path.open('w') as s:
            s.write('# vtk DataFile Version 3.0\nADFNE15\nASCII\nDATASET POLYDATA\n'); s.write(f'POINTS {len(points)} float\n'); np.savetxt(s,points)
            s.write(f'POLYGONS {len(polygons)} {sum(len(p)+1 for p in polygons)}\n')
            for i,p in enumerate(polygons): s.write(f"{len(p)} "+' '.join(map(str,range(offsets[i],offsets[i+1])))+'\n')
        return path
    if kind=='geo':
        with path.open('w') as s:
            for i,p in enumerate(points,1): s.write(f'Point({i}) = {{{p[0]}, {p[1]}, {p[2]}, 1}};\n')
            line=1
            for pi,p in enumerate(polygons):
                ids=np.arange(offsets[pi]+1,offsets[pi+1]+1); lids=[]
                for a,b in zip(ids,np.roll(ids,-1)): s.write(f'Line({line}) = {{{a}, {b}}};\n'); lids.append(line); line+=1
                s.write(f'Line Loop({pi+1}) = {{{", ".join(map(str,lids))}}};\nPlane Surface({pi+1}) = {{{pi+1}}};\n')
        return path
    if kind in {'json','html','htm'}:
        payload=[p.tolist() for p in polygons]; path.write_text(json.dumps(payload) if kind=='json' else f'<html><body><script>const polygons={json.dumps(payload)};</script></body></html>'); return path
    raise ValueError(f'unsupported export format: {kind}')
