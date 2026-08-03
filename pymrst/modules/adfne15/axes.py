"""Native Matplotlib port of ADFNE15 ``Axes.m`` presentation settings."""
def axes(axis, *, box=None, view=None, padding=0, light=False):
    axis.set_aspect('equal',adjustable='box')
    if box is not None:
        box=list(box); axis.set_xlim(box[0],box[len(box)//2]); axis.set_ylim(box[1],box[len(box)//2+1])
        if len(box)==6 and hasattr(axis,'set_zlim'): axis.set_zlim(box[2],box[5])
    if view is not None and hasattr(axis,'view_init'): axis.view_init(elev=view[1],azim=view[0])
    if padding:
        axis.margins(padding/100)
    return axis
