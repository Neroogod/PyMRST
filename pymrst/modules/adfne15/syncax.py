"""Native port of ADFNE15 ``Syncax.m``."""
def syncax(*axes):
    if len(axes)<2:return axes
    source=axes[0]
    for axis in axes[1:]:
        axis.set_xlim(source.get_xlim()); axis.set_ylim(source.get_ylim())
        if hasattr(source,'get_zlim') and hasattr(axis,'set_zlim'): axis.set_zlim(source.get_zlim())
    return axes
