"""Deterministic callable demonstrations corresponding to ADFNE15 ``Demo.m``."""
from .dfn import generate_dfn
from .info import info
from .variocloud import variocloud
def demo(*,seed=1234567890):
    network=generate_dfn(count=50,seed=seed); metadata=info(network.fractures)
    cloud=variocloud(metadata.centers,metadata.lengths) if len(network.fractures)>1 else None
    return {'network':network,'info':metadata,'variogram_cloud':cloud}
