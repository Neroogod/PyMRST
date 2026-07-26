"""Compatibility module for MRST ``computeMultiPointTrans.m``."""

from pymrst.modules.mpfa.compute_multipoint_trans import compute_multipoint_trans

def computeMultiPointTrans(
    g, rock, *, eta=0.0, useTensorAssembly=False, **_options
):
    """MRST-compatible option spelling."""
    return compute_multipoint_trans(
        g, rock, eta=eta, use_tensor_assembly=useTensorAssembly
    )

__all__ = ["compute_multipoint_trans", "computeMultiPointTrans"]
