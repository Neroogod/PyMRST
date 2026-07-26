"""MRST ``private/incompMPFATensorAssembly.m`` compatibility."""

from pymrst.modules.mpfa.private.incomp_mpfa_legacy import incomp_mpfa_legacy


def incomp_mpfa_tensor_assembly(
    state, g, mpfa_struct, fluid, **options
):
    """Solve through the common conservative kernel using tensor operators."""
    return incomp_mpfa_legacy(state, g, mpfa_struct, fluid, **options)


incompMPFATensorAssembly = incomp_mpfa_tensor_assembly

__all__ = ["incomp_mpfa_tensor_assembly", "incompMPFATensorAssembly"]
