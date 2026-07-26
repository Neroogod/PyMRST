"""MRST ``private/computeMultiPointTransTensorAssembly.m`` compatibility."""

from pymrst.modules.mpfa.compute_multipoint_trans import compute_multipoint_trans


def compute_multi_point_trans_tensor_assembly(g, rock, *, eta=0.0, **_options):
    """Assemble stabilized MPFA-O operators for general 3-D corners."""
    return compute_multipoint_trans(
        g, rock, eta=eta, use_tensor_assembly=True
    )


computeMultiPointTransTensorAssembly = compute_multi_point_trans_tensor_assembly

__all__ = [
    "compute_multi_point_trans_tensor_assembly",
    "computeMultiPointTransTensorAssembly",
]
