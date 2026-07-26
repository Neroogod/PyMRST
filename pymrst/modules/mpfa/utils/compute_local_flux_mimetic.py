"""Compatibility helper for MRST ``computeLocalFluxMimetic.m``."""

from pymrst.modules.mpfa.compute_multipoint_trans import _local_flux_mimetic_ip


def compute_local_flux_mimetic(g, rock, cno, fno, nno, subhfno, eta=0.0):
    """Return the local mixed inner product and continuity vectors."""
    return _local_flux_mimetic_ip(
        g, rock, cno, fno, nno, subhfno, eta=eta
    )


computeLocalFluxMimetic = compute_local_flux_mimetic

__all__ = ["compute_local_flux_mimetic", "computeLocalFluxMimetic"]
