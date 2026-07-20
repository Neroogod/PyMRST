"""Compatibility landing file for MRST ``checkCloggingMICP.m``."""

from pymrst.modules.ad_micp.micp_model import MICPModelAD


def check_clogging_micp(model: MICPModelAD, state) -> bool:
    """Return whether the MICP model has reached its clogging tolerance."""
    return model.clogged(state)


__all__ = ["check_clogging_micp"]
