"""Conductive-fracture validation (MRST ``Fig10to13TeneCasesHighKf.m``)."""

from pymrst.modules.shale.spe201243pa._tene_cross import TeneCrossCase, run_tene_cross_case, setup_tene_cross_case


def setup_fig_10_to_13_tene_high_kf(**kwargs):
    return setup_tene_cross_case(high_conductivity=True, **kwargs)


def run_fig_10_to_13_tene_high_kf(**kwargs):
    return run_tene_cross_case(high_conductivity=True, **kwargs)


Fig10to13TeneCasesHighKf = run_fig_10_to_13_tene_high_kf

__all__ = ["TeneCrossCase", "setup_fig_10_to_13_tene_high_kf", "run_fig_10_to_13_tene_high_kf", "Fig10to13TeneCasesHighKf"]
