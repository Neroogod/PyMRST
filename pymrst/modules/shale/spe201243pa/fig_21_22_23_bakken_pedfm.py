"""Bakken pEDFM workflow (MRST ``Fig21_22_23_bakken_pedfm.m``)."""

from pymrst.modules.shale.spe201243pa._bakken import BakkenCase, run_bakken_case, setup_bakken_case


def setup_fig_21_22_23_bakken_pedfm(**kwargs):
    return setup_bakken_case(method="pedfm", **kwargs)


def run_fig_21_22_23_bakken_pedfm(**kwargs):
    return run_bakken_case(method="pedfm", **kwargs)


Fig21_22_23_bakken_pedfm = run_fig_21_22_23_bakken_pedfm

__all__ = ["BakkenCase", "setup_fig_21_22_23_bakken_pedfm", "run_fig_21_22_23_bakken_pedfm", "Fig21_22_23_bakken_pedfm"]
