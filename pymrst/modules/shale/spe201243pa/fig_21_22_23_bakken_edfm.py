"""Bakken EDFM workflow (MRST ``Fig21_22_23_bakken_edfm.m``)."""

from pymrst.modules.shale.spe201243pa._bakken import BakkenCase, run_bakken_case, setup_bakken_case


def setup_fig_21_22_23_bakken_edfm(**kwargs):
    return setup_bakken_case(method="edfm", **kwargs)


def run_fig_21_22_23_bakken_edfm(**kwargs):
    return run_bakken_case(method="edfm", **kwargs)


Fig21_22_23_bakken_edfm = run_fig_21_22_23_bakken_edfm

__all__ = ["BakkenCase", "setup_fig_21_22_23_bakken_edfm", "run_fig_21_22_23_bakken_edfm", "Fig21_22_23_bakken_edfm"]
