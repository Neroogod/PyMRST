"""Graded Case 2 comparison (MRST ``Fig15_17_18_validationCase2.m``)."""

from pymrst.modules.shale.spe201243pa._validation_case2 import ValidationCase2, run_validation_case2, setup_validation_case2


def setup_fig_15_17_18_validation_case_2(**kwargs):
    return setup_validation_case2(structured=False, **kwargs)


def run_fig_15_17_18_validation_case_2(**kwargs):
    return run_validation_case2(structured=False, **kwargs)


Fig15_17_18_validationCase2 = run_fig_15_17_18_validation_case_2

__all__ = ["ValidationCase2", "setup_fig_15_17_18_validation_case_2", "run_fig_15_17_18_validation_case_2", "Fig15_17_18_validationCase2"]
