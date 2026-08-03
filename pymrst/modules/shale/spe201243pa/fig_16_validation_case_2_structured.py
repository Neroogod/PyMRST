"""Structured Case 2 comparison (MRST ``Fig16_validationCase2_structured.m``)."""

from pymrst.modules.shale.spe201243pa._validation_case2 import ValidationCase2, run_validation_case2, setup_validation_case2


def setup_fig_16_validation_case_2_structured(**kwargs):
    return setup_validation_case2(structured=True, **kwargs)


def run_fig_16_validation_case_2_structured(**kwargs):
    return run_validation_case2(structured=True, **kwargs)


Fig16_validationCase2_structured = run_fig_16_validation_case_2_structured

__all__ = ["ValidationCase2", "setup_fig_16_validation_case_2_structured", "run_fig_16_validation_case_2_structured", "Fig16_validationCase2_structured"]
