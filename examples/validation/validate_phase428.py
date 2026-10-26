"""Phase 428: compare setTimeDiscretization choices with MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_core.statefunctions.flux.flow_state import (
    AdaptiveImplicitFlowStateBuilder,
    ExplicitFlowStateBuilder,
    ImplicitFlowStateBuilder,
)
from pymrst.modules.ad_core.utils import set_time_discretization_ad

REF = Path(__file__).with_name("ref_p428_time_discretization.mat")


class _Flow:
    def setFlowStateBuilder(self, builder):
        self.FlowStateBuilder = builder
        return self


def _model():
    return SimpleNamespace(FlowDiscretization=_Flow())


def main():
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase428_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    fim = set_time_discretization_ad(_model(), "fim").FlowDiscretization.FlowStateBuilder
    aim = set_time_discretization_ad(
        _model(), "aim", firstStepImplicit=False, saturationCFL=0.7
    ).FlowDiscretization.FlowStateBuilder
    impes = set_time_discretization_ad(
        _model(), "impes", initialStep=12, changeTimestep=False
    ).FlowDiscretization.FlowStateBuilder

    actual = (
        np.array([[isinstance(fim, ImplicitFlowStateBuilder)]], dtype=float),
        np.array([[
            isinstance(aim, AdaptiveImplicitFlowStateBuilder),
            aim.first_step_implicit,
            aim.saturation_cfl,
        ]], dtype=float),
        np.array([[
            isinstance(impes, ExplicitFlowStateBuilder),
            impes.initial_step,
            impes.change_timestep,
        ]], dtype=float),
    )
    failed = 0
    for name, value, key in zip(
        ("FIM", "AIM options", "IMPES options"),
        actual,
        ("out_is_fim", "out_aim", "out_impes"),
    ):
        error = np.max(np.abs(value - ref[key]))
        ok = value.shape == ref[key].shape and error < 1e-12
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: max|d|={error:.3e}")
    print("\nsetTimeDiscretization matches MRST!" if not failed
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
