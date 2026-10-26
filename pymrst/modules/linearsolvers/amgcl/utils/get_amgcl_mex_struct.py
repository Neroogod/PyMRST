"""MRST ``linearsolvers/amgcl/utils/getAMGCLMexStruct.m``: the option struct
the AMGCL MEX gateway reads, MRST's defaults overridden by keyword, with the
preconditioner, coarsening, relaxations and solver translated to their ids.

MATLAB's `maxNumCompThreads()` is `os.cpu_count()` here; the gateway is built
without OpenMP on this machine, so it runs serially regardless.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from pymrst.modules.linearsolvers.amgcl.utils.translate_options_amgcl import (
    translate_options_amgcl,
)

_RELAX = (("ilut_p", 2), ("ilut_tau", 0.01), ("iluk_k", 1), ("ilu_damping", 1),
          ("jacobi_damping", 0.72), ("chebyshev_degree", 5), ("chebyshev_lower", 1.0/30),
          ("chebyshev_power_iters", 0))


def get_amgcl_mex_struct(**options: Any) -> Dict[str, Any]:
    """MRST `amg_opt = getAMGCLMexStruct(...)`."""
    amg_opt: Dict[str, Any] = dict(
        preconditioner="amg", coarsening="aggregation", relaxation="spai0",
        s_relaxation="ilu0", solver="bicgstab", block_size=0, active_rows=0, use_drs=False,
        drs_eps_ps=0.02, drs_eps_dd=0.2, drs_row_weights=np.zeros(0), update_sprecond=False,
        update_ptransfer=False, cpr_blocksolver=True, coarse_enough=-1, direct_coarse=True,
        max_levels=-1, ncycle=1, npre=1, npost=1, pre_cycles=1, gmres_m=30, lgmres_k=3,
        lgmres_always_reset=True, lgmres_store_av=True, idrs_s=4, idrs_omega=0.7,
        idrs_replacement=False, bicgstabl_l=2, bicgstabl_delta=0, bicgstabl_convex=True,
        aggr_eps_strong=0.08, aggr_over_interp=1.0, rs_eps_strong=0.25, rs_trunc=True,
        rs_eps_trunc=0.2, aggr_relax=2.0/3.0, write_params=False,
        nthreads=os.cpu_count() or 1, verbose=False)
    for name, val in _RELAX:
        amg_opt[name] = val
        amg_opt["s_" + name] = val
    for key, val in options.items():
        if key not in amg_opt:
            raise KeyError(f"Unknown option '{key}'")
        amg_opt[key] = val
    amg_opt["preconditioner"] = translate_options_amgcl("preconditioner", amg_opt["preconditioner"])[0]
    amg_opt["coarsening"] = translate_options_amgcl("coarsening", amg_opt["coarsening"])[0]
    amg_opt["relaxation"] = translate_options_amgcl("relaxation", amg_opt["relaxation"])[0]
    amg_opt["s_relaxation"] = translate_options_amgcl("relaxation", amg_opt["s_relaxation"])[0]
    amg_opt["solver"] = translate_options_amgcl("solver", amg_opt["solver"])[0]
    return amg_opt


getAMGCLMexStruct = get_amgcl_mex_struct
__all__ = ["getAMGCLMexStruct", "get_amgcl_mex_struct"]


if __name__ == "__main__":
    def test_defaults_are_mrsts_translated():
        opt = get_amgcl_mex_struct()
        assert (opt["preconditioner"], opt["coarsening"], opt["relaxation"],
                opt["s_relaxation"], opt["solver"]) == (1, 3, 1, 3, 1)
        assert opt["s_ilu_damping"] == 1 and opt["gmres_m"] == 30
        print("ok test_defaults_are_mrsts_translated")

    def test_overrides_and_unknown_options():
        assert get_amgcl_mex_struct(relaxation="ilu0", block_size=3)["relaxation"] == 3
        try:
            get_amgcl_mex_struct(no_such_option=1)
        except KeyError:
            print("ok test_overrides_and_unknown_options")
        else:
            raise AssertionError("expected KeyError")

    test_defaults_are_mrsts_translated()
    test_overrides_and_unknown_options()
    print("\nAll tests passed!")
