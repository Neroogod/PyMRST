"""MRST ``linearsolvers/amgcl/utils/translateOptionsAMGCL.m``: the numeric id
(1-based, as the MEX gateway expects) of an AMGCL option name, with its name,
description and parameters; with an empty value, all of them."""

from __future__ import annotations

from typing import Any, List, Tuple

_GROUPS = {
    "preconditioner": (["amg", "relaxation", "dummy"],
                       ["Algebraic multigrid", "Relaxation as preconditioner",
                        "Dummy (do nothing)"],
                       [None, None, None]),
    "coarsening": (["smoothed_aggregation", "ruge_stuben", "aggregation", "smoothed_aggr_emin"],
                   ["Smoothed aggregation", "Ruge-Stuben / classic AMG coarsening",
                    "Aggregation with constant interpolation",
                    "Smoothed aggregation (energy minimizing)"],
                   [["aggr_eps_strong", "aggr_over_interp", "aggr_relax"],
                    ["rs_eps_strong", "rs_trunc", "rs_eps_trunc"],
                    ["aggr_eps_strong", "aggr_over_interp", "aggr_relax"],
                    ["aggr_eps_strong", "aggr_over_interp", "aggr_relax"]]),
    "relaxation": (["spai0", "gauss_seidel", "ilu0", "iluk", "ilut", "damped_jacobi", "spai1",
                    "chebyshev"],
                   ["Sparse approximate inverse of order 0", "Gauss-Seidel smoothing.",
                    "Incomplete LU-factorization with zero fill-in - ILU(0)",
                    "Incomplete LU-factorization of order k - ILU(k)",
                    "Incomplete LU-factorization with thresholding - ILU(t)",
                    "Damped Jacobi smoothing", "Sparse approximate inverse of order 1",
                    "Chebyshev smoothing"],
                   [[], [], ["ilu_damping"], ["ilu_damping, ilu_k parameter"],
                    ["ilu_damping, ilut_tau"], ["jacobi_damping"], [],
                    ["chebyshev_degree", "chebyshev_lower", "chebyshev_power_its"]]),
    "solver": (["bicgstab", "cg", "bicgstabl", "gmres", "lgmres", "fgmres", "idrs"],
               ["Biconjugate gradient stabilized method.", "Conjugate gradient method.",
                "Biconjugate gradient stabilized method (l variant)",
                "Generalized minimal residual method", "Generalized minimal residual method",
                "Generalized minimal residual method (f variant)",
                "Induced Dimension Reduction method"],
               [[], [], ["bicgstabl_convex", "bicgstabl_delta", "bicgstabl_l"], ["gmres_m"],
                ["lgmres_always_reset", "lgmres_k", "lgmres_store_av"], [],
                ["idrs_omega", "idrs_replacement", "idrs_s"]]),
}


def translate_options_amgcl(name: str, value: Any) -> Tuple[Any, Any, Any, Any]:
    """MRST `[n, name, description, parameter] = translateOptionsAMGCL(name, value)`."""
    assert isinstance(name, str), "First input must be a name"
    key = name.lower()
    if key not in _GROUPS:
        raise ValueError(f"Unknown option: '{name}'")
    names, descriptions, parameters = _GROUPS[key]
    index = _map_argument(value, names, key)
    if index == ":":
        return index, names, descriptions, parameters
    return index, names[index - 1], descriptions[index - 1], parameters[index - 1]


def _map_argument(value: Any, types: List[str], group: str) -> Any:
    if value is None or (hasattr(value, "__len__") and len(value) == 0):
        return ":"
    if isinstance(value, str):
        matches = [i + 1 for i, t in enumerate(types) if t.lower() == value.lower()]
        if not matches:
            raise ValueError(f"Unknown {group} name: '{value}'")
        return matches[0]
    return int(value)


translateOptionsAMGCL = translate_options_amgcl
__all__ = ["translateOptionsAMGCL", "translate_options_amgcl"]


if __name__ == "__main__":
    def test_names_map_to_mrsts_one_based_ids():
        assert translate_options_amgcl("relaxation", "ilu0")[0] == 3
        assert translate_options_amgcl("coarsening", "aggregation")[0] == 3
        assert translate_options_amgcl("solver", "bicgstab")[:2] == (1, "bicgstab")
        assert translate_options_amgcl("preconditioner", 2)[1] == "relaxation"
        print("ok test_names_map_to_mrsts_one_based_ids")

    def test_an_unknown_name_is_refused():
        try:
            translate_options_amgcl("solver", "nope")
        except ValueError:
            print("ok test_an_unknown_name_is_refused")
        else:
            raise AssertionError("expected ValueError")

    test_names_map_to_mrsts_one_based_ids()
    test_an_unknown_name_is_refused()
    print("\nAll tests passed!")
