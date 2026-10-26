"""MRST ``ad-core/backends/diagonal/operators/mex/testMexDiagonalOperators.m``:
time the diagonal backend's discrete operators - plain, accelerated, and the
sparse `ADI` equivalent - on a random cell value with a `block_size`-wide
diagonal Jacobian, and check the accelerated results against the plain ones.

MRST's accelerated path is its C++ MEX operators; PyMRST's is the compiled
numba kernels the diagonal Jacobians dispatch to (`kernels/`). "Diagonal"
here runs with those kernels switched off, "Diagonal-MEX" with them on.
`results[name]` holds the per-call times (`t_matlab`, `t_mex`, `t_sparse`)
and the description, as MRST's struct does; with `print` the same report is
written.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import scipy.sparse as sp

_KERNEL_MODULES = (
    "pymrst.modules.ad_core.backends.diagonal.diagonal_jacobian",
    "pymrst.modules.ad_core.backends.diagonal.row_major_diagonal_jacobian",
    "pymrst.modules.ad_core.backends.diagonal.generic_ad",
)


@contextmanager
def _compiled_kernels(enabled: bool) -> Iterator[None]:
    """MRST's `useMex` flag: the Jacobians use the compiled kernels or not."""
    import importlib
    modules = [importlib.import_module(m) for m in _KERNEL_MODULES]
    saved = [m.HAS_NUMBA_KERNELS for m in modules]
    try:
        for m, flag in zip(modules, saved):
            m.HAS_NUMBA_KERNELS = flag and enabled
        yield
    finally:
        for m, flag in zip(modules, saved):
            m.HAS_NUMBA_KERNELS = flag


def test_mex_diagonal_operators(model: Any, *, print: bool = False, block_size: int = 5,
                                iterations: int = 5, nc: Optional[int] = None,
                                testSparse: bool = True, testDiag: bool = True,
                                testMex: bool = True, timeit: bool = False,
                                ordering: Any = None) -> Dict[str, Any]:
    """MRST `results = testMexDiagonalOperators(model, ...)`; `model` may be a
    model, a grid, or Cartesian dimensions."""
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.backends.diagonal.diagonal_jacobian import DiagonalJacobian
    from pymrst.modules.ad_core.backends.diagonal.generic_ad import GenericAD
    from pymrst.modules.ad_core.backends.diagonal.operators import (
        divergence_with_accumulation, discrete_divergence, face_average,
        single_point_upwind, two_point_gradient,
    )
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.params import make_rock
    from pymrst.utils.adi import ADI

    opt = SimpleNamespace(print=print, block_size=block_size, iterations=iterations, nc=nc,
                          testSparse=testSparse, testDiag=testDiag, testMex=testMex,
                          timeit=timeit)
    if opt.timeit:
        opt.iterations = 1
    if isinstance(model, (list, tuple, np.ndarray)):
        model = compute_geometry(cart_grid(list(model)))
    if not hasattr(model, "operators"):
        G = model
        model = SimpleNamespace(G=G, operators=setup_operators_tpfa(G, make_rock(G, 1, 1)))
    N = np.asarray(model.operators.N, dtype=np.int64)
    G = model.G
    if opt.nc is None:
        opt.nc = int(N.max()) + 1
    if ordering is not None:
        N = np.asarray(ordering)[N]
    results: Dict[str, Any] = dict(vars(opt))
    nf = N.shape[0]
    ncell = G.cells.num
    rng = np.random.default_rng()
    d = rng.random((opt.nc, opt.block_size))
    flag = rng.random(nf) > 0.5
    rows = np.tile(np.arange(opt.nc), opt.block_size)
    cols = np.arange(opt.nc*opt.block_size)
    cell_value = GenericAD(rng.random(opt.nc), [DiagonalJacobian(
        d.ravel(order="F"), rows, cols, (opt.nc, opt.nc*opt.block_size))])
    J = [sp.csr_array(sp.diags(d[:, i], shape=(ncell, ncell))) for i in range(opt.block_size)]
    cell_value_sparse = ADI(np.array(cell_value.val, copy=True), J)
    ops_sparse = setup_operators_tpfa(G, make_rock(G, 1, 1))

    def fn(x, y, z):
        return x*y + z

    def bench(name: str, desc: str, f: Callable[[], Any], f_sparse: Callable[[], Any]):
        return _test_function(f, f_sparse, name, desc, opt, results)

    bench("cellmult", "Multiply and add (cell)",
          lambda: fn(cell_value, 2*cell_value, 3*cell_value),
          lambda: fn(cell_value_sparse, 2*cell_value_sparse, 3*cell_value_sparse))
    subset = np.arange(0, ncell, 5)
    bench("cellsubset", "Take large subset (cell)", lambda: cell_value[subset],
          lambda: cell_value_sparse[subset])
    small = np.array([0, int(np.ceil(ncell/2)) - 1, ncell - 1])
    bench("cellsubset_small", "Take small subset (cell)", lambda: cell_value[small],
          lambda: cell_value_sparse[small])
    bench("faceavg", "Face average", lambda: face_average(N, cell_value),
          lambda: ops_sparse.avg(cell_value_sparse))
    face_value, face_value_mex, face_value_sparse = bench(
        "upwind", "Single-point upwind", lambda: single_point_upwind(flag, N, cell_value),
        lambda: ops_sparse.upw(flag, cell_value_sparse))
    bench("gradient", "Two-point gradient", lambda: two_point_gradient(N, cell_value),
          lambda: ops_sparse.grad(cell_value_sparse))
    bench("facemult", "Multiply and add (face)",
          lambda: fn(face_value, 2*face_value, 3*face_value),
          lambda: fn(face_value_sparse, 2*face_value_sparse, 3*face_value_sparse))
    options = {"N": N, "nc": ncell}
    bench("div", "Discrete divergence", lambda: discrete_divergence(N, ncell, face_value),
          lambda: ops_sparse.div(face_value_sparse))
    bench("accdiv", "Accumulation + divergence",
          lambda: divergence_with_accumulation(cell_value, face_value, options),
          lambda: ops_sparse.AccDiv(cell_value_sparse, face_value_sparse))
    bench("sparse", "Class -> Sparse", lambda: cell_value.jac[0].to_sparse(),
          (lambda: face_value_sparse.jac[0]) if opt.testSparse else (lambda: np.nan))
    if opt.print:
        _out("*"*70)
    return results


def _test_function(fn: Callable[[], Any], fn_sparse: Callable[[], Any], shortname: str,
                   name: str, opt: Any, results: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    check = True
    if opt.testDiag:
        with _compiled_kernels(False):
            matlab, t_m = _perform_benchmark(fn, opt)
    else:
        matlab, t_m, check = np.nan, np.nan, False
    if opt.testMex:
        with _compiled_kernels(True):
            mex, t_c = _perform_benchmark(fn, opt)
    else:
        mex, t_c, check = np.nan, np.nan, False
    if check:
        if sp.issparse(matlab):
            v_error = 0.0
            j_error = _norm_inf(matlab - mex)/_norm_inf(matlab)
        else:
            v_error = np.linalg.norm(matlab.val - mex.val)/np.linalg.norm(matlab.val)
            jm, jc = matlab.jac[0].to_sparse(), mex.jac[0].to_sparse()
            j_error = _norm_inf(jm - jc)/_norm_inf(jm)
    else:
        v_error = j_error = np.nan
    if opt.testSparse:
        out_sparse, t_s = _perform_benchmark(fn_sparse, opt)
    else:
        out_sparse, t_s = np.nan, np.nan
    its = opt.iterations
    if opt.print:
        _out("*"*70)
        _out(f"* {name}\n* Diagonal: {t_m/its:4f}s, Diagonal-MEX: {t_c/its:4f}s "
             f"({t_m/t_c:1.2f} speedup)")
        if opt.testSparse:
            _out(f"*   Sparse: {t_s/its:4f}s, Diagonal-MEX: {t_c/its:4f}s ({t_s/t_c:1.2f} speedup)")
        _out(f"* -> Errors: Value {v_error:1.2g}, Jacobian {j_error:1.2g}")
    results[shortname] = {"t_matlab": t_m/its, "t_mex": t_c/its, "t_sparse": t_s/its,
                          "description": name, "value_error": v_error,
                          "jacobian_error": j_error}
    return matlab, mex, out_sparse


def _perform_benchmark(fn: Callable[[], Any], opt: Any) -> Tuple[Any, float]:
    timer = time.perf_counter()
    for _ in range(opt.iterations):
        out = fn()
    return out, time.perf_counter() - timer


def _norm_inf(M: Any) -> float:
    return float(abs(sp.csr_array(M)).sum(axis=1).max()) if M.shape[0] else 0.0


def _out(text: str) -> None:
    import builtins
    builtins.print(text)


testMexDiagonalOperators = test_mex_diagonal_operators
__all__ = ["testMexDiagonalOperators", "test_mex_diagonal_operators"]


if __name__ == "__main__":
    def test_every_operator_agrees_with_and_without_kernels():
        results = test_mex_diagonal_operators([12, 10, 4], block_size=3, iterations=2)
        names = ("cellmult", "cellsubset", "cellsubset_small", "faceavg", "upwind", "gradient",
                 "facemult", "div", "accdiv", "sparse")
        for name in names:
            r = results[name]
            assert r["value_error"] < 1e-14 and r["jacobian_error"] < 1e-14, (name, r)
            assert r["t_mex"] > 0 and r["t_matlab"] > 0
        print("ok test_every_operator_agrees_with_and_without_kernels")

    def test_the_report_prints():
        test_mex_diagonal_operators([6, 6, 2], block_size=2, iterations=1, print=True)
        print("ok test_the_report_prints")

    test_every_operator_agrees_with_and_without_kernels()
    test_the_report_prints()
    print("\nAll tests passed!")
