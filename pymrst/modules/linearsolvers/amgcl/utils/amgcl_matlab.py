"""MRST ``linearsolvers/amgcl/utils/amgcl_matlab.m``: the AMGCL MEX gateway.

`mex/` holds MRST's own C++ sources (`amgcl_matlab.cpp` and the three files it
includes) unchanged, a stand-in `mex.h` for the part of MATLAB's MEX API they
use, and `amgcl_matlab_py.cpp`, which calls their `mexFunction` from Python.
As MRST's `.m` builds the MEX on first use, this builds the extension against
the AMGCL commit MRST bundles and Homebrew's Boost (`build_amgcl_matlab`).

`amgcl_matlab(A, b, opts, tol, maxit, solver_id, reuse_id, x0)` takes `A` as
MATLAB passes it - MRST's solvers pass `A'`, whose CSC arrays the gateway reads
as the CSR arrays of `A`.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
MRST_AMGCL = sorted((HERE.parents[5]/"mrst-2024a-Geomech"/"modules"/"linearsolvers"/
                     "amgcl"/"dependencies").glob("amgcl-*"))


def build_amgcl_matlab() -> Path:
    """Compile `mex/amgcl_matlab_py.cpp` as MRST's build script compiles the MEX."""
    import pybind11
    if not MRST_AMGCL:
        raise FileNotFoundError("MRST's bundled AMGCL headers were not found next to PyMRST")
    boost = subprocess.run(["brew", "--prefix", "boost"], capture_output=True, text=True,
                           check=True).stdout.strip()
    target = HERE/("_amgcl_matlab" + sysconfig.get_config_var("EXT_SUFFIX"))
    cmd = [os.environ.get("CXX", "clang++"), "-O3", "-std=c++17", "-shared", "-fPIC",
           "-undefined", "dynamic_lookup", "-DAMGCL_ASYNCSETUP", f"-I{HERE/'mex'}",
           f"-I{MRST_AMGCL[-1]}", f"-I{boost}/include", f"-I{pybind11.get_include()}",
           f"-I{sysconfig.get_paths()['include']}", str(HERE/"mex"/"amgcl_matlab_py.cpp"),
           "-o", str(target)]
    subprocess.run(cmd, check=True)
    return target


def _gateway():
    name = "pymrst.modules.linearsolvers.amgcl.utils._amgcl_matlab"
    try:
        return importlib.import_module(name)
    except ImportError:
        build_amgcl_matlab()
        importlib.invalidate_caches()
        return importlib.import_module(name)


def amgcl_matlab(A: Any, b: Any, opts: Dict[str, Any], tol: float, maxit: float,
                 solver_id: float, reuse_id: Optional[float] = None,
                 x0: Optional[Any] = None) -> Tuple[np.ndarray, float, int]:
    """MRST `[x, err, its] = amgcl_matlab(A, b, opts, tol, maxit, solver_id, reuse_id, x0)`."""
    # MATLAB's sparse matrices have sorted row indices and no duplicates;
    # AMGCL's block extraction and ILU rely on the sorted order.
    A = sp.csc_matrix(A, dtype=float)
    A.sum_duplicates()
    A.sort_indices()
    x, err, its = _gateway().amgcl_matlab(
        A.shape[0], A.indptr.astype(np.int64), A.indices.astype(np.int64),
        A.data.astype(float), np.asarray(b, dtype=float).ravel(), dict(opts), float(tol),
        float(maxit), float(solver_id), reuse_id, None if x0 is None else np.asarray(x0, float))
    return x, float(err), int(its)


__all__ = ["amgcl_matlab", "build_amgcl_matlab"]


if __name__ == "__main__":
    from pymrst.modules.linearsolvers.amgcl.utils.get_amgcl_mex_struct import (
        get_amgcl_mex_struct,
    )

    def _poisson(n):
        main = 4*np.ones(n*n)
        off = -np.ones(n*n - 1)
        off[np.arange(1, n*n) % n == 0] = 0
        return sp.diags([main, off, off, -np.ones(n*n - n), -np.ones(n*n - n)],
                        [0, 1, -1, n, -n], format="csr")

    def test_the_mex_solves_a_poisson_problem():
        A = _poisson(40)
        b = np.ones(A.shape[0])
        x, err, its = amgcl_matlab(A.T, b, get_amgcl_mex_struct(), 1e-10, 100, 1)
        assert err <= 1e-10 and 0 < its < 100
        assert np.linalg.norm(A @ x - b)/np.linalg.norm(b) < 1e-8
        print(f"ok test_the_mex_solves_a_poisson_problem ({its} iterations)")

    def test_a_transposed_matrix_is_read_as_mrst_reads_it():
        A = _poisson(10) + sp.diags(0.5*np.ones(99), 1, shape=(100, 100))
        b = np.arange(100.0)
        x, err, _ = amgcl_matlab(A.T.tocsr(), b, get_amgcl_mex_struct(), 1e-12, 200, 1)
        assert np.linalg.norm(A @ x - b)/np.linalg.norm(b) < 1e-9
        print("ok test_a_transposed_matrix_is_read_as_mrst_reads_it")

    test_the_mex_solves_a_poisson_problem()
    test_a_transposed_matrix_is_read_as_mrst_reads_it()
    print("\nAll tests passed!")
