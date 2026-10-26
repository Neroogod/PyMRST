"""MATLAB's `ode45` and `deval`, for MRST ports that integrate with them.

`ode45(fun, [t0, tfinal], y0, AbsTol=..., RelTol=...)` is the Dormand-Prince
5(4) pair with MATLAB's step-size control: the initial step from the scaled
norm of `f(t0, y0)` (unless `InitialStep` is given), `MaxStep` defaulting to
a tenth of the interval, a minimum step of `16*eps(t)`, the last step
stretched to hit `tfinal` when it is within 10%, the mixed relative/absolute
error norm `max|fE|/max(|y|, |ynew|, AbsTol/RelTol)` (RelTol at least
`100*eps`), and MATLAB's step shrink on failure and growth (at most 5x) on
success. It returns MATLAB's solution structure: `x`, `y` and the seven
stages of each step (`idata.f3d`). `deval(sol, x)` evaluates it there, with
MATLAB's 4th-order continuous extension (`ntrp45`) inside a step and the
stored values at the mesh points.

Not ported: events, output functions, `NonNegative`, `NormControl`,
`Refine` output (only the solution structure is returned) and more than two
`tspan` entries.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

import numpy as np

_POW = 1/5
_A = (1/5, 3/10, 4/5, 8/9)
_B = {"11": 1/5, "21": 3/40, "31": 44/45, "41": 19372/6561, "51": 9017/3168,
      "61": 35/384, "22": 9/40, "32": -56/15, "42": -25360/2187, "52": -355/33,
      "33": 32/9, "43": 64448/6561, "53": 46732/5247, "63": 500/1113,
      "44": -212/729, "54": 49/176, "64": 125/192, "55": -5103/18656,
      "65": -2187/6784, "66": 11/84}
_E = (71/57600, -71/16695, 71/1920, -17253/339200, 22/525, -1/40)
_BI = np.array([[1, -183/64, 37/12, -145/128],
                [0, 0, 0, 0],
                [0, 1500/371, -1000/159, 1000/371],
                [0, -125/32, 125/12, -375/64],
                [0, 9477/3392, -729/106, 25515/6784],
                [0, -11/7, 11/3, -55/28],
                [0, 3/2, -4, 5/2]])


def _eps(x: float) -> float:
    """MATLAB `eps(x)`."""
    return float(np.spacing(abs(float(x))))


def ode45(fun: Callable, tspan: Sequence[float], y0: Any, *, AbsTol: float = 1e-6,
          RelTol: float = 1e-3, MaxStep: Optional[float] = None,
          InitialStep: Optional[float] = None) -> SimpleNamespace:
    """MATLAB `sol = ode45(fun, [t0 tfinal], y0, odeset(...))`."""
    t0, tfinal = float(tspan[0]), float(tspan[-1])
    assert len(tspan) == 2, "only [t0 tfinal] is ported"
    y = np.atleast_1d(np.asarray(y0, dtype=float)).ravel()
    tdir = np.sign(tfinal - t0)
    htspan = abs(tfinal - t0)
    rtol = float(RelTol)
    if rtol < 100*np.finfo(float).eps:
        rtol = 100*np.finfo(float).eps
    atol = float(AbsTol)
    threshold = atol/rtol
    safehmax = 16.0*np.finfo(float).eps*max(abs(t0), abs(tfinal))
    defaulthmax = max(0.1*abs(tfinal - t0), safehmax)
    hmax = min(abs(tfinal - t0), abs(defaulthmax if MaxStep is None else MaxStep))
    f0 = np.atleast_1d(np.asarray(fun(t0, y), dtype=float)).ravel()
    t = t0
    tout, yout, f3d = [t], [y.copy()], [np.zeros((y.size, 7))]
    hmin = 16*_eps(t)
    if InitialStep is None:
        absh = min(hmax, htspan)
        rh = np.max(np.abs(f0/np.maximum(np.abs(y), threshold)))/(0.8*rtol**_POW)
        if absh*rh > 1:
            absh = 1/rh
        absh = max(absh, hmin)
    else:
        absh = min(hmax, max(hmin, abs(InitialStep)))
    f1 = f0
    b = _B
    done = False
    while not done:
        hmin = 16*_eps(t)
        absh = min(hmax, max(hmin, absh))
        h = tdir*absh
        if 1.1*absh >= abs(tfinal - t):
            h = tfinal - t
            absh = abs(h)
            done = True
        nofailed = True
        while True:
            y2 = y + h*(b["11"]*f1)
            f2 = _f(fun, t + h*_A[0], y2)
            y3 = y + h*(b["21"]*f1 + b["22"]*f2)
            f3 = _f(fun, t + h*_A[1], y3)
            y4 = y + h*(b["31"]*f1 + b["32"]*f2 + b["33"]*f3)
            f4 = _f(fun, t + h*_A[2], y4)
            y5 = y + h*(b["41"]*f1 + b["42"]*f2 + b["43"]*f3 + b["44"]*f4)
            f5 = _f(fun, t + h*_A[3], y5)
            y6 = y + h*(b["51"]*f1 + b["52"]*f2 + b["53"]*f3 + b["54"]*f4 + b["55"]*f5)
            f6 = _f(fun, t + h, y6)
            tnew = t + h
            if done:
                tnew = tfinal                   # Hit end point exactly.
            h = tnew - t                        # Purify h.
            ynew = y + h*(b["61"]*f1 + b["63"]*f3 + b["64"]*f4 + b["65"]*f5 + b["66"]*f6)
            f7 = _f(fun, tnew, ynew)
            fE = f1*_E[0] + f3*_E[1] + f4*_E[2] + f5*_E[3] + f6*_E[4] + f7*_E[5]
            err = absh*np.max(np.abs(fE/np.maximum(np.maximum(np.abs(y), np.abs(ynew)),
                                                   threshold)))
            if err > rtol:                      # Failed step
                if absh <= hmin:
                    import warnings
                    warnings.warn(f"Failure at t={t:e}. Unable to meet integration tolerances "
                                  f"without reducing the step size below the smallest value "
                                  f"allowed ({hmin:e}) at time t.")
                    return _solution(tout, yout, f3d)
                if nofailed:
                    nofailed = False
                    absh = max(hmin, absh*max(0.1, 0.8*(rtol/err)**_POW))
                else:
                    absh = max(hmin, 0.5*absh)
                h = tdir*absh
                done = False
            else:
                break
        tout.append(tnew)
        yout.append(ynew.copy())
        f3d.append(np.column_stack([f1, f2, f3, f4, f5, f6, f7]))
        if done:
            break
        if nofailed:
            temp = 1.25*(err/rtol)**_POW
            absh = absh/temp if temp > 0.2 else 5.0*absh
        t = tnew
        y = ynew
        f1 = f7
    return _solution(tout, yout, f3d)


def _f(fun: Callable, t: float, y: np.ndarray) -> np.ndarray:
    return np.atleast_1d(np.asarray(fun(t, y), dtype=float)).ravel()


def _solution(tout, yout, f3d) -> SimpleNamespace:
    return SimpleNamespace(solver="ode45", x=np.asarray(tout, dtype=float),
                           y=np.column_stack(yout),
                           idata=SimpleNamespace(f3d=np.stack(f3d, axis=2)))


def ntrp45(tinterp: np.ndarray, t: float, y: np.ndarray, h: float, f: np.ndarray) -> np.ndarray:
    """MATLAB's `ntrp45`: the continuous extension of one ode45 step."""
    s = (np.atleast_1d(np.asarray(tinterp, dtype=float)) - t)/h
    S = np.cumprod(np.vstack([s, s, s, s]), axis=0)
    return y[:, None] + (f @ (h*_BI)) @ S


def deval(sol: SimpleNamespace, xint: Any) -> np.ndarray:
    """MATLAB `Sxint = deval(sol, xint)` for an `ode45` solution."""
    t, y = sol.x, sol.y
    tint = np.atleast_1d(np.asarray(xint, dtype=float)).ravel()
    tdir = np.sign(t[-1] - t[0])
    order = np.argsort(tdir*tint, kind="stable")
    sorted_t = tint[order]
    if (np.any(np.isnan(sorted_t)) or tdir*(sorted_t[0] - t[0]) < 0
            or tdir*(sorted_t[-1] - t[-1]) > 0):
        raise ValueError(f"Attempting to interpolate a solution in [{t[0]:e}, {t[-1]:e}] "
                         f"outside its interval")
    out = np.zeros((y.shape[0], tint.size))
    for k, j in enumerate(order):
        x = sorted_t[k]
        # The last mesh point at or before x (MATLAB's `bottom`).
        bottom = int(np.flatnonzero(tdir*(t - x) <= 0)[-1])
        if x == t[bottom]:
            if bottom > 1 and t[bottom] == t[bottom - 1] and bottom != t.size - 1:
                out[:, j] = (y[:, bottom - 1] + y[:, bottom])/2      # Interface point
            else:
                out[:, j] = y[:, bottom]
        else:
            h = t[bottom + 1] - t[bottom]
            out[:, j] = ntrp45(x, t[bottom], y[:, bottom], h,
                               sol.idata.f3d[:, :, bottom + 1])[:, 0]
    return out


__all__ = ["deval", "ntrp45", "ode45"]


if __name__ == "__main__":
    def test_exponential_growth_to_tolerance():
        sol = ode45(lambda t, y: y, [0, 1], 1.0, AbsTol=1e-10, RelTol=5e-8)
        assert sol.x[0] == 0 and sol.x[-1] == 1
        np.testing.assert_allclose(sol.y[0, -1], np.e, rtol=1e-7)
        mid = deval(sol, [0.37, 0.5, 1.0])
        np.testing.assert_allclose(mid[0], np.exp([0.37, 0.5, 1.0]), rtol=1e-7)
        print("ok test_exponential_growth_to_tolerance")

    def test_integration_backwards_and_at_mesh_points():
        sol = ode45(lambda t, y: -2*t*np.ones_like(y), [2, -1], 0.0, AbsTol=1e-10,
                    RelTol=5e-8)
        np.testing.assert_allclose(deval(sol, [0.0, -1.0])[0], [4.0, 3.0], rtol=1e-9)
        np.testing.assert_array_equal(deval(sol, sol.x[2])[:, 0], sol.y[:, 2])
        print("ok test_integration_backwards_and_at_mesh_points")

    def test_the_default_max_step_is_a_tenth_of_the_interval():
        sol = ode45(lambda t, y: np.zeros_like(y), [0, 10], 1.0)
        assert np.max(np.diff(sol.x)) <= 1.0 + 1e-12
        print("ok test_the_default_max_step_is_a_tenth_of_the_interval")

    test_exponential_growth_to_tolerance()
    test_integration_backwards_and_at_mesh_points()
    test_the_default_max_step_is_a_tenth_of_the_interval()
    print("\nAll tests passed!")
