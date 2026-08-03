"""Derive mutually consistent isotropic poroelastic parameters.

This is the Python owner for MRST ``ad-mechanics/utils/poroParams.m``.  The
MRST-style interface accepts porosity, a uniform-grain flag and any consistent
subset of named parameters.  The historical four-argument PyMRST interface
(``E, nu, alpha, M``) remains supported.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, Mapping, Sequence
import warnings

import numpy as np
from scipy.optimize import least_squares, root_scalar


PARAMETER_NAMES = (
    "K", "H", "R", "M", "K_s", "K_p", "K_f", "K_phi", "K_v",
    "S", "S_sigma", "S_epsilon", "S_gamma", "alpha", "beta",
    "gamma", "eta", "c_m", "B", "K_u", "K_vu", "E", "E_u",
    "lambda", "lambda_u", "nu", "nu_u", "G",
)

_MODULI = {
    "K", "H", "R", "M", "K_s", "K_p", "K_f", "K_phi", "K_v",
    "K_u", "K_vu", "E", "E_u", "lambda", "lambda_u", "G",
}
_STORATIVITIES = {"S", "S_sigma", "S_epsilon", "S_gamma", "c_m"}


@dataclass(frozen=True)
class Relation:
    """One zero-valued poroelastic relation."""

    name: str
    args: tuple[str, ...]
    residual: Callable[..., float]


def _elastic_relations(suffix: str = "") -> list[Relation]:
    if suffix:
        k, e, lam, nu, kv = "K_u", "E_u", "lambda_u", "nu_u", "K_vu"
    else:
        k, e, lam, nu, kv = "K", "E", "lambda", "nu", "K_v"
    # The shear modulus is unchanged between drained and undrained response.
    relations = [
        Relation(f"{k}-lambda-G", (k, lam, "G"), lambda K, l, G: K - l - 2*G/3),
        Relation(f"{e}-lambda-G", (e, lam, "G"), lambda E, l, G: E - G*(3*l+2*G)/(l+G)),
        Relation(f"{nu}-lambda-G", (nu, lam, "G"), lambda n, l, G: n-l/(2*(l+G))),
        Relation(f"{kv}-lambda-G", (kv, lam, "G"), lambda v, l, G: v-l-2*G),
        Relation(f"{e}-{k}-G", (e, k, "G"), lambda E, K, G: E-9*K*G/(3*K+G)),
        Relation(f"{nu}-{k}-G", (nu, k, "G"), lambda n, K, G: n-(3*K-2*G)/(2*(3*K+G))),
        Relation(f"{kv}-{k}-G", (kv, k, "G"), lambda v, K, G: v-K-4*G/3),
        Relation(f"{e}-{k}-{nu}", (e, k, nu), lambda E, K, n: E-3*K*(1-2*n)),
        Relation(f"G-{k}-{nu}", ("G", k, nu), lambda G, K, n: G-3*K*(1-2*n)/(2*(1+n))),
        Relation(f"{lam}-{k}-{nu}", (lam, k, nu), lambda l, K, n: l-3*K*n/(1+n)),
        Relation(f"{kv}-{k}-{nu}", (kv, k, nu), lambda v, K, n: v-3*K*(1-n)/(1+n)),
        Relation(f"{e}-G-{nu}", (e, "G", nu), lambda E, G, n: E-2*G*(1+n)),
        Relation(f"{lam}-G-{nu}", (lam, "G", nu), lambda l, G, n: l-2*G*n/(1-2*n)),
    ]
    # Complete MRST's pairwise elasticity basis.  These identities are
    # intentionally redundant: poroParams uses whichever relation has one
    # unknown, so callers may supply any physically independent pair.
    relations.extend([
        Relation(f"{e}-{k}-{lam}", (e, k, lam), lambda E, K, l: E-9*K*(K-l)/(3*K-l)),
        Relation(f"{nu}-{k}-{lam}", (nu, k, lam), lambda n, K, l: n-l/(3*K-l)),
        Relation(f"{kv}-{k}-{lam}", (kv, k, lam), lambda v, K, l: v-(3*K-2*l)),
        Relation(f"G-{k}-{lam}", ("G", k, lam), lambda G, K, l: G-1.5*(K-l)),
        Relation(f"{k}-{e}-G", (k, e, "G"), lambda K, E, G: K-E*G/(3*(3*G-E))),
        Relation(f"{lam}-{e}-G", (lam, e, "G"), lambda l, E, G: l-G*(E-2*G)/(3*G-E)),
        Relation(f"{nu}-{e}-G", (nu, e, "G"), lambda n, E, G: n-(E/(2*G)-1)),
        Relation(f"{kv}-{e}-G", (kv, e, "G"), lambda v, E, G: v-G*(4*G-E)/(3*G-E)),
        Relation(f"{lam}-{k}-{e}", (lam, k, e), lambda l, K, E: l-3*K*(3*K-E)/(9*K-E)),
        Relation(f"{nu}-{k}-{e}", (nu, k, e), lambda n, K, E: n-(3*K-E)/(6*K)),
        Relation(f"{kv}-{k}-{e}", (kv, k, e), lambda v, K, E: v-3*K*(3*K+E)/(9*K-E)),
        Relation(f"G-{k}-{e}", ("G", k, e), lambda G, K, E: G-3*K*E/(9*K-E)),
        Relation(f"{k}-{lam}-{nu}", (k, lam, nu), lambda K, l, n: K-l*(1+n)/(3*n)),
        Relation(f"{e}-{lam}-{nu}", (e, lam, nu), lambda E, l, n: E-l*(1+n)*(1-2*n)/n),
        Relation(f"{kv}-{lam}-{nu}", (kv, lam, nu), lambda v, l, n: v-l*(1-n)/n),
        Relation(f"G-{lam}-{nu}", ("G", lam, nu), lambda G, l, n: G-l*(1-2*n)/(2*n)),
        Relation(f"{k}-G-{nu}", (k, "G", nu), lambda K, G, n: K-2*G*(1+n)/(3*(1-2*n))),
        Relation(f"{kv}-G-{nu}", (kv, "G", nu), lambda v, G, n: v-2*G*(1-n)/(1-2*n)),
        Relation(f"{k}-{e}-{nu}", (k, e, nu), lambda K, E, n: K-E/(3*(1-2*n))),
        Relation(f"{lam}-{e}-{nu}", (lam, e, nu), lambda l, E, n: l-E*n/((1+n)*(1-2*n))),
        Relation(f"{kv}-{e}-{nu}", (kv, e, nu), lambda v, E, n: v-E*(1-n)/((1+n)*(1-2*n))),
        Relation(f"G-{e}-{nu}", ("G", e, nu), lambda G, E, n: G-E/(2*(1+n))),
    ])
    return relations


def _relations(phi: float, uniform: bool) -> list[Relation]:
    r = _elastic_relations() + _elastic_relations("_u")
    r.extend([
        Relation("R-S_sigma", ("R", "S_sigma"), lambda R, S: S-1/R),
        Relation("M-S_epsilon", ("M", "S_epsilon"), lambda M, S: S-1/M),
        Relation("S_epsilon-S_sigma-K-H", ("S_epsilon", "S_sigma", "K", "H"), lambda Se, Ss, K, H: Se-(Ss-K/H**2)),
        Relation("B-R-H", ("B", "R", "H"), lambda B, R, H: B-R/H),
        Relation("alpha-K-H", ("alpha", "K", "H"), lambda a, K, H: a-K/H),
        Relation("K_u-K-alpha-M", ("K_u", "K", "alpha", "M"), lambda Ku, K, a, M: Ku-K-a*a*M),
        Relation("B-alpha-M-K_u", ("B", "alpha", "M", "K_u"), lambda B, a, M, Ku: B-a*M/Ku),
        Relation("K_s-K-alpha", ("K_s", "K", "alpha"), lambda Ks, K, a: a-(1-K/Ks)),
        Relation("K_s-K-H", ("K_s", "K", "H"), lambda Ks, K, H: 1/Ks-(1/K-1/H)),
        Relation("K_p-alpha-K", ("K_p", "alpha", "K"), lambda Kp, a, K: 1/Kp-a/(phi*K)),
        Relation("S_epsilon-S_sigma-alpha-K", ("S_epsilon", "S_sigma", "alpha", "K"), lambda Se, Ss, a, K: Se-Ss+a*a/K),
        Relation("S_epsilon-S_sigma-alpha-B", ("S_epsilon", "S_sigma", "alpha", "B"), lambda Se, Ss, a, B: Se-Ss*(1-a*B)),
        Relation("S_epsilon-alpha-K_u-B", ("S_epsilon", "alpha", "K_u", "B"), lambda Se, a, Ku, B: Se-a/(Ku*B)),
        Relation("S_epsilon-S_sigma-K-K_u", ("S_epsilon", "S_sigma", "K", "K_u"), lambda Se, Ss, K, Ku: Se/Ss-K/Ku),
        Relation("S_epsilon-alpha-K_u-K", ("S_epsilon", "alpha", "K_u", "K"), lambda Se, a, Ku, K: Se-a*a/(Ku-K)),
        Relation("S_sigma-alpha-K-B", ("S_sigma", "alpha", "K", "B"), lambda Ss, a, K, B: Ss-a/(K*B)),
        Relation("K_u-K-alpha-B", ("K_u", "K", "alpha", "B"), lambda Ku, K, a, B: 1/Ku-(1-a*B)/K),
        Relation("nu_u-nu-alpha-B", ("nu_u", "nu", "alpha", "B"), lambda n_u, n, a, B: n_u-(3*n+a*B*(1-2*n))/(3-a*B*(1-2*n))),
        Relation("eta-nu-alpha", ("eta", "nu", "alpha"), lambda eta, n, a: eta-(1-2*n)*a/(2*(1-n))),
        Relation("gamma-B-nu_u", ("gamma", "B", "nu_u"), lambda g, B, n: g-B*(1+n)/(3*(1-n))),
        Relation("c_m-alpha-K_v", ("c_m", "alpha", "K_v"), lambda c, a, Kv: c-a/Kv),
        Relation("beta-K_p-K_phi", ("beta", "K_p", "K_phi"), lambda b, Kp, Kphi: b-(1-Kp/Kphi)),
        Relation("S_gamma-K_f-K_phi", ("S_gamma", "K_f", "K_phi"), lambda Sg, Kf, Kphi: Sg-phi*(1/Kf-1/Kphi)),
        Relation("K_phi-alpha-K-B-K_f", ("K_phi", "alpha", "K", "B", "K_f"), lambda Kphi, a, K, B, Kf: 1/Kphi+(a/(K*B)-phi/Kf-a/K)/phi),
        Relation("K_phi-K_s-K-B-K_f", ("K_phi", "K_s", "K", "B", "K_f"), lambda Kphi, Ks, K, B, Kf: 1/Kphi+((1/K-1/Ks)*(1/B-1)-phi/Kf)/phi),
        Relation("S_sigma-K-K_s-K_f-K_phi", ("S_sigma", "K", "K_s", "K_f", "K_phi"), lambda Ss, K, Ks, Kf, Kphi: Ss-((1/K-1/Ks)+phi*(1/Kf-1/Kphi))),
        Relation("S_epsilon-K-K_s-K_f-K_phi", ("S_epsilon", "K", "K_s", "K_f", "K_phi"), lambda Se, K, Ks, Kf, Kphi: Se-((1-K/Ks)/Ks+phi*(1/Kf-1/Kphi))),
        Relation("S_gamma-S_epsilon-alpha-K", ("S_gamma", "S_epsilon", "alpha", "K"), lambda Sg, Se, a, K: Sg-Se+a*(1-a)/K),
        Relation("S-S_epsilon-nu-nu_u", ("S", "S_epsilon", "nu", "nu_u"), lambda S, Se, n, nu: S-Se*((1-nu)/(1-n))*((1-2*n)/(1-2*nu))),
        Relation("S-S_sigma-eta-B", ("S", "S_sigma", "eta", "B"), lambda S, Ss, eta, B: S-Ss*(1-4*eta*B/3)),
        Relation("S-S_epsilon-K_vu-K_v", ("S", "S_epsilon", "K_vu", "K_v"), lambda S, Se, Kvu, Kv: S-Se*Kvu/Kv),
        Relation("S-S_sigma-nu-nu_u", ("S", "S_sigma", "nu", "nu_u"), lambda S, Ss, n, nu: S-Ss*((1-nu)/(1+nu))*((1+n)/(1-n))),
        Relation("S-S_sigma-K-K_u-K_v-K_vu", ("S", "S_sigma", "K", "K_u", "K_v", "K_vu"), lambda S, Ss, K, Ku, Kv, Kvu: S-Ss*(K/Ku)*(Kvu/Kv)),
        Relation("K_vu-K_u-nu_u", ("K_vu", "K_u", "nu_u"), lambda Kvu, Ku, n: 1/Kvu-(1+n)/(3*Ku*(1-n))),
        Relation("S-alpha-K_v-gamma", ("S", "alpha", "K_v", "gamma"), lambda S, a, Kv, g: S-a/(Kv*g)),
        Relation("gamma-eta-G-S", ("gamma", "eta", "G", "S"), lambda gamma, eta, G, S: gamma-eta/(G*S)),
        Relation("H-K_p", ("H", "K_p"), lambda H, Kp: 1/H-phi/Kp),
        Relation("c_m-eta-G", ("c_m", "eta", "G"), lambda cm, eta, G: cm-eta/G),
        Relation("B-K-K_s-K_f-K_phi", ("B", "K", "K_s", "K_f", "K_phi"), lambda B, K, Ks, Kf, Kphi: B-(1/K-1/Ks)/(1/K-1/Ks+phi*(1/Kf-1/Kphi))),
        Relation("B-K-K_u-K_s", ("B", "K", "K_u", "K_s"), lambda B, K, Ku, Ks: B-(1-K/Ku)/(1-K/Ks)),
        Relation("B-nu-nu_u-alpha", ("B", "nu", "nu_u", "alpha"), lambda B, n, nu, a: B-3*(nu-n)/(a*(1+nu)*(1-2*n))),
        Relation("M-G-nu-nu_u-alpha", ("M", "G", "nu", "nu_u", "alpha"), lambda M, G, n, nu, a: M-2*G*(nu-n)/(a*a*(1-2*nu)*(1-2*n))),
    ])
    if uniform:
        r.append(Relation("uniform-grains", ("K_s", "K_phi"), lambda Ks, Kphi: 1/Ks-1/Kphi))
    return r


def _scale(name: str) -> float:
    if name in _MODULI:
        return 1.0e7
    if name in _STORATIVITIES:
        return 1.0e-7
    return 1.0


def _solve_one(relation: Relation, unknown: str, values: Mapping[str, float]) -> float:
    pos = relation.args.index(unknown)

    def fun(x: np.ndarray) -> np.ndarray:
        args = [values[a] if a != unknown else float(x[0]) for a in relation.args]
        raw = float(relation.residual(*args))
        return np.array([raw])

    positive_guesses = tuple(
        multiplier * 10.0**power
        for power in range(-3, 5)
        for multiplier in (1.0, 2.0, 5.0)
    )
    guesses = positive_guesses + tuple(-value for value in positive_guesses)
    candidates: list[tuple[float, float]] = []
    for guess in guesses:
        try:
            delta = max(abs(guess) * 1.0e-4, 1.0e-8)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                root = root_scalar(
                    lambda x: float(fun(np.array([x]))[0]),
                    x0=guess,
                    x1=guess + delta,
                    method="secant",
                    xtol=1.0e-12,
                    maxiter=200,
                )
            trial = float(root.root)
            score = abs(float(fun(np.array([trial]))[0]))
            if (root.converged and score <= 1.0e-10
                    and math.isfinite(trial)
                    and _physically_admissible(unknown, trial, values)
                    and _mrst_branch_admissible(relation, unknown, trial, values)):
                candidates.append((score, trial))
        except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
            pass

    # A tightly checked least-squares fallback handles smooth relations for
    # which the secant happens to encounter a pole from every seed.
    if not candidates:
        for guess in guesses:
            candidate = least_squares(fun, np.array([guess]), xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=500)
            score = abs(float(fun(candidate.x)[0]))
            trial = float(candidate.x[0])
            if (score <= 1.0e-10 and math.isfinite(trial)
                    and _physically_admissible(unknown, trial, values)
                    and _mrst_branch_admissible(relation, unknown, trial, values)):
                candidates.append((score, trial))
    if not candidates:
        raise ValueError(
            f"Could not find a physically admissible root for {unknown!r}"
        )
    minimum_score = min(item[0] for item in candidates)
    roots = [item for item in candidates
             if item[0] <= max(1.0e-8, minimum_score * 100.0)]
    # Match MRST's ascending small-magnitude fzero seeds when more than one
    # mechanically stable branch exists (notably the E/K_v basis).
    best = min(roots, key=lambda item: abs(item[1]))
    if not math.isfinite(best[1]) or best[0] > 1e-8:
        raise ValueError(f"Could not solve relation {relation.name!r} for {unknown!r}")
    return best[1]


def _mrst_branch_admissible(
    relation: Relation, name: str, value: float, known: Mapping[str, float]
) -> bool:
    """Mirror fzero's branch for the two rational E/Kv relations.

    These relations have two positive roots.  MATLAB's scalar ``fzero``
    reaches the larger root from poroParams' prescribed seed sequence,
    whereas a secant search followed by minimum-magnitude selection reaches
    the smaller one.  The inequalities encode only that observed MRST branch
    and are covered by the exhaustive MATLAB fixture.
    """
    if relation.name in {"K_v-E-G", "K_vu-E_u-G"} and name == "G":
        e_name = "E_u" if "E_u" in relation.args else "E"
        return value > known[e_name] / 2.0
    if relation.name in {"K_v-K-E", "K_vu-K_u-E_u"} and name in {"K", "K_u"}:
        e_name = "E_u" if "E_u" in relation.args else "E"
        return value > known[e_name] / 3.0
    return True


def _physically_admissible(
    name: str, value: float, known: Mapping[str, float]
) -> bool:
    """Reject algebraic roots that violate stable isotropic elasticity."""
    strictly_positive = _MODULI.difference({"lambda", "lambda_u"})
    if name in strictly_positive and value <= 0.0:
        return False
    if name in {"nu", "nu_u"} and not (-1.0 < value < 0.5):
        return False
    # Do not reject algebraically valid auxetic branches here.  MRST's
    # ``solve_relation`` accepts the first root reached from its ascending
    # small-magnitude fzero seeds.  In particular, the (E, lambda) basis can
    # yield K < lambda.  Reproducing that choice is necessary for exact input
    # basis parity; inconsistent redundant relations remain visible through
    # the returned residual vector and warning, just as in MATLAB.
    return True


def solve_poro_params(
    phi: float,
    uniform: bool = False,
    parameters: Mapping[str, float] | None = None,
    *,
    residual_tolerance: float = 1.0e-6,
    **known: float,
) -> tuple[Dict[str, float], list[tuple[int, str]], np.ndarray, list[Relation]]:
    """MRST-style parameter-relation solver with diagnostic outputs."""
    if not 0.0 < float(phi) < 1.0:
        raise ValueError("phi must lie strictly between zero and one")
    supplied = dict(parameters or {})
    supplied.update(known)
    unknown_names = sorted(set(supplied).difference(PARAMETER_NAMES))
    if unknown_names:
        raise KeyError(f"Unknown poroelastic parameter(s): {', '.join(unknown_names)}")
    values = {name: math.nan for name in PARAMETER_NAMES}
    for name, value in supplied.items():
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        values[name] = value / _scale(name)

    relations = _relations(float(phi), bool(uniform))
    history: list[tuple[int, str]] = []
    while True:
        if uniform:
            if math.isfinite(values["K_s"]) and math.isnan(values["K_phi"]):
                values["K_phi"] = values["K_s"]
                history.append((len(relations) - 1, "K_phi"))
            elif math.isfinite(values["K_phi"]) and math.isnan(values["K_s"]):
                values["K_s"] = values["K_phi"]
                history.append((len(relations) - 1, "K_s"))
        candidates: dict[str, tuple[int, Relation]] = {}
        for index, relation in enumerate(relations):
            missing = [name for name in relation.args if math.isnan(values[name])]
            if len(missing) == 1:
                candidates.setdefault(missing[0], (index, relation))
        if not candidates:
            break
        progressed = False
        for name, (index, relation) in candidates.items():
            try:
                values[name] = _solve_one(relation, name, values)
            except (ValueError, ZeroDivisionError, FloatingPointError):
                continue
            history.append((index, name))
            progressed = True
        if not progressed:
            break

    residuals = np.full(len(relations), np.nan)
    for index, relation in enumerate(relations):
        args = [values[name] for name in relation.args]
        if all(math.isfinite(v) for v in args):
            try:
                raw = float(relation.residual(*args))
                residuals[index] = raw / max(1.0, *(abs(v) for v in args))
            except (ZeroDivisionError, FloatingPointError):
                pass
    finite = residuals[np.isfinite(residuals)]
    if finite.size and np.max(np.abs(finite)) > residual_tolerance:
        warnings.warn(
            "Some poroelastic relation residuals did not vanish; input parameters may be inconsistent",
            RuntimeWarning,
            stacklevel=2,
        )
    result = {name: value * _scale(name) if math.isfinite(value) else math.nan for name, value in values.items()}
    return result, history, residuals, relations


def poro_params(*args: float, return_details: bool = False, **kwargs: float):
    """Compute poroelastic constants using MRST or legacy PyMRST syntax.

    ``poro_params(E, nu, alpha, M)`` is the legacy convenience form.
    ``poro_params(phi, uniform=False, E=..., nu=..., alpha=..., M=...)`` is
    the MRST-style form and can accept any supported consistent subset.
    """
    if len(args) == 4 and not kwargs:
        E, nu, alpha, M = map(float, args)
        K = E / (3.0 * (1.0 - 2.0 * nu))
        G = E / (2.0 * (1.0 + nu))
        K_u = K + alpha**2 * M
        B = alpha * M / K_u
        nu_u = (3.0 * K_u - 2.0 * G) / (2.0 * (3.0 * K_u + G))
        return {"K": K, "G": G, "K_u": K_u, "B": B, "nu_u": nu_u}
    if not args:
        raise TypeError("MRST-style poro_params requires porosity as the first argument")
    if len(args) > 2:
        raise TypeError("MRST-style syntax is poro_params(phi, uniform=False, **parameters)")
    phi = float(args[0])
    uniform = bool(args[1]) if len(args) == 2 else bool(kwargs.pop("uniform", False))
    details = solve_poro_params(phi, uniform, kwargs)
    return details if return_details else details[0]


def consolidation_coefficient(perm: float, mu_f: float, E: float, nu: float, alpha: float, M: float) -> float:
    """Mandel consolidation coefficient."""
    p = poro_params(E, nu, alpha, M)
    G, B, nu_u = p["G"], p["B"], p["nu_u"]
    return (2.0 * perm * B**2 * G * (1.0 - nu) * (1.0 + nu_u) ** 2) / (
        9.0 * mu_f * (1.0 - nu_u) * (nu_u - nu)
    )


__all__ = ["PARAMETER_NAMES", "Relation", "consolidation_coefficient", "poro_params", "solve_poro_params"]
