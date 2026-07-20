"""Global elastoplastic mechanics solve on the VEM displacement space."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.modules.ad_mechanics.mechanics import MechanicModel
from pymrst.modules.ad_mechanics.plasticity import (
    PlasticityState,
    empty_plasticity_state,
    von_mises_return_map,
)
from pymrst.modules.vemmech import ElasticBC, enu2c, enu2c_2d


@dataclass
class ElastoplasticStep:
    """One converged elastoplastic load step."""

    load_factor: float
    xd: np.ndarray
    displacement: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    plastic_state: PlasticityState
    residual_norm: float
    iterations: int


@dataclass
class ElastoplasticResult:
    """History from a global elastoplastic Newton solve."""

    model: MechanicModel
    steps: list[ElastoplasticStep]

    @property
    def last(self) -> ElastoplasticStep:
        return self.steps[-1]


def _elastic_stress(C: np.ndarray, strain: np.ndarray) -> np.ndarray:
    nlin = strain.shape[1]
    return np.einsum("cij,cj->ci", C.reshape(-1, nlin, nlin), strain)


def _plastic_stress_for_grid(g, stress6: np.ndarray) -> np.ndarray:
    if g.griddim == 2:
        return stress6[:, [0, 1, 5]]
    return stress6


def _free_displacement(model: MechanicModel, xd: np.ndarray) -> np.ndarray:
    u = model.operators.u_dir.copy()
    u[~model.operators.is_dir] = xd
    return u


def _plastic_residual(
    model: MechanicModel,
    xd: np.ndarray,
    committed: PlasticityState,
    young: float,
    poisson: float,
    yield_stress: float,
    hardening: float,
    load_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PlasticityState]:
    op = model.operators
    free = ~op.is_dir
    u = _free_displacement(model, xd)
    strain = model.strain_full(u)
    stress6, trial_state, _ = von_mises_return_map(
        strain, young, poisson, yield_stress, hardening, committed
    )
    stress = _plastic_stress_for_grid(model.G, stress6)
    elastic = _elastic_stress(model.C, strain)
    correction = (stress - elastic) * model.G.cells.volumes[:, None]
    internal = model.operators.A @ xd
    internal += model._S_free_dir @ op.u_dir[op.is_dir]
    internal += (op.strain[:, free].T @ correction.reshape(-1))
    external = load_factor * model._load[free]
    return internal - external, strain, stress, trial_state


def _numerical_jacobian(fun, x: np.ndarray, r0: np.ndarray) -> np.ndarray:
    n = x.size
    J = np.zeros((r0.size, n))
    scale = np.maximum(np.abs(x), 1.0)
    for j in range(n):
        h = 1e-7 * scale[j]
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (fun(xp) - fun(xm)) / (2.0 * h)
    return J


def solve_elastoplastic_vem(
    g,
    el_bc: ElasticBC,
    *,
    young: float = 1.0e9,
    poisson: float = 0.25,
    yield_stress: float = 1.0e6,
    hardening: float = 0.0,
    load_factors: Sequence[float] = (1.0,),
    alpha_scaling: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 12,
) -> ElastoplasticResult:
    """Solve a small-strain elastoplastic VEM mechanics problem.

    The global residual is the elastic VEM equilibrium plus the cell-wise
    plastic stress correction from a von-Mises radial return map. Plastic state
    is committed only after each load step converges.
    """
    if g.griddim == 2:
        C = enu2c_2d(
            np.full(g.cells.num, young, dtype=float),
            np.full(g.cells.num, poisson, dtype=float),
        )
    elif g.griddim == 3:
        C = enu2c(
            np.full(g.cells.num, young, dtype=float),
            np.full(g.cells.num, poisson, dtype=float),
        )
    else:
        raise ValueError("elastoplastic VEM solve supports 2D and 3D grids")
    model = MechanicModel(
        g, C, alpha=np.zeros(g.cells.num), el_bc=el_bc,
        load=None, alpha_scaling=alpha_scaling,
    )
    committed = empty_plasticity_state(g.cells.num)
    xd = np.zeros(model.n_free)
    steps: list[ElastoplasticStep] = []

    for lf in load_factors:
        if steps:
            xd = steps[-1].xd.copy()
        else:
            xd = spla.spsolve(
                model.operators.A.tocsc(),
                lf * model._load[~model.operators.is_dir],
            )
        converged_state = committed
        strain = np.zeros((g.cells.num, 6 if g.griddim == 3 else 3))
        stress = np.zeros_like(strain)
        res_norm = np.inf
        it_used = 0
        for it in range(1, max_iter + 1):
            r, strain, stress, trial_state = _plastic_residual(
                model, xd, committed, young, poisson, yield_stress,
                hardening, float(lf),
            )
            res_norm = float(np.linalg.norm(r, ord=np.inf))
            ref = max(
                float(np.linalg.norm(
                    lf * model._load[~model.operators.is_dir], ord=np.inf
                )),
                1.0,
            )
            it_used = it
            if res_norm <= tol * ref:
                converged_state = trial_state
                break

            def f(z):
                return _plastic_residual(
                    model, z, committed, young, poisson, yield_stress,
                    hardening, float(lf),
                )[0]

            J = _numerical_jacobian(f, xd, r)
            try:
                step = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(J, -r, rcond=None)[0]

            base = res_norm
            alpha = 1.0
            while alpha > 1e-4:
                cand = xd + alpha * step
                rc = f(cand)
                if np.linalg.norm(rc, ord=np.inf) <= (1.0 - 1e-4 * alpha) * base:
                    xd = cand
                    break
                alpha *= 0.5
            else:
                xd = xd + step
        else:
            raise RuntimeError(
                f"elastoplastic Newton did not converge for load factor {lf}"
            )

        committed = converged_state
        u = _free_displacement(model, xd)
        steps.append(ElastoplasticStep(
            load_factor=float(lf),
            xd=xd.copy(),
            displacement=u.reshape(-1, g.griddim),
            strain=strain.copy(),
            stress=stress.copy(),
            plastic_state=committed,
            residual_norm=res_norm,
            iterations=it_used,
        ))

    return ElastoplasticResult(model=model, steps=steps)
