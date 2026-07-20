"""Backend Newton helpers, including Torch autograd Jacobians."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pymrst.backends import ArrayBackend, get_backend


@dataclass
class NewtonReport:
    converged: bool
    iterations: int
    residual_norms: list[float] = field(default_factory=list)
    linear_iterations: list[int] = field(default_factory=list)
    linear_residuals: list[float] = field(default_factory=list)


@dataclass
class LinearSolveReport:
    converged: bool
    iterations: int
    residual_norm: float


def torch_newton(
    residual: Callable,
    x0,
    *,
    backend: str | ArrayBackend = "torch",
    device: str | None = None,
    dtype: str = "float64",
    max_iterations: int = 15,
    tolerance: float = 1e-9,
    damping: float = 1.0,
):
    """Solve ``residual(x) = 0`` using Torch autograd dense Jacobians."""
    be = get_backend(backend, device=device, dtype=dtype)
    if not be.is_torch:
        raise ValueError("torch_newton requires backend='torch'")
    torch = be.torch
    x = be.asarray(x0).clone().detach()
    report = NewtonReport(converged=False, iterations=0)

    for it in range(1, max_iterations + 1):
        x = x.clone().detach().requires_grad_(True)
        r = residual(x)
        norm = float(torch.linalg.norm(r.detach()).cpu())
        report.residual_norms.append(norm)
        if norm <= tolerance:
            report.converged = True
            report.iterations = it - 1
            return x.detach(), report
        jac = torch.autograd.functional.jacobian(residual, x, vectorize=True)
        dx = torch.linalg.solve(jac, -r)
        x = (x + damping * dx).detach()
        report.iterations = it

    final = float(torch.linalg.norm(residual(x).detach()).cpu())
    report.residual_norms.append(final)
    report.converged = final <= tolerance
    return x.detach(), report


def torch_matrix_free_gmres(
    matvec: Callable,
    rhs,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 80,
    restart: int | None = None,
):
    """Solve ``A x = rhs`` with restarted GMRES using Torch matvecs.

    The matrix is never assembled. ``matvec(v)`` must return ``A @ v`` on the
    same Torch device/dtype as ``rhs``. This is the scalable Newton-linear-solve
    path for backend experiments where forming a dense autograd Jacobian would
    dominate memory.
    """
    import torch
    b = rhs.detach()
    x = torch.zeros_like(b)
    bnorm = torch.linalg.norm(b)
    if float(bnorm.cpu()) == 0.0:
        return x, LinearSolveReport(True, 0, 0.0)
    tol_abs = tolerance * float(bnorm.cpu())
    n = b.numel()
    restart = min(int(restart or max_iterations), int(max_iterations), n)
    total_iterations = 0
    final_norm = float(bnorm.cpu())

    while total_iterations < max_iterations:
        r = b - matvec(x)
        beta = torch.linalg.norm(r)
        beta_f = float(beta.cpu())
        final_norm = beta_f
        if beta_f <= tol_abs:
            return x, LinearSolveReport(True, total_iterations, beta_f)

        v_basis = [r / beta]
        h = torch.zeros(
            (restart + 1, restart),
            dtype=b.dtype,
            device=b.device,
        )
        steps = 0
        for j in range(restart):
            w = matvec(v_basis[j])
            for i in range(j + 1):
                h[i, j] = torch.dot(v_basis[i].reshape(-1), w.reshape(-1))
                w = w - h[i, j] * v_basis[i]
            h[j + 1, j] = torch.linalg.norm(w)
            if float(h[j + 1, j].cpu()) > 0.0:
                v_basis.append(w / h[j + 1, j])
            steps = j + 1
            total_iterations += 1

            e1 = torch.zeros(j + 2, dtype=b.dtype, device=b.device)
            e1[0] = beta
            sol = torch.linalg.lstsq(h[:j + 2, :j + 1], e1).solution
            v_mat = torch.stack(v_basis[:j + 1], dim=1)
            candidate = x + v_mat @ sol
            res_norm = float(torch.linalg.norm(b - matvec(candidate)).cpu())
            final_norm = res_norm
            if res_norm <= tol_abs or total_iterations >= max_iterations:
                return candidate, LinearSolveReport(
                    res_norm <= tol_abs,
                    total_iterations,
                    res_norm,
                )
            if len(v_basis) == j + 1:
                return candidate, LinearSolveReport(False, total_iterations, res_norm)

        e1 = torch.zeros(steps + 1, dtype=b.dtype, device=b.device)
        e1[0] = beta
        sol = torch.linalg.lstsq(h[:steps + 1, :steps], e1).solution
        v_mat = torch.stack(v_basis[:steps], dim=1)
        x = x + v_mat @ sol

    return x, LinearSolveReport(False, total_iterations, final_norm)


def torch_jvp_newton_step(
    residual: Callable,
    x,
    rhs,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 80,
    restart: int | None = None,
    left_scale=None,
    right_scale=None,
):
    """Solve one Newton system with matrix-free Torch JVP products.

    Optional diagonal scaling solves ``D_l J D_r y = D_l rhs`` and returns
    ``D_r y``. This is row/variable scaling, not an approximate inverse, so it
    preserves the Newton correction while improving GMRES magnitudes.
    """
    import torch
    x0 = x.detach()
    lscale = left_scale.detach() if left_scale is not None else None
    rscale = right_scale.detach() if right_scale is not None else None

    def matvec(v):
        tangent = v if rscale is None else rscale * v
        _, jv = torch.autograd.functional.jvp(
            residual,
            (x0,),
            (tangent,),
            create_graph=False,
            strict=False,
        )
        out = jv.detach()
        return out if lscale is None else lscale * out

    scaled_rhs = rhs.detach() if lscale is None else lscale * rhs.detach()
    y, report = torch_matrix_free_gmres(
        matvec,
        scaled_rhs,
        tolerance=tolerance,
        max_iterations=max_iterations,
        restart=restart,
    )
    return (y if rscale is None else rscale * y), report


def finite_difference_jacobian(residual: Callable[[np.ndarray], np.ndarray], x, eps=1e-7):
    """Small dense finite-difference Jacobian for backend validation tests."""
    x = np.asarray(x, dtype=float)
    r0 = np.asarray(residual(x), dtype=float)
    jac = np.zeros((r0.size, x.size))
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        jac[:, i] = (np.asarray(residual(xp)) - np.asarray(residual(xm))) / (2 * eps)
    return jac
