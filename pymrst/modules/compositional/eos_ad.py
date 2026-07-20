r"""Differentiating the flash (MRST `getPhaseFractionAsADI`).

`eos.py` solves the flash - given `(p, T, z)` it returns the phase split
`x, y, L` and the compressibility factors - but it solves it on *values*: there
is no derivative anywhere in the answer. A fully-implicit compositional flow
model cannot use that. Its primary variables are pressure and the overall
composition `z`, and every property it assembles (mass fractions, saturations,
densities) is a function of the flash result, so it needs

    dx/dp, dx/dz,  dy/dp, dy/dz,  dL/dp, dL/dz

to build a Jacobian. The flash is an *implicit* function of `(p, T, z)` - it is
defined by the equilibrium conditions, not by a closed form - so these come from
the **implicit function theorem**, exactly as MRST's `getPhaseFractionAsADI`
does.

Let `F(s; u) = 0` be the equilibrium system, with **secondary** unknowns
`s = (L, x_1..x_n, y_1..y_n)` (what the flash solved for) and **primary**
variables `u = (p, T, z_1..z_n)` (what the flow model differentiates against).
`F` is:

- `z_i - L x_i - (1-L) y_i = 0`      (component material balance)
- `f_i^V - f_i^L = 0`                (equal fugacity)
- `sum_i (x_i - y_i) = 0`            (phase closure)

At the solution, `dF/ds · ds/du + dF/du = 0`, so

    ds/du = -(dF/ds)^{-1} (dF/du).

The two Jacobians are obtained by evaluating `F` twice under PyMRST's
forward-mode AD: once with `u` seeded (giving `dF/du`) and once with `s` seeded
(giving `dF/ds`). Because each cell's equilibrium couples only that cell's own
`s` and `u`, both Jacobians are block-diagonal by cell, so the solve is a batch
of small dense `(2n+1) x (2n+1)` systems rather than one global sparse LU - the
same mathematics as MRST, arranged for NumPy.

The fugacity itself has a subtlety the value-only `eos.py` sidesteps: the
compressibility `Z` is a *root of the cubic* and carries no automatic
derivative. MRST's `setZDerivatives` differentiates the cubic implicitly -
`dZ = -(dE2 Z^2 + dE1 Z + dE0)/(3Z^2 + 2 E2 Z + E1)` - and that is done here too,
so `Z`'s dependence on `(p, T, x)` propagates into the fugacity. The A/B mixing
rule, cubic coefficients and fugacity formula are the same ones `eos.py`
already matches MRST on; this module only re-expresses them so they carry
derivatives.

Layout note (`fullZ`): the derivative columns are `[p, T, z_1, ..., z_n]`, all
`n` compositions treated as independent, matching MRST's `getPhaseFractionAsADI`
default. The `z` columns therefore do not assume `sum z = 1`; the flow model
imposes that separately (`z_n = 1 - sum_{i<n} z_i`) via the chain rule.
"""

from typing import Any, List, Tuple

import numpy as np

from pymrst.modules.compositional.eos import (
    OMEGA_A,
    OMEGA_B,
    Mixture,
    _pr_m,
    flash,
)
from pymrst.utils.adi import ADI, _diag_mul, init_variables_adi, value

SQRT2 = np.sqrt(2.0)
M1 = 1.0 + SQRT2                          # PR cubic coefficients (eosA/eosB)
M2 = 1.0 - SQRT2


def _is_adi(x: Any) -> bool:
    return isinstance(x, ADI) or x.__class__.__name__ == "GenericAD"


def _make_ad_like(template: Any, val: np.ndarray, jac: list) -> Any:
    if template.__class__.__name__ == "GenericAD":
        from pymrst.modules.ad_core.backends.diagonal.generic_ad import GenericAD

        return GenericAD(val, jac)
    return ADI(val, jac)


def _scale_block(coeff: np.ndarray, block):
    if hasattr(block, "left_scale"):
        return block.left_scale(coeff)
    return _diag_mul(coeff, block)


def _log(x: Any) -> Any:
    if _is_adi(x):
        from pymrst.utils.adi import ad_log
        return ad_log(x)
    return np.log(x)


def _exp(x: Any) -> Any:
    if _is_adi(x):
        from pymrst.utils.adi import ad_exp
        return ad_exp(x)
    return np.exp(x)


def _mixing_parameters(mix: Mixture, p: Any, T: Any
                       ) -> Tuple[List[List[Any]], List[Any]]:
    r"""Per-component `sqrt(A_i)` cross-terms `A_ij` and covolumes `B_i`.

    `A_ij = sqrt(A_i A_j)(1 - k_ij)` with `A_i = omegaA·alpha_i·Pr_i/Tr_i^2` and
    `B_i = omegaB·Pr_i/Tr_i`, the dimensionless PR parameters (`Pr = p/pc`,
    `Tr = T/Tc`). Works on values or ADI interchangeably.
    """
    m = _pr_m(mix.acentric, mix.eos_type)
    kij = mix.bin_interaction()
    n = mix.n
    s_ai: List[Any] = []
    bi: List[Any] = []
    for i in range(n):
        pr = p * (1.0 / float(mix.Pcrit[i]))
        tr = T * (1.0 / float(mix.Tcrit[i]))
        o_a = OMEGA_A * (1.0 + m[i] * (1.0 - tr ** 0.5)) ** 2
        s_ai.append((o_a * pr) ** 0.5 / tr)
        bi.append(OMEGA_B * pr / tr)
    a_ij: List[List[Any]] = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            aij = s_ai[i] * s_ai[j] * (1.0 - float(kij[i, j]))
            a_ij[i][j] = aij
            a_ij[j][i] = aij
    return a_ij, bi


def _phase_mix(comp: List[Any], a_ij: List[List[Any]], bi: List[Any]
               ) -> Tuple[List[Any], Any, Any]:
    r"""Mixture attraction `A = sum_ij x_i A_ij x_j`, covolume `B = sum_i x_i B_i`
    and the per-component sums `S_i = sum_j A_ij x_j` the fugacity needs."""
    n = len(comp)
    b = comp[0] * bi[0]
    for i in range(1, n):
        b = b + comp[i] * bi[i]
    xj_aij = [[a_ij[i][j] * comp[j] for j in range(n)] for i in range(n)]
    s_i = []
    for i in range(n):
        si = xj_aij[i][0]
        for j in range(1, n):
            si = si + xj_aij[i][j]
        s_i.append(si)
    a = None
    for i in range(n):
        for j in range(i, n):
            tmp = comp[j] * xj_aij[j][i]
            tmp = tmp if i == j else tmp * 2.0
            a = tmp if a is None else a + tmp
    return s_i, a, b


def _cubic_coeffs(a: Any, b: Any) -> Tuple[Any, Any, Any]:
    """`Z^3 + E2 Z^2 + E1 Z + E0 = 0` (PR): `E2 = B-1`, `E1 = A-3B^2-2B`,
    `E0 = -(AB - B^2 - B^3)`."""
    e2 = (M1 + M2 - 1.0) * b - 1.0
    e1 = a - (M1 + M2 - M1 * M2) * b ** 2 - (M1 + M2) * b
    e0 = -(a * b + (M1 * M2) * b ** 2 * (b + 1.0))
    return e2, e1, e0


def _set_z_derivatives(z_val: np.ndarray, a: Any, b: Any) -> Any:
    r"""Attach the cubic root's derivatives (MRST `setZDerivatives`).

    `Z` is a root of the cubic, so it carries no automatic derivative. Implicit
    differentiation of `Z^3 + E2 Z^2 + E1 Z + E0 = 0` gives
    `dZ = -(dE2 Z^2 + dE1 Z + dE0)/(3 Z^2 + 2 E2 Z + E1)`; the value of `Z` is
    the one already solved for, only the Jacobian is added.
    """
    if not (_is_adi(a) or _is_adi(b)):
        return z_val
    e2, e1, e0 = _cubic_coeffs(a, b)
    e2v = np.asarray(value(e2), float)
    e1v = np.asarray(value(e1), float)
    denom = 3.0 * z_val ** 2 + 2.0 * e2v * z_val + e1v
    # `combo`'s value is irrelevant; its Jacobian is the numerator dE2 Z^2 + ...
    combo = (z_val ** 2) * e2 + z_val * e1 + e0
    coeff = -1.0 / denom
    jac = [_scale_block(coeff, jb) for jb in combo.jac]
    return _make_ad_like(combo, z_val, jac)


def _fugacity(mix: Mixture, p: Any, comp: List[Any], z_fac: Any,
              a: Any, b: Any, s_i: List[Any], bi: List[Any]) -> List[Any]:
    r"""Component fugacities `f_i = phi_i · p · x_i` (MRST `computeFugacity`)."""
    a1 = -_log(z_fac - b)
    b1 = _log((z_fac + M2 * b) / (z_fac + M1 * b)) * (a / ((M1 - M2) * b))
    a_inv = a ** -1.0
    b_inv = b ** -1.0
    b2 = (z_fac - 1.0) * b_inv
    f = []
    for i in range(mix.n):
        phi = a1 + b1 * (2.0 * s_i[i] * a_inv - bi[i] * b_inv) + bi[i] * b2
        f.append(_exp(phi) * p * comp[i])
    return f


def _compressibility_and_fugacity(mix: Mixture, p: Any, T: Any,
                                  x: List[Any], y: List[Any],
                                  z_l: np.ndarray, z_v: np.ndarray
                                  ) -> Tuple[Any, Any, List[Any], List[Any]]:
    """Attach `Z` derivatives and evaluate both phases' fugacities
    (MRST `getCompressibilityAndFugacity`)."""
    a_ij, bi = _mixing_parameters(mix, p, T)
    si_l, a_l, b_l = _phase_mix(x, a_ij, bi)
    si_v, a_v, b_v = _phase_mix(y, a_ij, bi)
    zl = _set_z_derivatives(z_l, a_l, b_l)
    zv = _set_z_derivatives(z_v, a_v, b_v)
    f_l = _fugacity(mix, p, x, zl, a_l, b_l, si_l, bi)
    f_v = _fugacity(mix, p, y, zv, a_v, b_v, si_v, bi)
    return zl, zv, f_l, f_v


def phase_compressibility_ad(mix: Mixture, p: Any, T: Any,
                             x: List[Any], y: List[Any],
                             z_l: np.ndarray, z_v: np.ndarray
                             ) -> Tuple[Any, Any]:
    r"""The compressibility factors with derivatives (MRST
    `getCompressibilityAndFugacity`, ADI branch).

    A flow model's density and viscosity depend on `Z`, which in turn depends on
    the primaries through the phase's `A, B`. The flash only returns `Z` values;
    this re-attaches the derivatives via :func:`_set_z_derivatives`, given the
    AD compositions `x, y` and pressure `p`. The `Z` *values* are unchanged (they
    are the converged flash roots); only the Jacobian is added.
    """
    a_ij, bi = _mixing_parameters(mix, p, T)
    _, a_l, b_l = _phase_mix(x, a_ij, bi)
    _, a_v, b_v = _phase_mix(y, a_ij, bi)
    return (_set_z_derivatives(z_l, a_l, b_l),
            _set_z_derivatives(z_v, a_v, b_v))


def equations_equilibrium(mix: Mixture, p: Any, T: Any,
                          x: List[Any], y: List[Any], z: List[Any], L: Any,
                          z_l: np.ndarray, z_v: np.ndarray) -> List[Any]:
    r"""The two-phase equilibrium residual `F` (MRST `equationsEquilibrium`).

    `2n+1` equations per cell: `n` component balances `z_i - L x_i - (1-L) y_i`,
    `n` fugacity equalities `f_i^V - f_i^L`, and the closure `sum_i (x_i - y_i)`.
    This is evaluated only on genuinely two-phase cells (pure-phase branches are
    handled by the caller), so the pure-liquid/pure-vapour special cases in
    MRST's version are omitted.
    """
    _, _, f_l, f_v = _compressibility_and_fugacity(mix, p, T, x, y, z_l, z_v)
    n = mix.n
    eqs: List[Any] = []
    for i in range(n):
        eqs.append(z[i] - L * x[i] - (1.0 - L) * y[i])
    for i in range(n):
        eqs.append(f_v[i] - f_l[i])
    closure = x[0] - y[0]
    for i in range(1, n):
        closure = closure + (x[i] - y[i])
    eqs.append(closure)
    return eqs


def _jac_diagonals(eqs: List[Any], n_block: int) -> np.ndarray:
    """Stack the per-cell diagonals of every equation's Jacobian blocks into a
    `(n_cells, n_eq, n_block)` array. Every block is diagonal because the
    equilibrium of a cell touches only that cell's variables. An equation that
    happens to be constant w.r.t. the seeded set comes back as a plain array and
    contributes a zero row."""
    n_eq = len(eqs)
    n_cells = len(value(eqs[0]))
    out = np.zeros((n_cells, n_eq, n_block))
    for r, eq in enumerate(eqs):
        if not _is_adi(eq):
            continue
        for c in range(n_block):
            out[:, r, c] = eq.jac[c].diagonal()
    return out


def phase_fraction_derivatives(mix: Mixture, p: np.ndarray, T: np.ndarray,
                               z: np.ndarray, x: np.ndarray, y: np.ndarray,
                               L: np.ndarray, z_l: np.ndarray, z_v: np.ndarray
                               ) -> Tuple[np.ndarray, List[np.ndarray],
                                          List[np.ndarray]]:
    r"""`dL, dx, dy` w.r.t. `[p, T, z_1..z_n]` (MRST
    `getPhaseFractionDerivativesPTZ`).

    Given a converged flash (`x, y, L, Z_L, Z_V`) at the two-phase cells,
    returns the sensitivities via the implicit function theorem. `dL` is
    `(n_cells, 2+n)`; `dx`/`dy` are lists of `n` such arrays. Column order is
    `[p, T, z_1, ..., z_n]`.
    """
    n = mix.n
    ncell = p.size
    nprimary = 2 + n
    dL = np.zeros((ncell, nprimary))
    dx = [np.zeros((ncell, nprimary)) for _ in range(n)]
    dy = [np.zeros((ncell, nprimary)) for _ in range(n)]
    if ncell == 0:
        return dL, dx, dy

    # -- dF/du : seed the primary variables (p, T, z), s held at values --------
    prim = init_variables_adi(p, T, *[z[:, i] for i in range(n)])
    p_ad, T_ad, z_ad = prim[0], prim[1], list(prim[2:])
    x_val = [x[:, i] for i in range(n)]
    y_val = [y[:, i] for i in range(n)]
    eqs_p = equations_equilibrium(mix, p_ad, T_ad, x_val, y_val, z_ad, L,
                                  z_l, z_v)
    dFdu = _jac_diagonals(eqs_p, nprimary)              # (ncell, 2n+1, 2+n)

    # -- dF/ds : seed the secondary variables (L, x, y), u held at values ------
    sec = init_variables_adi(L, *[x[:, i] for i in range(n)],
                             *[y[:, i] for i in range(n)])
    L_ad = sec[0]
    x_ad = list(sec[1:1 + n])
    y_ad = list(sec[1 + n:1 + 2 * n])
    z_val = [z[:, i] for i in range(n)]
    eqs_s = equations_equilibrium(mix, p, T, x_ad, y_ad, z_val, L_ad,
                                  z_l, z_v)
    dFds = _jac_diagonals(eqs_s, 1 + 2 * n)             # (ncell, 2n+1, 2n+1)

    # -- ds/du = -(dF/ds)^{-1} (dF/du), one small dense solve per cell ---------
    dsdu = np.linalg.solve(dFds, -dFdu)                 # (ncell, 2n+1, 2+n)

    dL[:, :] = dsdu[:, 0, :]
    for i in range(n):
        dx[i][:, :] = dsdu[:, 1 + i, :]
        dy[i][:, :] = dsdu[:, 1 + n + i, :]
    return dL, dx, dy


def _add_derivative_ptz(val: np.ndarray, dvdu: np.ndarray,
                        p: ADI, z: List[ADI]) -> ADI:
    r"""Turn value + `[dv/dp, dv/dT, dv/dz_1..dv/dz_n]` into an ADI in the
    model's primaries (MRST `addDerivativePTZ`).

    The isothermal flow model differentiates against `p` and `z` only, so the
    temperature column is dropped and the chain rule composes the flash
    sensitivity with each primary's own Jacobian:
    `dv = dv/dp · dp + sum_i dv/dz_i · dz_i`.

    Passing the *dependent* `z_n = 1 - sum_{i<n} z_i` as an ADI in `z` makes the
    `sum z = 1` constraint fall out for free: its Jacobian is `-I` in the
    independent blocks, so `dv/dz_n` re-enters with the right sign and the
    combination reduces to the reduced-variable derivative.
    """
    val = np.asarray(val, float)
    acc = _make_ad_like(p, val.copy(), [_scale_block(dvdu[:, 0], jb) for jb in p.jac])
    for i, zi in enumerate(z):
        acc = acc + _make_ad_like(zi, np.zeros_like(val),
                                  [_scale_block(dvdu[:, 2 + i], jb) for jb in zi.jac])
    return acc


def flash_ad(mix: Mixture, p: ADI, T: np.ndarray, z: List[ADI]
             ) -> Tuple[List[ADI], List[ADI], ADI, np.ndarray, np.ndarray]:
    r"""Flash with derivatives w.r.t. the model primaries (MRST
    `getPhaseFractionAsADI`).

    `p` and the `z_i` are ADI in the flow model's primary variables. The flash
    itself runs on values; the two-phase cells get analytic `(p, z)`
    derivatives via :func:`phase_fraction_derivatives` composed through the
    chain rule; single-phase cells carry `x = y = z` (so their only dependence
    is the identity in `z`). Returns `x, y` (lists of `n` ADI), `L` (ADI), and
    the value-level `Z_L, Z_V`.
    """
    n = mix.n
    pv = np.asarray(value(p), float)
    zv = np.column_stack([np.asarray(value(zi), float) for zi in z])
    fr = flash(mix, pv, T, zv)
    Lval = 1.0 - fr.V
    two = fr.two_phase
    ncell = pv.size

    dL = np.zeros((ncell, 2 + n))
    dx = [np.zeros((ncell, 2 + n)) for _ in range(n)]
    dy = [np.zeros((ncell, 2 + n)) for _ in range(n)]
    if two.any():
        dLc, dxc, dyc = phase_fraction_derivatives(
            mix, pv[two], T[two], zv[two], fr.x[two], fr.y[two],
            Lval[two], fr.Z_L[two], fr.Z_V[two])
        dL[two] = dLc
        for i in range(n):
            dx[i][two] = dxc[i]
            dy[i][two] = dyc[i]

    # Single-phase cells: x = y = z (MRST's convention - the absent phase's
    # composition is undefined and set to the overall one), so d/dz is the
    # identity and d/dp is zero.
    single = ~two
    xv, yv = fr.x.copy(), fr.y.copy()
    xv[single] = zv[single]
    yv[single] = zv[single]
    for i in range(n):
        dx[i][single, 2 + i] = 1.0
        dy[i][single, 2 + i] = 1.0

    x_out = [_add_derivative_ptz(xv[:, i], dx[i], p, z) for i in range(n)]
    y_out = [_add_derivative_ptz(yv[:, i], dy[i], p, z) for i in range(n)]
    L_out = _add_derivative_ptz(Lval, dL, p, z)
    return x_out, y_out, L_out, fr.Z_L, fr.Z_V


if __name__ == "__main__":
    from pymrst.modules.compositional.eos import simple_mixture

    def _fd_flash(mix, p, T, z, eps_p=1.0, dz=1e-6):
        """A finite-difference reference for the flash derivatives.

        The `z` sensitivities are checked as directional derivatives that
        respect `sum z = 1` (raise `z_j`, lower `z_n` by the same amount), since
        a raw one-sided bump would leave the composition unnormalised. That
        directional check is `d/dz_j - d/dz_n`, which is exactly the combination
        the flow model uses.
        """
        n = mix.n
        base = flash(mix, p, T, z)
        Lb = 1.0 - base.V

        pp = flash(mix, p + eps_p, T, z)
        pm = flash(mix, p - eps_p, T, z)
        dL_dp = ((1 - pp.V) - (1 - pm.V)) / (2 * eps_p)
        dx_dp = (pp.x - pm.x) / (2 * eps_p)
        dy_dp = (pp.y - pm.y) / (2 * eps_p)

        dL_dz = np.zeros((p.size, n - 1))
        dx_dz = np.zeros((p.size, n, n - 1))
        dy_dz = np.zeros((p.size, n, n - 1))
        for j in range(n - 1):
            zp = z.copy(); zp[:, j] += dz; zp[:, -1] -= dz
            zm = z.copy(); zm[:, j] -= dz; zm[:, -1] += dz
            fp, fm = flash(mix, p, T, zp), flash(mix, p, T, zm)
            dL_dz[:, j] = ((1 - fp.V) - (1 - fm.V)) / (2 * dz)
            dx_dz[:, :, j] = (fp.x - fm.x) / (2 * dz)
            dy_dz[:, :, j] = (fp.y - fm.y) / (2 * dz)
        return Lb, dL_dp, dx_dp, dy_dp, dL_dz, dx_dz, dy_dz

    def test_pressure_derivative_matches_fd():
        """dL/dp, dx/dp, dy/dp from the implicit function theorem match a
        central finite difference of the flash."""
        mix = simple_mixture()
        p = np.array([50e5, 70e5, 90e5])
        T = np.full(3, 350.0)
        z = np.tile([0.5, 0.3, 0.2], (3, 1))
        fr = flash(mix, p, T, z)
        assert fr.two_phase.all(), "test conditions must be two-phase"
        dL, dx, dy = phase_fraction_derivatives(
            mix, p, T, z, fr.x, fr.y, 1.0 - fr.V, fr.Z_L, fr.Z_V)
        _, dL_dp, dx_dp, dy_dp, *_ = _fd_flash(mix, p, T, z)
        # column 0 is d/dp
        assert np.allclose(dL[:, 0], dL_dp, rtol=1e-5, atol=1e-11), \
            (dL[:, 0], dL_dp)
        for i in range(mix.n):
            assert np.allclose(dx[i][:, 0], dx_dp[:, i], rtol=1e-5, atol=1e-11)
            assert np.allclose(dy[i][:, 0], dy_dp[:, i], rtol=1e-5, atol=1e-11)
        print(f"ok test_pressure_derivative_matches_fd "
              f"(dL/dp = {dL[0, 0]:.3e} 1/Pa)")

    def test_composition_derivative_matches_fd():
        """The feasible directional derivative d/dz_j - d/dz_n matches FD."""
        mix = simple_mixture()
        n = mix.n
        p = np.array([50e5, 80e5])
        T = np.full(2, 360.0)
        z = np.tile([0.55, 0.3, 0.15], (2, 1))
        fr = flash(mix, p, T, z)
        assert fr.two_phase.all()
        dL, dx, dy = phase_fraction_derivatives(
            mix, p, T, z, fr.x, fr.y, 1.0 - fr.V, fr.Z_L, fr.Z_V)
        _, _, _, _, dL_dz, dx_dz, dy_dz = _fd_flash(mix, p, T, z)
        for j in range(n - 1):
            # analytic direction: column (2+j) minus column (2+n-1)
            aL = dL[:, 2 + j] - dL[:, 2 + (n - 1)]
            assert np.allclose(aL, dL_dz[:, j], rtol=1e-4, atol=1e-6), \
                (j, aL, dL_dz[:, j])
            for i in range(n):
                aXi = dx[i][:, 2 + j] - dx[i][:, 2 + (n - 1)]
                aYi = dy[i][:, 2 + j] - dy[i][:, 2 + (n - 1)]
                assert np.allclose(aXi, dx_dz[:, i, j], rtol=1e-4, atol=1e-6)
                assert np.allclose(aYi, dy_dz[:, i, j], rtol=1e-4, atol=1e-6)
        print("ok test_composition_derivative_matches_fd")

    def test_flash_ad_chain_rule():
        """flash_ad composes the flash sensitivity with the primaries: seeding
        p alone must reproduce dx/dp in the ADI Jacobian."""
        mix = simple_mixture()
        n = mix.n
        p0 = np.array([60e5, 60e5])
        T = np.full(2, 350.0)
        z0 = np.tile([0.5, 0.3, 0.2], (2, 1))
        # primaries: p and z_1..z_{n-1}; z_n = 1 - sum
        prim = init_variables_adi(p0, z0[:, 0], z0[:, 1])
        p = prim[0]
        z = [prim[1], prim[2], 1.0 - prim[1] - prim[2]]
        x, y, L, ZL, ZV = flash_ad(mix, p, T, z)
        # dL/dp is jac block 0 (the p block), diagonal
        dLdp = L.jac[0].diagonal()
        fr = flash(mix, p0, T, z0)
        dLc, _, _ = phase_fraction_derivatives(
            mix, p0, T, z0, fr.x, fr.y, 1.0 - fr.V, fr.Z_L, fr.Z_V)
        assert np.allclose(dLdp, dLc[:, 0], rtol=1e-10, atol=1e-14), \
            (dLdp, dLc[:, 0])
        # value round-trips
        assert np.allclose(value(L), 1.0 - fr.V)
        assert np.allclose(value(x[0]), fr.x[:, 0])
        print("ok test_flash_ad_chain_rule")

    def test_single_phase_is_identity_in_z():
        """A single-phase cell has x = y = z; flash_ad must carry that with an
        identity z-Jacobian and no p-dependence."""
        mix = simple_mixture()
        p0 = np.array([600e5])                     # high p -> all liquid
        T = np.full(1, 300.0)
        z0 = np.array([[0.5, 0.3, 0.2]])
        prim = init_variables_adi(p0, z0[:, 0], z0[:, 1])
        p = prim[0]
        z = [prim[1], prim[2], 1.0 - prim[1] - prim[2]]
        x, y, L, _, _ = flash_ad(mix, p, T, z)
        assert np.allclose(value(x[0]), z0[:, 0])
        assert np.allclose(value(y[0]), z0[:, 0])
        # x_1 depends on z_1 with slope 1, not on p
        assert np.allclose(x[0].jac[0].diagonal(), 0.0)      # d x_1 / d p
        assert np.allclose(x[0].jac[1].diagonal(), 1.0)      # d x_1 / d z_1
        print("ok test_single_phase_is_identity_in_z")

    test_pressure_derivative_matches_fd()
    test_composition_derivative_matches_fd()
    test_flash_ad_chain_rule()
    test_single_phase_is_identity_in_z()
    print("\nAll tests passed!")
