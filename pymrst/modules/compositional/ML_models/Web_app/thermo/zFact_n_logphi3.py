#!/usr/bin/env python3
"""Peng-Robinson cubic EOS: Z-factor and component log-fugacity coefficients.

Torch implementation mirroring the original MATLAB routine, with an extra
`has_3_roots` flag so callers can gate physics losses on truly two-phase
regions.

Convention:
- Input units are SI (Pa, K). Composition `x` is renormalised on entry.
- `phase="liquid"` / `"vapor"` selects the smallest / largest positive Z when
  three real roots exist; `phase="stable"` picks the root with lower Gibbs
  free energy (matches MATLAB "stable" selection).
"""

import torch

RR: float = 8.3144621  # universal gas constant, J/(mol K)


def zFact_n_logPhi(
    pres,
    TinK,
    pCrit,
    tCrit,
    accFact,
    x,
    deltaij,
    phase: str = "stable",
):
    """Solve PR-EOS and return (z_sel, logPhi, has_3_roots, disc).

    Returns:
      z_sel:       scalar tensor (selected compressibility)
      logPhi:      tensor [nc] (component log-fugacity coefficients)
      has_3_roots: bool scalar tensor (True when disc < 0)
      disc:        scalar tensor (depressed-cubic discriminant)
    """
    pres    = torch.as_tensor(pres,    dtype=torch.float64)
    TinK    = torch.as_tensor(TinK,    dtype=torch.float64, device=pres.device)
    pCrit   = torch.as_tensor(pCrit,   dtype=torch.float64, device=pres.device)
    tCrit   = torch.as_tensor(tCrit,   dtype=torch.float64, device=pres.device)
    accFact = torch.as_tensor(accFact, dtype=torch.float64, device=pres.device)
    x       = torch.as_tensor(x,       dtype=torch.float64, device=pres.device)
    deltaij = torch.as_tensor(deltaij, dtype=torch.float64, device=pres.device)

    x = x / (x.sum() + 1e-18)

    # kappa correlation (matches MATLAB)
    Tr = TinK / tCrit
    kappa = (-0.26992 * accFact + 1.54226) * accFact + 0.37464

    w = accFact
    mask = (w > 0.5) & (kappa > 1.07829)
    kappa_alt = ((0.01667*w - 0.1644)*w + 1.485)*w + 0.3796
    kappa = torch.where(mask, kappa_alt, kappa)

    sqrtAlpha = 1.0 + kappa * (1.0 - torch.sqrt(Tr))
    alpha = sqrtAlpha * sqrtAlpha

    aa = alpha * (0.457235 * RR * RR) * tCrit * tCrit / pCrit
    bb = 0.077796 * RR * tCrit / pCrit

    bMix = torch.dot(x, bb)

    aij = (1.0 - deltaij) * torch.sqrt(torch.outer(aa, aa))
    aMix = torch.sum(torch.outer(x, x) * aij)

    A = aMix * pres / (RR*RR*TinK*TinK)
    B = bMix * pres / (RR*TinK)

    # PR cubic: z^3 + a2 z^2 + a1 z + a0 = 0
    a2 = -(1.0 - B)
    a1 = A - 3.0*B*B - 2.0*B
    a0 = -B*(A - B*(1.0 + B))

    # Depressed cubic: y^3 + p y + q = 0 with z = y - a2/3
    p = (3.0*a1 - a2*a2)/3.0
    q = (2.0*a2*a2*a2 - 9.0*a2*a1 + 27.0*a0)/27.0
    disc = (q/2.0)**2 + (p/3.0)**3

    has_3_roots = (disc < 0.0)

    POS_EPS = torch.tensor(1e-12, dtype=torch.float64, device=pres.device)
    eps = torch.tensor(1e-30, dtype=torch.float64, device=pres.device)

    # disc >= 0: single real root via Cardano
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    u = torch.sign(-q/2.0 + sqrt_disc) * torch.abs(-q/2.0 + sqrt_disc + 1e-30)**(1.0/3.0)
    v = torch.sign(-q/2.0 - sqrt_disc) * torch.abs(-q/2.0 - sqrt_disc + 1e-30)**(1.0/3.0)
    y_one = u + v
    z_one = y_one - a2/3.0

    z_one_safe = torch.where(z_one > POS_EPS, z_one, 1.0 + B)

    # disc < 0: three real roots via trig identity
    r = torch.sqrt(torch.clamp(-p/3.0, min=0.0) + 1e-30)
    denom = torch.sqrt(torch.clamp(-(p/3.0)**3, min=1e-30))
    phi = torch.acos(torch.clamp((-q/2.0)/denom, -1.0, 1.0))

    y1 = 2.0*r*torch.cos(phi/3.0)
    y2 = 2.0*r*torch.cos((phi + 2.0*torch.pi)/3.0)
    y3 = 2.0*r*torch.cos((phi + 4.0*torch.pi)/3.0)

    z_roots = torch.stack([y1 - a2/3.0, y2 - a2/3.0, y3 - a2/3.0])

    big = torch.tensor(1e30, dtype=torch.float64, device=pres.device)
    z_for_min = torch.where(z_roots > POS_EPS, z_roots, big)
    z_for_max = torch.where(z_roots > POS_EPS, z_roots, -big)

    z_min = torch.min(z_for_min)
    z_max = torch.max(z_for_max)

    # Root selection
    ph = phase.lower()
    if ph == "liquid":
        z_sel = torch.where(has_3_roots, z_min, z_one_safe)
    elif ph == "vapor":
        z_sel = torch.where(has_3_roots, z_max, z_one_safe)
    else:
        # MATLAB "stable" selection: pick min vs. max root by Gibbs difference.
        maxR = z_max
        minR = z_min

        term_log1 = torch.log((maxR - B + eps) / (minR - B + eps))

        log_arg = ((maxR + 2.414*B) * (minR - 0.414*B)) / ((maxR - 0.414*B) * (minR + 2.414*B) + eps)
        term_log2 = torch.log(torch.abs(log_arg) + eps)

        xaij = torch.matmul(x, aij)  # [nc]
        vec = (2.0*xaij/(aMix + eps) - bb/(bMix + eps))

        tempVar = bb/(bMix + eps) * (maxR - minR) - term_log1 \
                  - (A/(2.0*torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=pres.device))*(B + eps))) * vec * term_log2

        dG = RR*TinK*torch.dot(x, tempVar)
        z_stable = torch.where(dG > 0.0, minR, maxR)

        z_sel = torch.where(has_3_roots, z_stable, z_one_safe)

    # logPhi (MATLAB formula)
    logPhi = bb/(bMix + eps)*(z_sel - 1.0) \
             - torch.log(z_sel - B + eps) \
             - A/(2.0*torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=pres.device))*(B + eps)) \
               * (2.0*torch.matmul(x, aij)/(aMix + eps) - bb/(bMix + eps)) \
               * torch.log((z_sel + 2.414213562373095*B)/(z_sel - 0.414213562373095*B + eps) + eps)

    return z_sel, logPhi, has_3_roots, disc


if __name__ == "__main__":
    # Smoke / regression tests for PR-EOS root selection.
    _device = torch.device("cpu")
    _T = 353.0
    _Pci = torch.tensor([4.604e6, 4.249e6, 1.620e6], dtype=torch.float64, device=_device)
    _Tci = torch.tensor([190.58, 369.82, 723.00],   dtype=torch.float64, device=_device)
    _acc = torch.tensor([0.0108, 0.1520, 0.8000],   dtype=torch.float64, device=_device)
    _dij = torch.zeros((3, 3), dtype=torch.float64, device=_device)
    _dij[0, 1] = _dij[1, 0] = 0.00683
    _dij[0, 2] = _dij[2, 0] = 0.01230
    _dij[1, 2] = _dij[2, 1] = 0.00082
    _x = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64, device=_device)

    def test_shapes_and_finite():
        z, lp, has3, disc = zFact_n_logPhi(1e6, _T, _Pci, _Tci, _acc, _x, _dij, phase="liquid")
        assert lp.shape == (_x.numel(),), f"logPhi shape was {lp.shape}"
        assert torch.isfinite(z).item(), "z_sel is non-finite"
        assert torch.isfinite(lp).all().item(), "logPhi contains non-finite entries"
        assert isinstance(has3.item(), bool)
        assert torch.isfinite(disc).item(), "disc is non-finite"
        print("✓ test_shapes_and_finite passed")

    def test_liquid_vapor_ordering_three_roots():
        zl, _, has3l, _ = zFact_n_logPhi(5e6, _T, _Pci, _Tci, _acc, _x, _dij, phase="liquid")
        zv, _, has3v, _ = zFact_n_logPhi(5e6, _T, _Pci, _Tci, _acc, _x, _dij, phase="vapor")
        if has3l.item() and has3v.item():
            assert zv.item() >= zl.item(), f"Z_vap ({zv}) should be >= Z_liq ({zl}) in 3-root region"
        print("✓ test_liquid_vapor_ordering_three_roots passed")

    def test_high_pressure_single_root():
        z, _, has3, disc = zFact_n_logPhi(2e7, _T, _Pci, _Tci, _acc, _x, _dij, phase="vapor")
        assert z.item() > 0.0
        assert (disc.item() >= 0.0) == (not has3.item())
        print("✓ test_high_pressure_single_root passed")

    def test_low_pressure_vapor_z_near_one():
        z, _, _, _ = zFact_n_logPhi(1e4, _T, _Pci, _Tci, _acc, _x, _dij, phase="vapor")
        assert abs(z.item() - 1.0) < 0.1, f"Vapor Z at 1e4 Pa should be ~1, got {z.item()}"
        print("✓ test_low_pressure_vapor_z_near_one passed")

    test_shapes_and_finite()
    test_liquid_vapor_ordering_three_roots()
    test_high_pressure_single_root()
    test_low_pressure_vapor_z_near_one()
    print("\nAll thermo tests passed!")
