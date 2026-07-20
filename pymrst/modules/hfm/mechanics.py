"""HFM to TEDFM/VEM mechanics workflow coupling."""

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional, Sequence

import numpy as np

from pymrst.modules.hfm.edfm2d import EDFMGrid, FractureLine, build_edfm
from pymrst.modules.vemmech import (
    ElasticBC,
    calculate_stress_vem,
    create_augmented_grid,
    enu2c_2d,
    vem_lin_elast_tedfm,
)


@dataclass
class EDFMMechanicsResult:
    """Coupled EDFM/TEDFM mechanics result.

    ``normal_strain`` and ``aperture_change`` are per fracture cell. They are
    recovered by projecting the surrounding matrix-cell VEM strain onto the
    embedded fracture normal and averaging over the matrix-fracture NNCs.
    """

    edfm: EDFMGrid
    wrapper: SimpleNamespace
    displacement: np.ndarray
    stress: np.ndarray
    strain: np.ndarray
    normal_strain: np.ndarray
    aperture_change: np.ndarray
    matrix_cell_for_fracture: np.ndarray
    extra: dict


@dataclass
class FracturePropagationResult:
    """Result from a mechanics-driven fracture-tip propagation update."""

    lines: list[FractureLine]
    edfm: EDFMGrid
    changed: bool
    tip_extensions: np.ndarray
    driving_force: Optional[np.ndarray] = None


def build_tedfm_wrapper(edfm: EDFMGrid) -> SimpleNamespace:
    """Build an MRST-shaped TEDFM wrapper from a 2D EDFM grid."""
    matrix = create_augmented_grid(deepcopy(edfm.g))
    return SimpleNamespace(Matrix=matrix, griddim=matrix.griddim, EDFM=edfm)


def fracture_normals(lines: list[FractureLine], frac_line: np.ndarray) -> np.ndarray:
    """Return one unit normal per EDFM fracture cell."""
    normals = np.zeros((frac_line.size, 2))
    for i, line in enumerate(lines):
        tangent = line.b - line.a
        length = float(np.linalg.norm(tangent))
        if length <= 0.0:
            raise ValueError("fracture line has zero length")
        n = np.array([-tangent[1], tangent[0]]) / length
        normals[frac_line == i] = n
    return normals


def fracture_normal_strain(
    edfm: EDFMGrid,
    lines: list[FractureLine],
    strain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project matrix strain onto embedded fracture normals.

    The return values are ``(normal_strain, representative_matrix_cell)`` for
    each fracture cell. If a fracture cell has several matrix-fracture NNCs,
    PyMRST uses transmissibility weights, which is the same coupling weight
    used by the EDFM flow operator.
    """
    strain = np.asarray(strain, dtype=float)
    if strain.shape[0] != edfm.n_matrix or strain.shape[1] != 3:
        raise ValueError("2D matrix strain must have shape (n_matrix, 3)")
    normals = fracture_normals(lines, edfm.frac_line)
    fm = np.asarray(edfm.conn_type) == "frac_matrix"
    pairs = edfm.conn[fm]
    weights = edfm.trans[fm]

    normal_strain = np.zeros(edfm.frac_cells)
    total_weight = np.zeros(edfm.frac_cells)
    representative = np.full(edfm.frac_cells, -1, dtype=np.int64)
    for (a, b), w in zip(pairs, weights):
        if a < edfm.n_matrix <= b:
            m, f = int(a), int(b - edfm.n_matrix)
        elif b < edfm.n_matrix <= a:
            m, f = int(b), int(a - edfm.n_matrix)
        else:
            continue
        n = normals[f]
        exx, eyy, gamma_xy = strain[m]
        eps_n = exx * n[0] ** 2 + eyy * n[1] ** 2 + gamma_xy * n[0] * n[1]
        normal_strain[f] += w * eps_n
        total_weight[f] += w
        if representative[f] < 0 or w > total_weight[f]:
            representative[f] = m

    hit = total_weight > 0
    normal_strain[hit] /= total_weight[hit]
    if np.any(~hit):
        nearest = np.argmin(
            np.linalg.norm(
                edfm.g.cells.centroids[:, None, :2]
                - edfm.frac_centroids[None, :, :2],
                axis=2,
            ),
            axis=0,
        )
        for f in np.flatnonzero(~hit):
            m = int(nearest[f])
            n = normals[f]
            exx, eyy, gamma_xy = strain[m]
            normal_strain[f] = (
                exx * n[0] ** 2 + eyy * n[1] ** 2
                + gamma_xy * n[0] * n[1]
            )
            representative[f] = m
    return normal_strain, representative


def solve_edfm_tedfm_mechanics(
    edfm: EDFMGrid,
    lines: list[FractureLine],
    el_bc: ElasticBC,
    *,
    C: Optional[np.ndarray] = None,
    young: float | np.ndarray = 1.0e9,
    poisson: float | np.ndarray = 0.25,
    load: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    pressure: Optional[np.ndarray] = None,
    aperture: Optional[float] = None,
    alpha_scaling: float = 1.0,
) -> EDFMMechanicsResult:
    """Run a 2D EDFM/TEDFM mechanics workflow.

    ``pressure`` may be either matrix-only or full EDFM pressure. Full EDFM
    pressure is reduced to the matrix cells for the VEM solve, while fracture
    cells receive mechanically coupled aperture-change estimates afterward.
    """
    wrapper = build_tedfm_wrapper(edfm)
    matrix = wrapper.Matrix
    if C is None:
        C = enu2c_2d(
            np.broadcast_to(np.asarray(young, dtype=float), edfm.n_matrix),
            np.broadcast_to(np.asarray(poisson, dtype=float), edfm.n_matrix),
        )
    if load is None:
        load = lambda x: np.zeros_like(x)
    p_matrix = None
    if pressure is not None:
        pressure = np.asarray(pressure, dtype=float).ravel()
        if pressure.size == edfm.n_total:
            p_matrix = pressure[:edfm.n_matrix]
        elif pressure.size == edfm.n_matrix:
            p_matrix = pressure
        else:
            raise ValueError("pressure must be matrix-only or full EDFM length")

    uu, extra = vem_lin_elast_tedfm(
        wrapper, C, el_bc, load=load, pressure=p_matrix,
        alpha_scaling=alpha_scaling,
    )
    stress, strain = calculate_stress_vem(matrix, uu, C)
    normal_strain, representative = fracture_normal_strain(edfm, lines, strain)
    if aperture is None:
        aperture = float(np.mean(edfm.frac_volumes)) if edfm.frac_cells else 0.0
    aperture_change = float(aperture) * normal_strain
    return EDFMMechanicsResult(
        edfm=edfm,
        wrapper=wrapper,
        displacement=uu,
        stress=stress,
        strain=strain,
        normal_strain=normal_strain,
        aperture_change=aperture_change,
        matrix_cell_for_fracture=representative,
        extra=extra,
    )


def propagate_fractures_by_aperture(
    g,
    lines: Sequence[FractureLine],
    mechanics: EDFMMechanicsResult,
    *,
    aperture_threshold: float,
    extension: float,
    aperture: float,
    perm_matrix: np.ndarray,
    perm_frac: float,
    poro_matrix: np.ndarray,
    poro_frac: float = 1.0,
    cell_size: float = 0.1,
) -> FracturePropagationResult:
    """Extend 2D fracture tips where mechanics opens them enough.

    The criterion is local and MRST-workflow shaped: use the aperture change
    recovered on the terminal fracture cells. A positive terminal opening above
    ``aperture_threshold`` extends that tip by ``extension`` along the fracture
    tangent, then the EDFM grid/NNCs are rebuilt.
    """
    if extension < 0.0:
        raise ValueError("extension must be non-negative")
    new_lines: list[FractureLine] = []
    tip_ext = np.zeros((len(lines), 2), dtype=float)
    changed = False
    for i, line in enumerate(lines):
        tangent = line.b - line.a
        length = float(np.linalg.norm(tangent))
        if length <= 0.0:
            raise ValueError("fracture line has zero length")
        t = tangent / length
        fcells = np.flatnonzero(mechanics.edfm.frac_line == i)
        if fcells.size == 0:
            new_lines.append(FractureLine(line.endp.copy()))
            continue
        centers = mechanics.edfm.frac_centroids[fcells, :2]
        s = (centers - line.a) @ t
        left_cell = fcells[int(np.argmin(s))]
        right_cell = fcells[int(np.argmax(s))]
        a = line.a.copy()
        b = line.b.copy()
        if mechanics.aperture_change[left_cell] > aperture_threshold:
            a = a - extension * t
            tip_ext[i, 0] = extension
            changed = True
        if mechanics.aperture_change[right_cell] > aperture_threshold:
            b = b + extension * t
            tip_ext[i, 1] = extension
            changed = True
        new_lines.append(FractureLine(np.r_[a, b]))

    edfm = build_edfm(
        g,
        new_lines,
        aperture=aperture,
        perm_matrix=perm_matrix,
        perm_frac=perm_frac,
        poro_matrix=poro_matrix,
        poro_frac=poro_frac,
        cell_size=cell_size,
    )
    return FracturePropagationResult(
        lines=new_lines,
        edfm=edfm,
        changed=changed,
        tip_extensions=tip_ext,
        driving_force=None,
    )


def fracture_tip_mode_i_indicator(
    lines: Sequence[FractureLine],
    mechanics: EDFMMechanicsResult,
    *,
    young: float,
    poisson: float,
) -> np.ndarray:
    """Return a simple mode-I toughness indicator for each fracture tip.

    The indicator uses the terminal fracture-cell aperture change as crack
    opening and scales it by the plane-strain modulus and crack half-length:
    ``K ~= E' * opening / sqrt(pi*a)``. It is intended as a propagation driver
    for workflow studies, not a replacement for a full fracture-process-zone
    model.
    """
    eprime = young / (1.0 - poisson**2)
    out = np.zeros((len(lines), 2), dtype=float)
    for i, line in enumerate(lines):
        tangent = line.b - line.a
        length = float(np.linalg.norm(tangent))
        if length <= 0.0:
            continue
        t = tangent / length
        fcells = np.flatnonzero(mechanics.edfm.frac_line == i)
        if fcells.size == 0:
            continue
        centers = mechanics.edfm.frac_centroids[fcells, :2]
        s = (centers - line.a) @ t
        left_cell = fcells[int(np.argmin(s))]
        right_cell = fcells[int(np.argmax(s))]
        scale = eprime / np.sqrt(np.pi * max(0.5 * length, 1e-12))
        out[i, 0] = max(mechanics.aperture_change[left_cell], 0.0) * scale
        out[i, 1] = max(mechanics.aperture_change[right_cell], 0.0) * scale
    return out


def propagate_fractures_by_toughness(
    g,
    lines: Sequence[FractureLine],
    mechanics: EDFMMechanicsResult,
    *,
    toughness: float,
    young: float,
    poisson: float,
    extension: float,
    aperture: float,
    perm_matrix: np.ndarray,
    perm_frac: float,
    poro_matrix: np.ndarray,
    poro_frac: float = 1.0,
    cell_size: float = 0.1,
) -> FracturePropagationResult:
    """Extend tips where the mode-I indicator exceeds ``toughness``."""
    driving = fracture_tip_mode_i_indicator(
        lines, mechanics, young=young, poisson=poisson
    )
    new_lines: list[FractureLine] = []
    tip_ext = np.zeros_like(driving)
    changed = False
    for i, line in enumerate(lines):
        tangent = line.b - line.a
        length = float(np.linalg.norm(tangent))
        if length <= 0.0:
            raise ValueError("fracture line has zero length")
        t = tangent / length
        a = line.a.copy()
        b = line.b.copy()
        if driving[i, 0] > toughness:
            a = a - extension * t
            tip_ext[i, 0] = extension
            changed = True
        if driving[i, 1] > toughness:
            b = b + extension * t
            tip_ext[i, 1] = extension
            changed = True
        new_lines.append(FractureLine(np.r_[a, b]))
    edfm = build_edfm(
        g,
        new_lines,
        aperture=aperture,
        perm_matrix=perm_matrix,
        perm_frac=perm_frac,
        poro_matrix=poro_matrix,
        poro_frac=poro_frac,
        cell_size=cell_size,
    )
    return FracturePropagationResult(
        lines=new_lines,
        edfm=edfm,
        changed=changed,
        tip_extensions=tip_ext,
        driving_force=driving,
    )
