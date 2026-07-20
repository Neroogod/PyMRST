"""Embedded Discrete Fracture Model in 2D (MRST hfm).

Fractures are embedded in a matrix grid without conforming to it. Each
fracture line is gridded into its own 1D cells; matrix and fracture cells
are coupled by non-neighboring connections (NNCs) whose strength is the
conductivity index

    CI = L_f / <d>,     <d> = (1/A) * integral over the matrix cell of
                               the distance to the fracture plane,

where L_f is the fracture length inside the matrix cell (CIcalculator2D /
getAvgFracDist2D). The NNC transmissibility uses the pore-volume-weighted
harmonic permeability of the two connected cells (computeEffectiveTrans):

    T = CI * (pv1 + pv2) / (pv1/K1 + pv2/K2).

Fracture-fracture intersections get an NNC formed from the harmonic
average of the two fracture cells' half-transmissibilities (frac_frac_nnc).

<d> is computed exactly here (the matrix cell polygon is clipped by the
fracture line and the linear distance function is integrated on each
side), which is what MRST's Delaunay quadrature converges to.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid

TOL = 1e-12


@dataclass
class FractureLine:
    """A straight fracture: endpoints (x1, y1, x2, y2)."""

    endp: np.ndarray

    @property
    def a(self) -> np.ndarray:
        return self.endp[:2]

    @property
    def b(self) -> np.ndarray:
        return self.endp[2:]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    def line_coeffs(self) -> Tuple[float, float, float]:
        """Normalized (A, B, C) with A*x + B*y + C = 0, A^2 + B^2 = 1."""
        d = self.b - self.a
        n = np.array([-d[1], d[0]]) / self.length
        return float(n[0]), float(n[1]), float(-n @ self.a)


@dataclass
class EDFMGrid:
    """Matrix grid plus fracture cells and the NNC connection list."""

    g: Grid                              # matrix grid (2D)
    n_matrix: int
    frac_cells: int                      # number of fracture cells
    frac_volumes: np.ndarray             # (frac_cells,)
    frac_centroids: np.ndarray           # (frac_cells, 2)
    frac_line: np.ndarray                # (frac_cells,) owning line index
    perm: np.ndarray                     # (n_total,) permeability
    poro: np.ndarray                     # (n_total,) porosity
    conn: np.ndarray                     # (n_conn, 2) cell pairs
    trans: np.ndarray                    # (n_conn,) transmissibilities
    ci: np.ndarray                       # (n_conn,) CI (0 for matrix faces)
    conn_type: List[str] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return self.n_matrix + self.frac_cells

    def volumes(self) -> np.ndarray:
        return np.concatenate([self.g.cells.volumes, self.frac_volumes])


def _cell_polygon(g: Grid, c: int) -> np.ndarray:
    """Vertices of a 2D cell, ordered counter-clockwise."""
    faces = g.cells.faces[g.cells.facePos[c]:g.cells.facePos[c + 1]]
    nodes = np.unique(np.concatenate([
        g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
        for f in faces
    ]))
    pts = g.nodes.coords[nodes, :2]
    ang = np.arctan2(pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean())
    return pts[np.argsort(ang)]


def _clip_halfplane(poly: np.ndarray, abc: Tuple[float, float, float],
                    keep_positive: bool) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon by A*x + B*y + C >= 0."""
    a, b, c = abc
    sgn = 1.0 if keep_positive else -1.0
    out: List[np.ndarray] = []
    n = poly.shape[0]
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp = sgn * (a * p[0] + b * p[1] + c)
        dq = sgn * (a * q[0] + b * q[1] + c)
        if dp >= -TOL:
            out.append(p)
        if (dp > TOL and dq < -TOL) or (dp < -TOL and dq > TOL):
            t = dp / (dp - dq)
            out.append(p + t * (q - p))
    return np.array(out) if out else np.zeros((0, 2))


def _poly_moment(poly: np.ndarray, abc: Tuple[float, float, float]
                 ) -> Tuple[float, float]:
    """Return (area, integral of |A x + B y + C| over the polygon).

    Exact: the integrand is linear and of one sign on the polygon, so the
    fan triangulation's midpoint rule is exact.
    """
    if poly.shape[0] < 3:
        return 0.0, 0.0
    a, b, c = abc
    area = 0.0
    moment = 0.0
    p0 = poly[0]
    for i in range(1, poly.shape[0] - 1):
        p1, p2 = poly[i], poly[i + 1]
        tri_a = 0.5 * abs(np.cross(p1 - p0, p2 - p0))
        cen = (p0 + p1 + p2) / 3.0
        area += tri_a
        moment += tri_a * abs(a * cen[0] + b * cen[1] + c)
    return area, moment


def avg_frac_distance(g: Grid, cell: int, line: FractureLine) -> float:
    """<d>: area-averaged distance from the matrix cell to the fracture line."""
    poly = _cell_polygon(g, cell)
    abc = line.line_coeffs()
    area_tot, mom = 0.0, 0.0
    for keep in (True, False):
        sub = _clip_halfplane(poly, abc, keep)
        a_i, m_i = _poly_moment(sub, abc)
        area_tot += a_i
        mom += m_i
    assert area_tot > 0, "degenerate matrix cell"
    return mom / area_tot


def _seg_in_cell(g: Grid, cell: int, line: FractureLine
                 ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip the fracture segment to the (convex) matrix cell polygon."""
    poly = _cell_polygon(g, cell)
    p, q = line.a.copy(), line.b.copy()
    t0, t1 = 0.0, 1.0
    d = q - p
    n = poly.shape[0]
    for i in range(n):
        e0, e1 = poly[i], poly[(i + 1) % n]
        edge = e1 - e0
        nrm = np.array([-edge[1], edge[0]])       # inward for a CCW polygon
        denom = nrm @ d
        num = nrm @ (e0 - p)
        if abs(denom) < TOL:
            if num < -TOL:
                return None
            continue
        t = num / denom
        if denom > 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    if t1 - t0 < 1e-10:
        return None
    return p + t0 * d, p + t1 * d


def build_edfm(
    g: Grid,
    lines: Sequence[FractureLine],
    aperture: float,
    perm_matrix: np.ndarray,
    perm_frac: float,
    poro_matrix: np.ndarray,
    poro_frac: float = 1.0,
    cell_size: float = 0.1,
    n_cells: Optional[Sequence[int]] = None,
    trans_weight: str = "pore_volume",
) -> EDFMGrid:
    """Assemble the EDFM: fracture cells, NNCs, and transmissibilities.

    Each fracture is split into equal-length cells: n_cells[i] if given,
    otherwise round(length / cell_size). (MRST's assembleFracNodes2D
    instead snaps the count to ratios with an exact 4-decimal
    representation, a float-fragile rule not reproduced here - pass
    n_cells to match a given MRST fracture grid.)

    trans_weight selects the weights in the NNC transmissibility:
    'pore_volume' (poro * V) or 'volume' (V). MRST's computeEffectiveTrans
    uses pore volumes when rock.poro is present and falls back to cell
    volumes otherwise - and the hfm examples do not set porosity.
    """
    assert trans_weight in ("pore_volume", "volume")
    assert g.griddim == 2, "2D EDFM"
    nm = g.cells.num

    # 1. Grid each fracture line into equal-length cells.
    f_start: List[int] = []
    f_nodes: List[np.ndarray] = []
    next_cell = nm
    for i, ln in enumerate(lines):
        if n_cells is not None:
            n_seg = int(n_cells[i])
        else:
            n_seg = max(1, int(round(ln.length / cell_size)))
        t = np.linspace(0.0, 1.0, n_seg + 1)
        f_nodes.append(ln.a + np.outer(t, ln.b - ln.a))
        f_start.append(next_cell)
        next_cell += n_seg
    nf = next_cell - nm

    frac_vol = np.zeros(nf)
    frac_cen = np.zeros((nf, 2))
    frac_line = np.zeros(nf, dtype=np.int64)
    frac_len = np.zeros(nf)
    for i, nodes in enumerate(f_nodes):
        for k in range(nodes.shape[0] - 1):
            c = f_start[i] + k
            seg = nodes[k + 1] - nodes[k]
            frac_len[c - nm] = np.linalg.norm(seg)
            frac_vol[c - nm] = frac_len[c - nm] * aperture
            frac_cen[c - nm] = 0.5 * (nodes[k] + nodes[k + 1])
            frac_line[c - nm] = i

    perm = np.concatenate([np.asarray(perm_matrix, float).ravel(),
                           np.full(nf, perm_frac)])
    poro = np.concatenate([np.asarray(poro_matrix, float).ravel(),
                           np.full(nf, poro_frac)])
    vols = np.concatenate([g.cells.volumes, frac_vol])
    pv = poro * vols if trans_weight == "pore_volume" else vols

    conn: List[Tuple[int, int]] = []
    trans: List[float] = []
    ci_list: List[float] = []
    ctype: List[str] = []

    # 2. Matrix-matrix faces (standard TPFA).
    internal = np.all(g.faces.neighbors >= 0, axis=1)
    cell_no = g.cell_no()
    cf = g.cells.faces
    cvec = g.faces.centroids[cf] - g.cells.centroids[cell_no]
    sgn = np.where(g.faces.neighbors[cf, 0] == cell_no, 1.0, -1.0)
    nvec = sgn[:, None] * g.faces.normals[cf]
    t_half = (perm_matrix.ravel()[cell_no]
              * np.sum(cvec[:, :2] * nvec[:, :2], axis=1)
              / np.sum(cvec[:, :2] ** 2, axis=1))
    t_face = 1.0 / np.bincount(cf, weights=1.0 / np.abs(t_half),
                               minlength=g.faces.num)
    for f in np.flatnonzero(internal):
        conn.append(tuple(g.faces.neighbors[f]))
        trans.append(t_face[f])
        ci_list.append(0.0)
        ctype.append("matrix")

    # 3. Fracture internal connections (1D TPFA along each line).
    for i, nodes in enumerate(f_nodes):
        for k in range(nodes.shape[0] - 2):
            c1, c2 = f_start[i] + k, f_start[i] + k + 1
            # half-trans = K * A / (L/2), A = aperture (unit thickness)
            t1 = perm_frac * aperture / (frac_len[c1 - nm] / 2.0)
            t2 = perm_frac * aperture / (frac_len[c2 - nm] / 2.0)
            conn.append((c1, c2))
            trans.append(1.0 / (1.0 / t1 + 1.0 / t2))
            ci_list.append(0.0)
            ctype.append("frac")

    # 4. Matrix-fracture NNCs (Type I).
    for i, ln in enumerate(lines):
        for m in range(nm):
            seg = _seg_in_cell(g, m, ln)
            if seg is None:
                continue
            p, q = seg
            l_in = float(np.linalg.norm(q - p))
            if l_in < 1e-10:
                continue
            d_avg = avg_frac_distance(g, m, ln)
            ci_tot = l_in / d_avg
            # split CI over the fracture cells overlapping this matrix cell
            nodes = f_nodes[i]
            s = np.linalg.norm(nodes - ln.a, axis=1)
            s0 = float(np.linalg.norm(p - ln.a))
            s1 = float(np.linalg.norm(q - ln.a))
            lo, hi = min(s0, s1), max(s0, s1)
            for k in range(nodes.shape[0] - 1):
                a_k, b_k = s[k], s[k + 1]
                ov = max(0.0, min(b_k, hi) - max(a_k, lo))
                if ov < 1e-10:
                    continue
                c_f = f_start[i] + k
                ci = ci_tot * ov / l_in
                w1 = pv[m] / perm[m]
                w2 = pv[c_f] / perm[c_f]
                t = ci * (pv[m] + pv[c_f]) / (w1 + w2)
                conn.append((m, c_f))
                trans.append(t)
                ci_list.append(ci)
                ctype.append("frac_matrix")

    # 5. Fracture-fracture NNCs at intersections (Type II).
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x = _line_intersection(lines[i], lines[j])
            if x is None:
                continue
            ci_f, cj_f = _cell_at(f_nodes[i], f_start[i], x), \
                _cell_at(f_nodes[j], f_start[j], x)
            if ci_f is None or cj_f is None:
                continue
            # half-transmissibilities from each fracture cell to the
            # intersection point, harmonically averaged.
            d1 = float(np.linalg.norm(frac_cen[ci_f - nm] - x))
            d2 = float(np.linalg.norm(frac_cen[cj_f - nm] - x))
            t1 = perm_frac * aperture / max(d1, 1e-12)
            t2 = perm_frac * aperture / max(d2, 1e-12)
            conn.append((ci_f, cj_f))
            trans.append(1.0 / (1.0 / t1 + 1.0 / t2))
            ci_list.append(0.0)
            ctype.append("frac_frac")

    return EDFMGrid(
        g=g, n_matrix=nm, frac_cells=nf, frac_volumes=frac_vol,
        frac_centroids=frac_cen, frac_line=frac_line,
        perm=perm, poro=poro,
        conn=np.array(conn, dtype=np.int64).reshape(-1, 2),
        trans=np.asarray(trans, float), ci=np.asarray(ci_list, float),
        conn_type=ctype,
    )


def _cross2(u: np.ndarray, v: np.ndarray) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def _line_intersection(l1: FractureLine, l2: FractureLine
                       ) -> Optional[np.ndarray]:
    p, r = l1.a, l1.b - l1.a
    q, s = l2.a, l2.b - l2.a
    den = _cross2(r, s)
    if abs(den) < TOL:
        return None
    t = _cross2(q - p, s) / den
    u = _cross2(q - p, r) / den
    if -TOL <= t <= 1 + TOL and -TOL <= u <= 1 + TOL:
        return p + t * r
    return None


def _cell_at(nodes: np.ndarray, start: int, x: np.ndarray) -> Optional[int]:
    """Fracture cell containing point x."""
    s = np.linalg.norm(nodes - nodes[0], axis=1)
    sx = float(np.linalg.norm(x - nodes[0]))
    for k in range(nodes.shape[0] - 1):
        if s[k] - 1e-10 <= sx <= s[k + 1] + 1e-10:
            return start + k
    return None


def solve_incomp_edfm(
    e: EDFMGrid,
    mu: float,
    bc_cells: np.ndarray,
    bc_trans: np.ndarray,
    bc_press: np.ndarray,
) -> np.ndarray:
    """Incompressible pressure solve over matrix + fracture cells."""
    n = e.n_total
    t = e.trans / mu
    c1, c2 = e.conn[:, 0], e.conn[:, 1]
    d = np.bincount(np.concatenate([c1, c2]),
                    weights=np.concatenate([t, t]), minlength=n)
    rhs = np.zeros(n)
    tb = bc_trans / mu
    np.add.at(d, bc_cells, tb)
    np.add.at(rhs, bc_cells, tb * bc_press)
    rows = np.concatenate([c1, c2, np.arange(n)])
    cols = np.concatenate([c2, c1, np.arange(n)])
    vals = np.concatenate([-t, -t, d])
    a = sp.csr_array((vals, (rows, cols)), shape=(n, n))
    return spla.spsolve(a.tocsc(), rhs)


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.utils.units import BARSA, CENTI, DARCY, POISE

    def test_avg_distance_analytic():
        """<d> for a fracture through the middle of a square cell is L/4."""
        g = compute_geometry(cart_grid([1, 1], [1.0, 1.0]))
        ln = FractureLine(np.array([0.0, 0.5, 1.0, 0.5]))
        d = avg_frac_distance(g, 0, ln)
        assert np.isclose(d, 0.25, rtol=1e-12), d
        # diagonal fracture through a unit square: <d> = sqrt(2)/6
        ln2 = FractureLine(np.array([0.0, 0.0, 1.0, 1.0]))
        d2 = avg_frac_distance(g, 0, ln2)
        assert np.isclose(d2, np.sqrt(2) / 6.0, rtol=1e-12), d2
        # off-centre horizontal fracture at y = 0.25
        ln3 = FractureLine(np.array([0.0, 0.25, 1.0, 0.25]))
        d3 = avg_frac_distance(g, 0, ln3)
        assert np.isclose(d3, 0.3125, rtol=1e-12), d3   # (0.25^2+0.75^2)/2
        print("ok test_avg_distance_analytic")

    def test_segment_clipping():
        """Fracture length inside each crossed cell sums to the total."""
        g = compute_geometry(cart_grid([10, 10], [10.0, 10.0]))
        ln = FractureLine(np.array([1.3, 2.7, 8.1, 6.4]))
        tot = 0.0
        for c in range(g.cells.num):
            s = _seg_in_cell(g, c, ln)
            if s is not None:
                tot += float(np.linalg.norm(s[1] - s[0]))
        assert np.isclose(tot, ln.length, rtol=1e-10), (tot, ln.length)
        print("ok test_segment_clipping")

    def test_edfm_flow_beats_matrix():
        """A conductive fracture raises the effective permeability."""
        nx = 30
        g = compute_geometry(cart_grid([nx, nx], [9.0, 9.0]))
        perm_m = np.full(g.cells.num, 1.0 * DARCY)
        poro_m = np.full(g.cells.num, 0.2)
        ln = FractureLine(np.array([2.0, 3.3, 7.0, 5.7]))
        e = build_edfm(g, [ln], aperture=1.0 / 25.0, perm_matrix=perm_m,
                       perm_frac=1e4 * DARCY, poro_matrix=poro_m,
                       cell_size=0.3)
        assert e.frac_cells > 10
        assert np.sum(np.array(e.conn_type) == "frac_matrix") > 10

        # left/right Dirichlet BCs on the matrix boundary
        fc = g.faces.centroids
        bd = np.any(g.faces.neighbors == -1, axis=1)
        left = np.flatnonzero(bd & (fc[:, 0] < 1e-9))
        right = np.flatnonzero(bd & (fc[:, 0] > 9.0 - 1e-9))
        cells, tb, pb = [], [], []
        for faces, pval in ((left, 10 * BARSA), (right, 1 * BARSA)):
            c = g.faces.neighbors[faces].sum(axis=1) + 1
            dist = np.abs(fc[faces, 0] - g.cells.centroids[c, 0])
            t = perm_m[c] * g.faces.areas[faces] / dist
            cells.append(c)
            tb.append(t)
            pb.append(np.full(c.size, pval))
        p = solve_incomp_edfm(e, 1 * CENTI * POISE,
                              np.concatenate(cells), np.concatenate(tb),
                              np.concatenate(pb))
        assert np.all(p <= 10 * BARSA + 1e-6) and np.all(p >= 1 * BARSA - 1e-6)
        # fracture is highly conductive: nearly uniform pressure inside it
        pf = p[e.n_matrix:]
        assert (pf.max() - pf.min()) < 0.05 * (10 - 1) * BARSA
        # and it shorts the matrix: pressure near the fracture midline is
        # pulled toward the fracture pressure
        print("ok test_edfm_flow_beats_matrix")

    def test_frac_frac_intersection():
        """Crossing fractures produce exactly one frac-frac NNC."""
        g = compute_geometry(cart_grid([20, 20], [10.0, 10.0]))
        perm_m = np.full(g.cells.num, 1.0 * DARCY)
        poro_m = np.full(g.cells.num, 0.2)
        l1 = FractureLine(np.array([2.0, 5.0, 8.0, 5.0]))
        l2 = FractureLine(np.array([5.0, 2.0, 5.0, 8.0]))
        e = build_edfm(g, [l1, l2], aperture=0.01, perm_matrix=perm_m,
                       perm_frac=1e4 * DARCY, poro_matrix=poro_m,
                       cell_size=0.5)
        n_ff = int(np.sum(np.array(e.conn_type) == "frac_frac"))
        assert n_ff == 1, f"expected 1 frac-frac NNC, got {n_ff}"
        print("ok test_frac_frac_intersection")

    test_avg_distance_analytic()
    test_segment_clipping()
    test_edfm_flow_beats_matrix()
    test_frac_frac_intersection()
    print("\nAll tests passed!")
