"""One-sided NFVM flux stencils, ported from MRST ``findOSflux.m``."""

from itertools import combinations

import numpy as np

from pymrst.params.rock.perm_tensor import perm_tensor


def _decompose(g, hap, cell, direction):
    lo, hi = g.cells.facePos[cell:cell + 2]
    faces = np.asarray(g.cells.faces[lo:hi], dtype=np.int64)
    vectors = hap.coords[faces] - g.cells.centroids[cell]
    lengths = np.linalg.norm(vectors, axis=1)
    bases = vectors / lengths[:, None]
    dnorm = np.linalg.norm(direction)
    unit = direction / dnorm
    order = np.argsort(np.arccos(np.clip(bases @ unit, -1.0, 1.0)))
    fallback = []
    dim = g.griddim
    for chosen in combinations(order, dim):
        basis = np.column_stack([bases[i] for i in chosen])
        if abs(np.linalg.det(basis)) <= 1e-9:
            continue
        coeff = np.linalg.solve(basis, unit)
        coeff[np.abs(coeff) < 1e-9] = 0.0
        if np.all(coeff >= 0):
            candidate = (
                coeff*dnorm/lengths[np.asarray(chosen)],
                tuple(int(faces[i]) for i in chosen),
            )
            if np.all(coeff <= 1):
                return candidate
            fallback.append((float(np.max(coeff)), candidate))
    if fallback:
        return min(fallback, key=lambda item: item[0])[1]
    raise ValueError(f"one-sided flux decomposition failed for cell {cell}")


def _unique_trans(rows):
    order = []
    values = {}
    for cell, value in rows:
        cell = int(cell)
        if cell not in values:
            order.append(cell)
            values[cell] = 0.0
        values[cell] += float(value)
    out = np.array([[cell, values[cell]] for cell in order], dtype=float)
    out[1:, 1] *= -1.0
    if out.shape[0] > 2:
        out[2:] = out[2:][np.argsort(-out[2:, 0], kind="stable")]
    return out


def _stencil(g, hap, cell, other, direction):
    coeff, selected_faces = _decompose(g, hap, cell, direction)
    rows = [(cell, coeff.sum()), (other, 0.0)]
    for a, face in zip(coeff, selected_faces):
        for neighbor, weight in zip(g.faces.neighbors[face], hap.weights[face]):
            rows.append((neighbor, -a*weight))
    rows.append((-1, 0.0))
    return _unique_trans(rows)


def find_os_flux(g, rock, interp_face):
    """Return an ``(nface, 2)`` array of one-sided 2-D/3-D stencils."""
    if g.griddim not in (2, 3):
        raise NotImplementedError("NFVM supports 2-D and 3-D grids")
    dim = g.griddim
    tensors = perm_tensor(rock, dim)[0].reshape(g.cells.num, dim, dim)
    stencils = np.empty((g.faces.num, 2), dtype=object)
    stencils[:] = None
    for face, (c1, c2) in enumerate(np.asarray(g.faces.neighbors)):
        if c1 < 0 or c2 < 0:
            continue
        normal = np.asarray(g.faces.normals[face], dtype=float)
        stencils[face, 0] = _stencil(
            g, interp_face, c1, c2, tensors[c1] @ normal
        )
        stencils[face, 1] = _stencil(
            g, interp_face, c2, c1, -(tensors[c2] @ normal)
        )
    return stencils


findOSflux = find_os_flux

__all__ = ["find_os_flux", "findOSflux"]
