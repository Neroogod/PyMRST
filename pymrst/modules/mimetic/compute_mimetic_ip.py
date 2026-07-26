"""Mimetic inner products, ported from MRST ``computeMimeticIP.m``."""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from pymrst.params.rock.perm_tensor import perm_tensor


@dataclass
class MimeticSystem:
    """Block-diagonal mimetic mass matrix data."""

    type: str
    ip: str
    B: sp.csr_array | None = None
    BI: sp.csr_array | None = None

    def __getitem__(self, name):
        return getattr(self, name)


def _projector(matrix):
    q = la.orth(np.asarray(matrix, dtype=float))
    return np.eye(matrix.shape[0]) - q @ q.T


def _ip_simple(a, volume, permeability, centroids, normals, inverse):
    stabilization = 6.0*np.trace(permeability)/permeability.shape[0]
    if inverse:
        u = _projector(centroids*a[:, None])
        d = np.diag(a)
        return (
            normals @ permeability @ normals.T
            + stabilization*(d @ u @ d)
        )/volume
    u = _projector(normals/a[:, None])
    di = np.diag(1.0/a)
    return (
        centroids @ np.linalg.solve(permeability, centroids.T)/volume
        + (volume/stabilization)*(di @ u @ di)
    )


def _ip_qfamily(a, volume, permeability, centroids, normals, inverse, qparam):
    primary = normals @ permeability @ normals.T
    if inverse:
        u = _projector(centroids)
        diagonal = np.diag(np.diag(primary))
        return (primary + qparam*(u @ diagonal @ u))/volume
    u = _projector(normals)
    inverse_diagonal = np.diag(1.0/np.diag(primary))
    return (
        centroids @ np.linalg.solve(permeability, centroids.T)/volume
        + (volume/qparam)*(u @ inverse_diagonal @ u)
    )


def _ip_tpf(a, volume, permeability, centroids, normals, inverse):
    del a, volume
    td = np.sum(centroids*(normals @ permeability), axis=1)
    td /= np.sum(centroids*centroids, axis=1)
    values = np.abs(td) if inverse else 1.0/np.abs(td)
    return np.diag(values)


def _ip_quasitpf(a, volume, permeability, centroids, normals, inverse):
    del a
    u = _projector(centroids)
    primary = normals @ permeability @ normals.T
    matrix = (primary + 2.0*u @ np.diag(np.diag(primary)) @ u)/volume
    return matrix if inverse else np.linalg.inv(matrix)


def _ip_rt(a, volume, permeability, centroids, normals, inverse):
    ki = np.linalg.inv(permeability)
    di = np.diag(1.0/a)
    q = np.abs(normals.T) @ di/np.sqrt(2.0)
    diagonal = np.diag(np.diag(ki))*(volume/6.0)
    mask_values = np.abs(normals @ normals.T)
    mask = mask_values > np.finfo(float).eps*np.max(mask_values)
    matrix = mask*(
        centroids @ np.linalg.solve(permeability, centroids.T)/volume
        + di @ q.T @ diagonal @ q @ di
    )
    return np.linalg.inv(matrix) if inverse else matrix


def _inner_product(name, qparam):
    methods = {
        "ip_simple": _ip_simple,
        "ip_tpf": _ip_tpf,
        "ip_quasitpf": _ip_quasitpf,
        "ip_rt": _ip_rt,
        "ip_quasirt": lambda *args: _ip_qfamily(*args, 6.0),
        "ip_qfamily": lambda *args: _ip_qfamily(*args, qparam),
    }
    try:
        return methods[name.lower()]
    except KeyError as error:
        raise ValueError(f"unknown mimetic inner product {name!r}") from error


def compute_mimetic_ip(
    g, rock, *, type="hybrid", inner_product="ip_simple",
    facetrans=None, qparam=6.0, **options,
):
    """Assemble MRST's cellwise mimetic ``B`` and/or ``BI`` matrices."""
    type = options.pop("Type", type)
    inner_product = options.pop("InnerProduct", inner_product)
    facetrans = options.pop("facetrans", facetrans)
    qparam = options.pop("QParam", options.pop("qparam", qparam))
    options.pop("Verbose", options.pop("verbose", None))
    if options:
        raise TypeError(f"unknown options: {sorted(options)}")

    system_type = str(type).lower()
    if system_type == "hybrid":
        compute_b, compute_bi = False, True
    elif system_type in ("mixed", "tpfa"):
        compute_b, compute_bi = True, False
    elif system_type == "comp_hybrid":
        compute_b = compute_bi = True
    else:
        raise ValueError(f"unknown mimetic system type {type!r}")

    dim = int(g.griddim)
    permeability = perm_tensor(rock, dim)[0]
    if permeability.shape[0] != g.cells.num:
        raise ValueError("permeability must be defined in active cells")
    method = _inner_product(str(inner_product), float(qparam))
    cell_number = np.repeat(
        np.arange(g.cells.num), np.diff(np.asarray(g.cells.facePos))
    )
    faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
    sign = 2.0*(neighbors[faces, 0] == cell_number) - 1.0
    areas = np.asarray(g.faces.areas)[faces]
    centroid_vectors = (
        np.asarray(g.faces.centroids)[faces]
        - np.asarray(g.cells.centroids)[cell_number]
    )
    outward_normals = np.asarray(g.faces.normals)[faces]*sign[:, None]

    half_face_resistance = np.zeros(faces.size)
    if facetrans is not None:
        ft = np.asarray(facetrans, dtype=float)
        if ft.size:
            by_face = np.zeros(g.faces.num)
            by_face[ft[:, 0].astype(np.int64)] = ft[:, 1]
            count = np.sum(neighbors >= 0, axis=1)
            nonzero = by_face != 0
            by_face[nonzero] = 1.0/(by_face[nonzero]*count[nonzero])
            half_face_resistance = by_face[faces]

    b_blocks, bi_blocks = [], []
    for cell in range(g.cells.num):
        start, stop = map(int, g.cells.facePos[cell:cell + 2])
        ix = slice(start, stop)
        a = areas[ix]
        volume = float(g.cells.volumes[cell])
        k = permeability[cell].reshape(dim, dim)
        c = centroid_vectors[ix]
        n = outward_normals[ix]
        resistance = np.diag(half_face_resistance[ix])
        if compute_b:
            b_blocks.append(method(a, volume, k, c, n, False) + resistance)
        if compute_bi:
            inverse_ip = method(a, volume, k, c, n, True)
            if np.any(half_face_resistance[ix]):
                inverse_ip = np.linalg.inv(
                    np.linalg.inv(inverse_ip) + resistance
                )
            bi_blocks.append(inverse_ip)

    return MimeticSystem(
        type=system_type,
        ip=str(inner_product),
        B=sp.block_diag(b_blocks, format="csr") if compute_b else None,
        BI=sp.block_diag(bi_blocks, format="csr") if compute_bi else None,
    )


def computeMimeticIP(g, rock, **options):
    """MRST-compatible public spelling."""
    return compute_mimetic_ip(g, rock, **options)


__all__ = ["MimeticSystem", "compute_mimetic_ip", "computeMimeticIP"]
