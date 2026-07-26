"""HAP correction, ported from MRST ``correctHAP.m``."""

import numpy as np

from pymrst.modules.nfvm.solvers.utils.find_hap import _inside_convex_hull


def _radius(g):
    areas = np.asarray(g.faces.areas, dtype=float)
    return 0.5*areas if g.griddim == 2 else np.sqrt(areas/np.pi)


def _correct_face(g, hap, original, face, ratio):
    center = np.asarray(g.faces.centroids[face], dtype=float)
    direction = original[face] - center
    norm = np.linalg.norm(direction)
    if ratio <= 0 or norm == 0:
        hap.coords[face] = center
    else:
        hap.coords[face] = center + ratio*_radius(g)[face]*direction/norm


def _refresh_hull_status(g, hap):
    status = np.zeros(g.cells.num, dtype=bool)
    for cell in range(g.cells.num):
        lo, hi = g.cells.facePos[cell:cell + 2]
        faces = np.asarray(g.cells.faces[lo:hi], dtype=np.int64)
        status[cell] = _inside_convex_hull(
            np.asarray(g.cells.centroids[cell]), hap.coords[faces]
        )
    hap.counter = status
    hap.fraction = 1.0 - float(np.mean(status))


def correct_hap(g, interp_face, ratio=None, max_iterations=10000):
    """Correct ill-positioned HAPs in place and return ``interp_face``.

    With an explicit ``ratio``, every HAP farther than ``ratio`` face radii
    from its face centroid is clipped. Without one, MRST's hull-repair logic
    repeatedly clips the most displaced HAP of an offending cell.
    """
    hap = interp_face
    original = np.asarray(hap.coords, dtype=float).copy()
    radii = _radius(g)
    if ratio is not None:
        distance = np.linalg.norm(hap.coords - g.faces.centroids, axis=1)
        selected = np.flatnonzero(distance > ratio*radii)
        for face in selected:
            _correct_face(g, hap, original, int(face), float(ratio))
        hap.corrected = selected.size/g.faces.num
        _refresh_hull_status(g, hap)
        return hap

    _refresh_hull_status(g, hap)
    if hap.fraction == 0:
        hap.corrected = 0.0
        return hap
    touched = set()
    for _ in range(max_iterations):
        bad = np.flatnonzero(~hap.counter)
        if bad.size == 0:
            break
        changed = False
        for cell in bad:
            lo, hi = g.cells.facePos[cell:cell + 2]
            faces = np.asarray(g.cells.faces[lo:hi], dtype=np.int64)
            distance = np.linalg.norm(
                hap.coords[faces] - g.faces.centroids[faces], axis=1
            )
            normalized = distance/radii[faces]
            local = int(np.argmax(normalized))
            if normalized[local] <= 1e-14:
                continue
            face = int(faces[local])
            _correct_face(g, hap, original, face, 0.9*normalized[local])
            touched.add(face)
            changed = True
        _refresh_hull_status(g, hap)
        if not changed:
            break
    if hap.fraction > 0:
        raise RuntimeError("HAP hull correction did not converge")
    hap.corrected = len(touched)/g.faces.num
    return hap


correctHAP = correct_hap

__all__ = ["correct_hap", "correctHAP"]
