"""Port of ADFNE15 ``Split.m``."""

from dataclasses import dataclass
import numpy as np

from .length import lengths
from .group import group
from .intersect import intersect
from .sort import sort_points


@dataclass(frozen=True, slots=True)
class SplitResult:
    lines: np.ndarray | tuple[np.ndarray, ...]
    source_ids: np.ndarray
    segment_counts: np.ndarray
    intersections: np.ndarray
    intersection_pairs: np.ndarray
    labels: np.ndarray


def split(lines, interval=0.0):
    lines = np.atleast_2d(np.asarray(lines, dtype=float)); count, width = lines.shape
    dimension = width//2
    if interval > 0:
        point_sets, counts = [], []
        for line in lines:
            distance = np.linalg.norm(line[dimension:]-line[:dimension])
            segments = max(1, int(np.floor(distance/interval))+1)
            points = np.linspace(line[:dimension], line[dimension:], segments+1)
            point_sets.append(points); counts.append(segments)
        return SplitResult(tuple(point_sets), np.arange(count), np.asarray(counts),
                           np.empty((0, dimension)), np.empty((0, 2), int),
                           -np.arange(1, count+1))
    if dimension != 2: raise ValueError("intersection splitting is defined for 2-D lines")
    intersections = intersect(lines); grouped, _ = group(
        intersections.geometry, intersections.pairs, count)
    pieces, source, counts = [], [], []
    for index, line in enumerate(lines):
        points = sort_points(np.vstack((grouped[index], line[:2], line[2:])), line)
        candidates = np.hstack((points[:-1], points[1:]))
        candidates = candidates[lengths(candidates) > 1e-14]
        pieces.extend(candidates); source.extend([index]*len(candidates)); counts.append(len(candidates))
    return SplitResult(np.asarray(pieces).reshape(-1, 4), np.asarray(source), np.asarray(counts),
                       intersections.geometry, intersections.pairs, intersections.labels)
