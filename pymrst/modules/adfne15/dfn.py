"""Native 2-D/3-D DFN generation corresponding to ADFNE15 ``DFN.m``."""

from dataclasses import dataclass
import numpy as np

from .angle import angle
from .clip import clip_lines, clip_polygons
from .length import lengths
from .rand import random_values


@dataclass(frozen=True, slots=True)
class DFNResult:
    dimension: int
    original: np.ndarray | tuple[np.ndarray, ...]
    fractures: np.ndarray | tuple[np.ndarray, ...]

    @property
    def Line(self):
        return self.fractures if self.dimension == 2 else None

    @property
    def Poly(self):
        return self.fractures if self.dimension == 3 else None

    @property
    def Orig(self):
        return self.original


def generate_dfn(*, dimension=2, count=100, minimum_length=.05,
                 mean_length=.3, maximum_length=.6, box=None, direction=0,
                 direction_variation=-1e-7, dip=45, dip_variation=-1e-7,
                 positions=None, shape="circle", polygon_segments=24,
                 scale=1.0, seed=None):
    if dimension not in {2, 3} or count < 0:
        raise ValueError("dimension must be 2 or 3 and count nonnegative")
    if not 0 <= minimum_length <= maximum_length or mean_length <= 0:
        raise ValueError("invalid fracture length distribution")
    rng = np.random.default_rng(seed)
    half_lengths = .5*random_values(
        count, distribution="exp", bounds=(minimum_length, maximum_length),
        mean=mean_length, rng=rng)
    if dimension == 2:
        box = np.asarray((0, 0, 1, 1) if box is None else box, dtype=float)
        points = (rng.uniform(box[:2], box[2:], size=(count, 2))
                  if positions is None else np.asarray(positions, dtype=float))
        directions = _orientations(count, direction, direction_variation, 360, rng)
        radians = np.radians(directions)
        delta = half_lengths[:, None]*np.column_stack((np.cos(radians), np.sin(radians)))
        original = np.hstack((points-delta, points+delta))
        return DFNResult(2, original, clip_lines(original, box))
    box = np.asarray((0, 0, 0, 1, 1, 1) if box is None else box, dtype=float)
    points = (rng.uniform(box[:3], box[3:], size=(count, 3))
              if positions is None else np.asarray(positions, dtype=float))
    dips = _orientations(count, angle(dip, 90), dip_variation, 90, rng)
    directions = _orientations(count, angle(direction, 360), direction_variation, 360, rng)
    polygons = []
    for i in range(count):
        base = _base_polygon(shape, 2*half_lengths[i], polygon_segments)
        rd, rp = np.radians(directions[i]), np.radians(dips[i])
        rz = np.array([[np.cos(rd), -np.sin(rd), 0],
                       [np.sin(rd), np.cos(rd), 0], [0, 0, 1]])
        ry = np.array([[np.cos(rp), 0, np.sin(rp)], [0, 1, 0],
                       [-np.sin(rp), 0, np.cos(rp)]])
        polygons.append(base @ (rz @ ry).T*scale+points[i])
    clipped = clip_polygons(polygons, box)
    clipped = [polygon for polygon, size in zip(clipped, lengths(clipped))
               if size > .9*minimum_length]
    return DFNResult(3, tuple(polygons), tuple(clipped))


def _orientations(count, mean, variation, limit, rng):
    if variation > 0:
        return rng.uniform(max(0, mean-variation), min(limit, mean+variation), count)
    if variation < 0:
        return random_values(count, distribution="fisher", bounds=(0, limit),
                             mean=mean, kappa=-variation, rng=rng)
    return np.full(count, mean, dtype=float)


def _base_polygon(shape, length, segments):
    name = str(shape).lower()
    if name in {"c", "circle", "e", "ellipse"}:
        angles = np.linspace(0, 2*np.pi, int(segments), endpoint=False)
        ratio = .5 if name in {"e", "ellipse"} else 1.0
        return np.column_stack((np.zeros(angles.size), .5*length*np.cos(angles),
                                .5*length*ratio*np.sin(angles)))
    if name in {"s", "square"}:
        radius = .5*length
        return np.array([[0, -radius, -radius], [0, radius, -radius],
                         [0, radius, radius], [0, -radius, radius]], dtype=float)
    raise ValueError(f"unsupported fracture shape: {shape}")
