"""Reusable Jacobian structures and bounded batched property evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True, slots=True)
class SparsityKey:
    topology: str
    discretization: str
    variables: tuple[str, ...]
    facilities: str = ""
    controls: str = ""


@dataclass(frozen=True, slots=True)
class CSRPattern:
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]
    structural_hash: str

    @classmethod
    def from_matrix(cls, matrix):
        value = sp.csr_matrix(matrix, copy=True)
        value.sum_duplicates(); value.sort_indices()
        indices = value.indices.astype(np.int64, copy=True)
        indptr = value.indptr.astype(np.int64, copy=True)
        digest = hashlib.sha256()
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(indptr.tobytes()); digest.update(indices.tobytes())
        indices.flags.writeable = False; indptr.flags.writeable = False
        return cls(indices, indptr, value.shape, digest.hexdigest())

    @property
    def nonzeros(self):
        return int(self.indices.size)

    def refill(self, data, *, copy=True):
        values = np.asarray(data)
        if values.ndim != 1 or values.size != self.nonzeros:
            raise ValueError(f"expected {self.nonzeros} CSR values")
        # Indices are copied to prevent SciPy consumers from mutating the cache.
        return sp.csr_matrix((values.copy() if copy else values,
                              self.indices.copy(), self.indptr.copy()),
                             shape=self.shape, copy=False)


class SparsityPatternCache:
    def __init__(self):
        self._patterns = {}
        self._lock = threading.RLock()

    def put(self, key: SparsityKey, matrix):
        pattern = CSRPattern.from_matrix(matrix)
        with self._lock:
            self._patterns[key] = pattern
        return pattern

    def get(self, key):
        with self._lock:
            return self._patterns.get(key)

    def invalidate(self, *, topology=None, controls=None):
        with self._lock:
            keys = tuple(self._patterns)
            for key in keys:
                if ((topology is None or key.topology == topology)
                        and (controls is None or key.controls == controls)):
                    del self._patterns[key]

    def __len__(self):
        with self._lock:
            return len(self._patterns)


def batched_evaluate(function, *arrays, batch_size=65536):
    """Evaluate a vectorized property function without unbounded temporaries."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    values = tuple(np.asarray(value) for value in arrays)
    if not values:
        raise ValueError("at least one property input is required")
    count = values[0].shape[0]
    if any(value.ndim == 0 or value.shape[0] != count for value in values):
        raise ValueError("property inputs must share their leading dimension")
    chunks = [np.asarray(function(*(value[start:start+batch_size]
                                    for value in values)))
              for start in range(0, count, int(batch_size))]
    if not chunks:
        return np.asarray(function(*(value[:0] for value in values)))
    return np.concatenate(chunks, axis=0)


__all__ = ["CSRPattern", "SparsityKey", "SparsityPatternCache",
           "batched_evaluate"]
