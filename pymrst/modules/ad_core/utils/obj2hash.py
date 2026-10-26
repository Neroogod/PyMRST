"""Deterministic MRST-compatible recursive object hashing."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
from typing import Any

import numpy as np
import scipy.sparse as sp


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _combine(hashes) -> str:
    return _md5_bytes("_".join(hashes).encode("latin-1"))


def _matlab_class_name(value) -> str:
    if isinstance(value, (float, int, complex, np.floating, np.integer)):
        return "double"
    if isinstance(value, (bool, np.bool_)):
        return "logical"
    if isinstance(value, str):
        return "char"
    if isinstance(value, np.ndarray):
        names = {
            "f8": "double", "f4": "single",
            "i1": "int8", "i2": "int16", "i4": "int32", "i8": "int64",
            "u1": "uint8", "u2": "uint16", "u4": "uint32", "u8": "uint64",
            "b1": "logical", "c8": "single", "c16": "double",
        }
        return names.get(value.dtype.str[-2:], str(value.dtype))
    return type(value).__name__


def _numeric_bytes(value) -> bytes:
    if isinstance(value, (float, int, complex)):
        value = np.asarray(value, dtype=np.complex128 if isinstance(value, complex)
                           else np.float64)
    array = np.asarray(value)
    if array.dtype == bool:
        return array.ravel(order="F").astype(np.uint8).tobytes()
    return np.asarray(array).tobytes(order="F")


def _char_bytes(value: str) -> bytes:
    # MATLAB md5sum converts char values to uint8 before updating the digest.
    return bytes((ord(character) & 0xFF for character in value))


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes, list, tuple, dict)):
        return len(value) == 0
    if isinstance(value, np.ndarray) or sp.issparse(value):
        return value.size == 0
    return False


def _properties(value):
    if isinstance(value, dict):
        return list(value.items())
    if is_dataclass(value) and not isinstance(value, type):
        return [(field.name, getattr(value, field.name)) for field in fields(value)]
    if hasattr(value, "__dict__"):
        return [
            (name, item) for name, item in vars(value).items()
            if not name.startswith("_") and not callable(item)
        ]
    return []


def _hash(value, skip, max_size, path=frozenset()):
    """`path` holds the containers being hashed above this one. A MATLAB value
    cannot contain itself; a Python object can (a model's facility points
    back at the model), and a container met again on its own path hashes as
    its class name plus a cycle marker."""
    if _is_empty(value):
        value = "[]"

    if sp.issparse(value):
        matrix = value.tocsc()
        nonzero = np.count_nonzero(matrix.data)
        if nonzero > max_size:
            return _md5_bytes(_char_bytes(_matlab_class_name(value))), None
        # The MEX hashes MATLAB's CSC buffers directly: values, zero-based row
        # indices, then column pointers. mwIndex is 64-bit on modern MATLAB.
        payload = np.asarray(matrix.data, dtype=np.float64).tobytes()
        payload += np.asarray(matrix.indices, dtype=np.uint64).tobytes()
        payload += np.asarray(matrix.indptr, dtype=np.uint64).tobytes()
        return _md5_bytes(payload), None

    if isinstance(value, str):
        if sum(character != "\0" for character in value) > max_size:
            value = "char"
        return _md5_bytes(_char_bytes(value)), None

    if isinstance(value, (bool, int, float, complex, np.number, np.ndarray)):
        array = np.asarray(value)
        if np.count_nonzero(array) > max_size:
            return _md5_bytes(_char_bytes(_matlab_class_name(value))), None
        return _md5_bytes(_numeric_bytes(value)), None

    if id(value) in path:
        return _md5_bytes(_char_bytes(_matlab_class_name(value) + ":cycle")), None
    path = path | {id(value)}

    if isinstance(value, (list, tuple)):
        hashes = [_hash(item, skip, max_size, path)[0] for item in value]
        return _combine(hashes), None

    properties = [
        (name, item) for name, item in _properties(value) if name not in skip
    ]
    if not properties:
        class_name = "struct" if isinstance(value, dict) else _matlab_class_name(value)
        return _hash(class_name, skip, max_size, path)[0], {}

    hash_structure = {}
    hashes = []
    for name, item in properties:
        item_hash, _ = _hash(item, skip, max_size, path)
        hash_structure[name] = item_hash
        hashes.append(item_hash)
    return _combine(hashes), hash_structure


def obj2hash(
    obj: Any,
    *,
    skip=(),
    max_size: int = 100_000_000,
    maxSize: int | None = None,
    return_structure: bool = False,
    verbose: bool | None = None,
):
    """Hash an object recursively using MRST ``obj2hash`` semantics.

    Set ``return_structure=True`` to also return the per-field hash dictionary,
    corresponding to MATLAB's optional second output.
    """
    del verbose
    if maxSize is not None:
        max_size = maxSize
    digest, structure = _hash(obj, frozenset(skip), int(max_size))
    return (digest, structure) if return_structure else digest


__all__ = ["obj2hash"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_an_object_that_contains_itself_hashes_finitely():
        model = SimpleNamespace(a=np.arange(3.0))
        model.facility = SimpleNamespace(owner=model)
        first, second = obj2hash(model), obj2hash(model)
        assert first == second and len(first) == 32
        print("ok test_an_object_that_contains_itself_hashes_finitely")

    def test_acyclic_objects_hash_as_before():
        a = SimpleNamespace(x=np.arange(3.0), y="name")
        b = SimpleNamespace(x=np.arange(3.0), y="name")
        assert obj2hash(a) == obj2hash(b)
        assert obj2hash(a) != obj2hash(SimpleNamespace(x=np.arange(4.0), y="name"))
        print("ok test_acyclic_objects_hash_as_before")

    def test_a_shared_but_acyclic_child_is_hashed_in_full_each_time():
        child = SimpleNamespace(v=np.ones(2))
        assert obj2hash(SimpleNamespace(a=child, b=child)) == \
            obj2hash(SimpleNamespace(a=SimpleNamespace(v=np.ones(2)),
                                     b=SimpleNamespace(v=np.ones(2))))
        print("ok test_a_shared_but_acyclic_child_is_hashed_in_full_each_time")

    test_an_object_that_contains_itself_hashes_finitely()
    test_acyclic_objects_hash_as_before()
    test_a_shared_but_acyclic_child_is_hashed_in_full_each_time()
    print("\nAll tests passed!")
