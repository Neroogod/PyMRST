from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.utils import obj2hash


def test_hash_is_deterministic_and_sensitive_to_values():
    value = {
        "numeric": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "cell": ["yes", np.array([True, False])],
    }
    digest = obj2hash(value)
    assert len(digest) == 32
    assert digest == obj2hash(value)
    changed = dict(value)
    changed["numeric"] = value["numeric"].copy()
    changed["numeric"][0, 0] = 9.0
    assert obj2hash(changed) != digest


def test_skip_structure_and_size_fallback():
    value = SimpleNamespace(keep=np.array([1.0, 2.0]), ignored=99.0)
    digest, structure = obj2hash(
        value, skip=("ignored",), return_structure=True
    )
    assert set(structure) == {"keep"}
    assert digest == obj2hash(SimpleNamespace(keep=np.array([1.0, 2.0])))
    assert obj2hash(np.array([1.0, 2.0]), max_size=1) == obj2hash("double")


def test_empty_values_use_mrst_nonempty_sentinel():
    expected = obj2hash("[]")
    assert obj2hash([]) == expected
    assert obj2hash(None) == expected
    assert obj2hash(np.array([])) == expected
