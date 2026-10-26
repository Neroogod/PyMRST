"""MRST book-ii exampleStateFunctionsBook and showConceptualAD with
TutorialNumber, TutorialProduct and TutorialAdd (ad_core/examples/book_ii)."""

import contextlib
import io

import numpy as np

from pymrst.modules.ad_core.examples.book_ii.example_state_functions_book import (
    example_state_functions_book,
)
from pymrst.modules.ad_core.examples.book_ii.show_conceptual_ad import show_conceptual_ad
from pymrst.utils.adi import value


def test_example_state_functions_book_caches_and_checks():
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = example_state_functions_book()
    assert result.G == 29 and result.ok is False
    assert out.getvalue().count("Retrieving a from state.") == 2


def test_show_conceptual_ad_values_agree_across_types():
    with contextlib.redirect_stdout(io.StringIO()):
        res = show_conceptual_ad(n=500, its=1, rng=np.random.default_rng(0))
    expected = res.a*res.b + res.x*res.y
    expected[res.subset] *= 2
    for kind in res.types:
        np.testing.assert_allclose(value(res.G[kind]), expected, rtol=1e-15)
