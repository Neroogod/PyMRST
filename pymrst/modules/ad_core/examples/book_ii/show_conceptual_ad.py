"""MRST ``ad-core/examples/book-ii/showConceptualAD.m``.

The cost of automatic differentiation, conceptually: the state-function
grouping `G = A*B + X*Y` of `exampleStateFunctionsBook`, evaluated `its` times
on `n` random values each of `a`, `b`, `x` and `y`, as plain doubles, as sparse
AD (`initVariablesADI`) and as diagonal AD (`initVariablesAD_diagonal`). Each
evaluation is timed, then taking 1 % of the values at scattered positions,
doubling that subset, and writing it back. The mean times are shown in bars,
absolute and relative to doubles. MRST's defaults are `n = 1e6` and
`its = 100`, unless already set; every evaluation prints the tutorial
functions' messages, as in MRST. The script ends by displaying `[mod(v-1,5)+1,
v]` for `v = 1:50`, which the port prints as well.

MRST's random values cannot be reproduced, and the times are this machine's,
so nothing here is compared with MRST; the values and derivatives of `G` are
checked against `a*b + x*y` instead.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.backends.diagonal.utils.init_variables_ad_diagonal import (
    init_variables_ad_diagonal,
)
from pymrst.modules.ad_core.examples.book_ii.example_state_functions_book import (
    tutorial_group,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import init_variables_adi

TYPES = ("double", "Sparse AD", "Diagonal AD")
OPNAMES = ("Compute", "Subset", "Subcompute", "Insertion")
SUBSET_STRIDE = 100


def show_conceptual_ad(n: int = 10**6, its: int = 100, plot: bool = False,
                       rng: np.random.Generator | None = None) -> SimpleNamespace:
    """MRST `showConceptualAD`: mean times per AD type and operation."""
    group = tutorial_group()
    group.disp()
    use_mex = True
    subset = np.arange(0, n, SUBSET_STRIDE)
    rng = np.random.default_rng() if rng is None else rng
    a, b, x, y = (rng.random(n) for _ in range(4))
    nt = len(TYPES)
    t_assign, t_compute, t_subset, t_subcompute = (np.zeros(nt) for _ in range(4))
    last = {}
    for i, kind in enumerate(TYPES):
        if kind == "double":
            values = (a, b, x, y)
        elif kind == "Sparse AD":
            values = init_variables_adi(a, b, x, y)
        else:
            values = init_variables_ad_diagonal(a, b, x, y, use_mex=use_mex)
        state0 = State(a=values[0], b=values[1], x=values[2], y=values[3])
        for _ in range(its):
            state = group.init_state_function_container(State(state0))
            tic = time.perf_counter()
            G = group.get(None, state, "G")
            t_compute[i] += time.perf_counter() - tic
            tic = time.perf_counter()
            G_sub = G[subset]
            t_subset[i] += time.perf_counter() - tic
            tic = time.perf_counter()
            v = 2*G_sub
            t_subcompute[i] += time.perf_counter() - tic
            tic = time.perf_counter()
            G[subset] = v
            t_assign[i] += time.perf_counter() - tic
        last[kind] = G
    data = np.column_stack([t_compute, t_subset, t_subcompute, t_assign])/its
    if plot:
        _plot(data)
    v = np.arange(1, 5*10 + 1)
    print(np.column_stack([(v - 1) % 5 + 1, v]))
    return SimpleNamespace(types=TYPES, opnames=OPNAMES, data=data, a=a, b=b, x=x, y=y,
                           subset=subset, G=last)


def _plot(data: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    nt, nop = data.shape
    width = 0.8/nop
    pos = np.arange(nt)

    def bars(ax, values, labels, annotate=None):
        for j in range(values.shape[1]):
            h = ax.bar(pos[:values.shape[0]] + (j - (nop - 1)/2)*width, values[:, j], width,
                       label=OPNAMES[j])
            if annotate is not None:
                for rect, txt in zip(h, annotate[:, j]):
                    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f"{txt:2.1f}",
                            ha="center", va="bottom")
        ax.set_xticks(pos[:values.shape[0]], labels)
        ax.legend()

    fig, ax = plt.subplots()
    bars(ax, data, TYPES)
    rel = data[1:, :]/data[0, :]
    fig, ax = plt.subplots()
    bars(ax, rel, TYPES[1:], annotate=rel)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_ylabel("Time relative to double")
    fig, ax = plt.subplots(figsize=(7.2, 1.8))
    bars(ax, data, TYPES, annotate=data/data[0, :])
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_ylabel("Time [s]")


showConceptualAD = show_conceptual_ad

__all__ = ["showConceptualAD", "show_conceptual_ad"]


if __name__ == "__main__":
    import contextlib
    import io

    from pymrst.utils.adi import value

    def test_g_and_its_derivatives_for_every_type():
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            res = show_conceptual_ad(n=1000, its=2, rng=np.random.default_rng(3))
        expected = res.a*res.b + res.x*res.y
        expected[res.subset] *= 2
        for kind in TYPES:
            np.testing.assert_allclose(value(res.G[kind]), expected, rtol=1e-15)
        jac = res.G["Sparse AD"].jac
        scale = np.ones(1000)
        scale[res.subset] = 2
        for block, factor in zip(jac, (res.b, res.a, res.y, res.x)):
            np.testing.assert_allclose(block.diagonal(), scale*factor, rtol=1e-15)
        assert res.data.shape == (3, 4) and np.all(res.data >= 0)
        assert out.getvalue().count("Adding AB and XY.") == 3*2
        print("ok test_g_and_its_derivatives_for_every_type")

    test_g_and_its_derivatives_for_every_type()
    print("\nAll tests passed!")
