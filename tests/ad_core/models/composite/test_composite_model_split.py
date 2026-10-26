"""Splitting a composite model's problem and variables (ad_core port).

A composite model is several submodels solved together, and two operations
have to survive that: splitting the assembled problem back into per-submodel
pieces, and flattening a nested variable tree for the AD layer then rebuilding
it afterwards.

The flatten/rebuild pair has an exact property - they are inverses - which is
the kind of claim worth testing directly rather than through a simulation.

One divergence from MRST is recorded here: its `packVars` takes the node count
from the current nesting level only, so a leaf sitting beside a nested list is
dropped without a word. This port refuses instead, because a silently shorter
variable list produces an update that is applied to the wrong unknowns.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.modules.ad_core.models.composite.composite_model import (
    pack_vars, unpack_vars,
)


# -- the inverse pair ------------------------------------------------------

@pytest.mark.parametrize("tree", [
    [["a", "b"], ["c"], ["d", "e", "f"]],
    [["x"]],
    [["a"], ["b", "c"]],
    [["p", "sW"], ["bhp", "qWs"]],
])
def test_flattening_and_rebuilding_are_inverses(tree):
    """The property the pair exists for."""
    flat, levels, nodes = unpack_vars(tree)
    assert pack_vars(flat, levels, nodes) == tree


def test_flattening_preserves_order():
    """The flat list feeds the AD layer, which returns results positionally -
    a reordering here would return each submodel another's increments."""
    flat, _, _ = unpack_vars([["a", "b"], ["c"], ["d"]])
    assert flat == ["a", "b", "c", "d"]


def test_the_index_lists_record_where_each_leaf_came_from():
    """Without them a flat list cannot say which submodel each entry belonged
    to, and the rebuild is impossible."""
    _, levels, nodes = unpack_vars([["a", "b"], ["c"]])
    assert levels == [2, 2, 2]
    assert nodes == [1, 1, 2]


def test_the_starting_level_is_derived_rather_than_assumed():
    """MRST hard-codes 2 at its one call site, which works only because its
    trees are uniformly one level deep. Deriving it means the round trip holds
    without the caller knowing the depth."""
    tree = [["a", "b"], ["c"]]
    flat, levels, nodes = unpack_vars(tree)
    assert pack_vars(flat, levels, nodes) == pack_vars(flat, levels, nodes, 2)


def test_a_mixed_depth_tree_is_refused_rather_than_truncated():
    """MRST's `max(mapn(subsetl))` counts nodes at the current level only, so
    a leaf beside a nested list is dropped silently - and a shorter variable
    list means an update applied to the wrong unknowns.
    """
    flat, levels, nodes = unpack_vars(["x", ["a", "b"]])
    assert len(flat) == 3
    with pytest.raises(ValueError, match="mixes nesting depths"):
        pack_vars(flat, levels, nodes)


def test_an_empty_tree_round_trips():
    flat, levels, nodes = unpack_vars([])
    assert flat == [] and pack_vars(flat, levels, nodes) == []


# -- splitting a problem ---------------------------------------------------

def _problem(state):
    from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearizedProblem
    return LinearizedProblem.from_equations(
        ["e1", "e2", "e3"],
        ["cell", "cell", "well"],
        ["flow.water", "flow.oil", "mech.momentum"],
        ["flow.pressure", "flow.sW", "mech.u"],
        state, 1.0)


def _composite():
    from pymrst.modules.ad_core.models.composite.composite_model import (
        CompositeModel,
    )
    return CompositeModel({"flow": object(), "mech": object()})


def test_equations_are_assigned_by_their_submodel_prefix():
    model = _composite()
    state = {"flow": {"pressure": 1.0}, "mech": {"u": 0.0}}
    problems, mapping = model.split_problem(_problem(state))

    assert len(problems) == 2
    assert problems[0].equation_names == ["water", "oil"]
    assert problems[1].equation_names == ["momentum"]
    assert mapping == [0, 0, 1]


def test_the_prefix_is_stripped_so_each_subproblem_stands_alone():
    """A subproblem still carrying `flow.` in its names would not match the
    submodel's own equation names."""
    model = _composite()
    state = {"flow": {}, "mech": {}}
    problems, _ = model.split_problem(_problem(state))
    assert all("." not in name for p in problems for name in p.equation_names)


def test_each_subproblem_carries_its_own_state_and_timestep():
    model = _composite()
    state = {"flow": {"pressure": 7.0}, "mech": {"u": 0.0}}
    problems, _ = model.split_problem(_problem(state))
    assert problems[0].state == {"pressure": 7.0}
    assert problems[0].dt == 1.0


def test_an_inactive_submodel_contributes_no_subproblem():
    """`get_active_models` filters on the state, so a submodel absent from the
    state is not solved for."""
    model = _composite()
    problems, _ = model.split_problem(_problem({"flow": {}}))
    assert len(problems) == 1
    assert problems[0].equation_names == ["water", "oil"]


# -- primary variable names -------------------------------------------------

def test_each_submodel_records_its_own_primary_variables():
    model = _composite()
    state = {"flow": {}, "mech": {}}
    model.set_primary_variables(state, [["pressure", "sW"], ["u"]])
    assert state["flow"]["primaryVariables"] == ["pressure", "sW"]
    assert state["mech"]["primaryVariables"] == ["u"]


def test_a_name_list_per_active_submodel_is_required():
    """A mismatch would silently give one submodel another's variable names."""
    model = _composite()
    with pytest.raises(ValueError, match="for 2 active submodels"):
        model.set_primary_variables({"flow": {}, "mech": {}}, [["pressure"]])
