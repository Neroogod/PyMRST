"""Order edges into connected chains (MRST ``private/sortedges.m``).

Given a set of two-node edges grouped into sections, put each section's edges
into the order you would meet them walking along the chain they form, and report
which way round each one was traversed.

That is the operation behind `sort_sub_edges`: the fine faces of a coarse face
arrive in whatever order the fine grid numbered them, which is not the order
they lie in along the face.

The walk starts at the section's first edge and grows in both directions, so the
chain comes back running from one end to the other, with the first edge always
traversed forwards. `sign` is +1 where an edge is traversed from its first node
to its second and -1 where it is traversed the other way, which is what lets a
caller read a consistent direction along the whole chain.

`success` is per section and is false when the section's edges do not form a
single chain. That is a real case rather than an error: a coarse face split
across a fault is a disconnected set of fine faces, and MRST warns rather than
raising. The port returns the flag and leaves the decision to the caller.

**A failed section still returns a full permutation here, and does not in
MRST.** MATLAB ends with `map = map(map > 0)`, which drops the slots the walk
never wrote, so a section that fails contributes fewer entries than it has
edges. Its caller then does `fconn = fconn(j)` regardless, and the packed
`connPos` layout no longer describes the array it indexes: every coarse face
after the failed one reads the wrong fine faces. The warning is printed, and
nothing else marks it. This port appends the unreachable edges in their original
order instead, so the permutation stays total and `success` is the only thing
that reports the problem.

MRST's implementation is vectorised across sections with a fixed `2n-1` write
window and a scan over candidate positions. This walks each section directly,
which is the same result: both take the lowest-numbered unused edge that
continues the chain.
"""

from __future__ import annotations

import numpy as np


def sort_edges(edges, pos=None):
    """Return ``(order, sign, success)``.

    `order` indexes into `edges` and is a permutation within each section.
    `sign` is aligned with `order`. `success` has one entry per section.
    """
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must be an n-by-2 array, got {edges.shape}")
    if pos is None:
        pos = np.array([0, edges.shape[0]], dtype=np.int64)
    pos = np.asarray(pos, dtype=np.int64).ravel()
    if pos.size < 2 or pos[0] != 0 or pos[-1] != edges.shape[0]:
        raise ValueError(
            f"pos must run from 0 to {edges.shape[0]}, got {pos[0]} to {pos[-1]}")

    n_sections = pos.size - 1
    order = np.empty(edges.shape[0], dtype=np.int64)
    sign = np.empty(edges.shape[0], dtype=np.int64)
    success = np.ones(n_sections, dtype=bool)

    for s in range(n_sections):
        lo, hi = int(pos[s]), int(pos[s + 1])
        if hi == lo:
            continue
        chain, signs, complete = _walk(edges[lo:hi])
        order[lo:hi] = np.asarray(chain, dtype=np.int64) + lo
        sign[lo:hi] = signs
        success[s] = complete
    return order, sign, success


def _walk(section: np.ndarray):
    """Chain one section, starting from its first edge and growing both ways.

    The two directions are grown **interleaved**, scanning the section's edges
    in order and offering each to the forward end first and the backward end
    second. That is MRST's loop, and the interleaving is not cosmetic: on a
    closed cycle, growing one direction to exhaustion consumes the whole loop
    and leaves the seed edge at the start, while the interleaved walk meets in
    the middle and leaves it partway along. Both are valid cycles, and only one
    matches MATLAB.
    """
    count = section.shape[0]
    used = np.zeros(count, dtype=bool)
    used[0] = True
    forward: list[tuple[int, int]] = []
    backward: list[tuple[int, int]] = []
    next_node = int(section[0, 1])
    prev_node = int(section[0, 0])

    progressed = True
    while progressed:
        progressed = False
        for e in range(1, count):
            if not used[e]:
                if section[e, 0] == next_node or section[e, 1] == next_node:
                    at_start = section[e, 0] == next_node
                    used[e] = True
                    forward.append((e, 1 if at_start else -1))
                    next_node = int(section[e, 1] if at_start else section[e, 0])
                    progressed = True
                    continue
            if not used[e]:
                if section[e, 0] == prev_node or section[e, 1] == prev_node:
                    at_end = section[e, 1] == prev_node
                    used[e] = True
                    backward.append((e, 1 if at_end else -1))
                    prev_node = int(section[e, 0] if at_end else section[e, 1])
                    progressed = True

    steps = list(reversed(backward)) + [(0, 1)] + forward
    complete = bool(np.all(used))
    if not complete:
        # Keep the permutation total: an unreachable edge keeps its place at the
        # end rather than vanishing from the result.
        steps += [(int(e), 1) for e in np.flatnonzero(~used)]
    return [e for e, _ in steps], [s for _, s in steps], complete


__all__ = ["sort_edges"]


if __name__ == "__main__":
    def _traversed(edges, order, sign):
        """The node sequence a caller reads off the result."""
        out = []
        for e, s in zip(order, sign):
            a, b = edges[e] if s > 0 else edges[e][::-1]
            if not out:
                out.append(int(a))
            out.append(int(b))
        return out

    def test_a_scrambled_chain_is_put_back_in_order():
        """The defining property: consecutive edges share a node."""
        edges = np.array([[2, 3], [0, 1], [3, 4], [1, 2]])
        order, sign, success = sort_edges(edges)
        assert success[0]
        nodes = _traversed(edges, order, sign)
        assert nodes == [0, 1, 2, 3, 4] or nodes == [4, 3, 2, 1, 0], nodes
        print(f"ok test_a_scrambled_chain_is_put_back_in_order ({nodes})")

    def test_the_signs_make_the_walk_continuous():
        """Without the sign the order alone says nothing about direction.

        An edge stored as (3, 2) inside a chain running 2 to 3 has to be read
        backwards, and only the sign records that.
        """
        edges = np.array([[0, 1], [2, 1], [2, 3]])
        order, sign, _ = sort_edges(edges)
        nodes = _traversed(edges, order, sign)
        for a, b in zip(nodes, nodes[1:]):
            assert a != b
        assert sorted(nodes) == [0, 1, 2, 3], nodes
        assert -1 in sign.tolist(), "the reversed edge was not marked"
        print(f"ok test_the_signs_make_the_walk_continuous (signs {sign.tolist()})")

    def test_the_first_edge_is_always_traversed_forwards():
        """MRST's convention, and what fixes the chain's overall direction."""
        edges = np.array([[2, 3], [0, 1], [3, 4], [1, 2]])
        order, sign, _ = sort_edges(edges)
        first = int(np.flatnonzero(order == 0)[0])
        assert sign[first] == 1, sign.tolist()
        print("ok test_the_first_edge_is_always_traversed_forwards")

    def test_the_result_is_a_permutation_within_each_section():
        edges = np.array([[2, 3], [0, 1], [3, 4], [1, 2],
                          [10, 11], [11, 12]])
        pos = np.array([0, 4, 6])
        order, _, _ = sort_edges(edges, pos)
        assert np.array_equal(np.sort(order[:4]), np.arange(4))
        assert np.array_equal(np.sort(order[4:]), np.arange(4, 6))
        print("ok test_the_result_is_a_permutation_within_each_section")

    def test_sections_do_not_leak_into_each_other():
        """Two chains sharing a node must stay separate.

        Nothing in the output shape reveals a leak; only the membership does.
        """
        edges = np.array([[0, 1], [1, 2], [1, 5], [5, 6]])
        pos = np.array([0, 2, 4])
        order, _, success = sort_edges(edges, pos)
        assert np.all(success)
        assert set(order[:2].tolist()) == {0, 1}
        assert set(order[2:].tolist()) == {2, 3}
        print("ok test_sections_do_not_leak_into_each_other")

    def test_a_disconnected_section_is_reported_not_raised():
        """A coarse face cut by a fault is a real case, not an error."""
        edges = np.array([[0, 1], [5, 6]])
        order, _, success = sort_edges(edges)
        assert not success[0], "a disconnected section claimed success"
        assert np.array_equal(np.sort(order), np.arange(2)), order
        print("ok test_a_disconnected_section_is_reported_not_raised")

    def test_a_failed_section_still_returns_a_full_permutation():
        """Where MRST drops entries, which silently corrupts the next section.

        MATLAB's `map = map(map > 0)` shortens a failed section, and its caller
        reindexes the packed array anyway, so every later coarse face reads the
        wrong fine faces. Only the warning says anything.
        """
        edges = np.array([[0, 1], [5, 6],        # section 0: two pieces
                          [10, 11], [11, 12]])   # section 1: a chain
        pos = np.array([0, 2, 4])
        order, sign, success = sort_edges(edges, pos)
        assert success.tolist() == [False, True], success
        assert order.size == edges.shape[0], order.size
        assert set(order[:2].tolist()) == {0, 1}
        assert set(order[2:].tolist()) == {2, 3}, "the failure shifted section 1"
        print("ok test_a_failed_section_still_returns_a_full_permutation")

    def test_a_closed_loop_is_handled():
        """A block's own boundary is a cycle, with no endpoint to start from."""
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        order, sign, success = sort_edges(edges)
        assert success[0]
        nodes = _traversed(edges, order, sign)
        assert nodes[0] == nodes[-1], nodes
        assert len(set(nodes[:-1])) == 4, nodes
        print(f"ok test_a_closed_loop_is_handled ({nodes})")

    def test_a_single_edge_is_left_alone():
        edges = np.array([[7, 9]])
        order, sign, success = sort_edges(edges)
        assert order.tolist() == [0] and sign.tolist() == [1] and success[0]
        print("ok test_a_single_edge_is_left_alone")

    def test_an_empty_section_is_allowed():
        edges = np.array([[0, 1], [1, 2]])
        order, _, success = sort_edges(edges, np.array([0, 0, 2]))
        assert success.size == 2 and success[0] and success[1]
        assert order.size == 2
        print("ok test_an_empty_section_is_allowed")

    def test_the_input_is_not_mutated():
        """MRST wipes edges as it consumes them; this must not."""
        edges = np.array([[2, 3], [0, 1], [3, 4], [1, 2]])
        before = edges.copy()
        sort_edges(edges)
        assert np.array_equal(edges, before), "the edge array was modified"
        print("ok test_the_input_is_not_mutated")

    def test_bad_input_is_rejected():
        try:
            sort_edges(np.zeros((3, 3), dtype=np.int64))
        except ValueError as exc:
            assert "n-by-2" in str(exc), exc
        else:
            raise AssertionError("expected a three-column array to fail")
        try:
            sort_edges(np.array([[0, 1], [1, 2]]), np.array([0, 1]))
        except ValueError as exc:
            assert "pos must run" in str(exc), exc
        else:
            raise AssertionError("expected a short pos to fail")
        print("ok test_bad_input_is_rejected")

    test_a_scrambled_chain_is_put_back_in_order()
    test_the_signs_make_the_walk_continuous()
    test_the_first_edge_is_always_traversed_forwards()
    test_the_result_is_a_permutation_within_each_section()
    test_sections_do_not_leak_into_each_other()
    test_a_disconnected_section_is_reported_not_raised()
    test_a_failed_section_still_returns_a_full_permutation()
    test_a_closed_loop_is_handled()
    test_a_single_edge_is_left_alone()
    test_an_empty_section_is_allowed()
    test_the_input_is_not_mutated()
    test_bad_input_is_rejected()
    print("\nAll tests passed!")
