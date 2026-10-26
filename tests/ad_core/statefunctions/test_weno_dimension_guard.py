"""WENO refuses a grid it cannot reconstruct, even under `python -O`.

One, two and three dimensions are ported: 1-D walks a directional stencil, 2-D
and 3-D build a local triangulation. Four has no construction behind it - the
stencil is built by tetrahedralising a point cloud - and MRST raises
'Dimension too high' at exactly the same place.

The guard is a `raise` rather than an `assert`, and the `-O` test is the point
of this file. Assertions are stripped under optimisation, and this one guards
*numerical correctness*: with the check gone an unsupported dimension falls
through to a scheme that reconstructs from coordinates it should not be using,
producing a field that looks like a solution and is not. A guard that only
holds when nobody optimises is not a guard.

This file used to assert the opposite - that anything above 1-D was refused -
because the triangulated path was unported. It is kept pointed at the *current*
boundary rather than deleted, since the `-O` property is what matters and it is
independent of where the boundary sits.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from pymrst.modules.ad_core.statefunctions.flux.upwind.weno import (
    WENOUpwindDiscretizationAD,
)

PROBE = (
    "from pymrst.modules.ad_core.statefunctions.flux.upwind.weno import "
    "WENOUpwindDiscretizationAD as W\n"
    "try:\n"
    "    W(None, {dim})\n"
    "    print('ACCEPTED')\n"
    "except NotImplementedError:\n"
    "    print('REFUSED')\n"
)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_the_supported_dimensions_are_accepted(dim):
    assert WENOUpwindDiscretizationAD(None, dim).dim == dim


@pytest.mark.parametrize("dim", [0, 4, 5])
def test_an_unsupported_dimension_is_refused(dim):
    with pytest.raises(NotImplementedError, match="one, two or three"):
        WENOUpwindDiscretizationAD(None, dim)


def test_the_message_names_what_is_missing():
    """A bare refusal sends the reader to the source to find out why."""
    with pytest.raises(NotImplementedError) as raised:
        WENOUpwindDiscretizationAD(None, 4)
    message = str(raised.value)
    assert "dim=4" in message
    assert "tetrahedralis" in message.lower(), "say what the construction is"


@pytest.mark.parametrize("flags,label", [([], "normal"), (["-O"], "optimised")])
def test_the_guard_survives_optimisation(flags, label):
    """The reason this is a `raise` and not an `assert`.

    Under `-O` an earlier assertion vanished and an unsupported dimension was
    accepted, after which the reconstruction silently used the wrong
    coordinates.
    """
    result = subprocess.run([sys.executable, *flags, "-c", PROBE.format(dim=4)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "REFUSED", (
        f"WENO accepted a 4-D grid when run {label}")
