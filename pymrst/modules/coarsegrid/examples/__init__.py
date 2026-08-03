"""MRST's own coarse-grid examples (``modules/coarsegrid/examples``).

Each file follows the convention used across `pymrst/examples`: the work is done
in functions that return values and can be asserted on, plotting is optional and
returns ``None`` without a backend, and the `__main__` block is the test suite.

MRST's examples are scripts whose output is a figure. Ported literally they
would assert nothing, so each one here states what its figure is meant to show
and checks that instead.
"""
