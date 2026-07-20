"""Phase 251: MRST-style SeparatorGroup property/method aliases."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities.separators import EOSSeparator, SeparatorGroup  # noqa: E402
from pymrst.modules.compositional.fluids import get_benchmark_mixture  # noqa: E402
from pymrst.utils.units import ATM, BARSA  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    mixture, _ = get_benchmark_mixture("simple")
    separator = EOSSeparator(pressure=ATM, T=300.0)
    phase_destination = np.array([[2, 0], [0, -1]])
    group = SeparatorGroup(
        separator,
        pressure=np.array([20.0, 1.0]) * BARSA,
        T=np.array([340.0, 300.0]),
        phaseDestination=phase_destination,
    )
    pythonic = SeparatorGroup(
        separator,
        pressure=np.array([20.0, 1.0]) * BARSA,
        T=np.array([340.0, 300.0]),
        phase_destination=phase_destination,
    )

    stream = np.array([1.0, 0.25, 0.1])
    camel_rates = group.computeSurfaceRates(mixture, stream)
    snake_rates = group.compute_surface_rates(mixture, stream)
    camel_streams, camel_diagnostics = group.computeSurfaceMoleRates(
        mixture,
        stream,
        return_separator_streams=True,
    )

    print("\nPhase 251: MRST-style SeparatorGroup aliases")
    failures = 0
    failures += _check("camelCase constructor alias normalizes phase destinations",
                       np.array_equal(group.phaseDestination, pythonic.phase_destination)
                       and np.array_equal(group.topologicalOrder, pythonic.topological_order))
    failures += _check("camelCase surfaceSeparator aliases Python storage",
                       group.surfaceSeparator is group.surface_separator)
    failures += _check("getTopologicalSort matches stored order",
                       np.array_equal(group.getTopologicalSort(), group.topological_order))
    failures += _check("computeSurfaceRates alias matches Python method",
                       all(
                           np.allclose(a, b)
                           for a, b in zip(camel_rates.surface_rates, snake_rates.surface_rates)
                       ))
    failures += _check("computeSurfaceMoleRates alias returns MRST diagnostics",
                       len(camel_streams) == 2
                       and len(camel_diagnostics.molestreams) == len(group.separators)
                       and len(camel_diagnostics.densities) == len(group.separators)
                       and len(camel_diagnostics.molefractions) == len(group.separators))

    new_dest = np.array([[0, 0]])
    group.phaseDestination = new_dest
    failures += _check("phaseDestination setter updates snake storage and order",
                       np.array_equal(group.phase_destination, np.array([[0, 0]]))
                       and np.array_equal(group.topological_order, np.array([0])))
    group.topologicalOrder = np.array([0])
    failures += _check("topologicalOrder setter updates snake storage",
                       np.array_equal(group.topological_order, np.array([0])))
    group.surfaceSeparator = separator
    failures += _check("surfaceSeparator setter updates snake storage",
                       group.surface_separator is separator)

    if failures:
        print(f"\nPhase 251 failed ({failures} checks).")
        return 1
    print("\nSeparatorGroup now exposes MRST-style property and method aliases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
