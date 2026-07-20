"""Phase 186: MRST-style separator intermediate stream diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core import SeparatorStreamResult  # noqa: E402
from pymrst.modules.compositional.examples.book_ii.compositional_separator_example import (  # noqa: E402
    make_separator_tree,
)
from pymrst.modules.compositional.fluids import get_compositional_fluid_case  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _component_outputs(molestreams, molefractions) -> np.ndarray:
    out = np.zeros((np.asarray(molefractions[0]).shape[0], np.asarray(molefractions[0]).shape[1]))
    for ms, frac in zip(molestreams, molefractions):
        out += np.asarray(frac, dtype=float) * np.asarray(ms, dtype=float).reshape(-1, 1)
    return out


def _phase_component_rows(streams) -> np.ndarray:
    return np.vstack([np.asarray(v, dtype=float).reshape(1, -1) for v in streams])


def validate_direct_separator_diagnostics() -> int:
    mixture, info = get_compositional_fluid_case("spe5")
    stream = np.asarray(info.initial, dtype=float)
    _, _, tree = make_separator_tree()
    surface_default = tree.compute_surface_mole_rates(mixture, stream)
    surface_diag, diagnostics = tree.compute_surface_mole_rates(
        mixture,
        stream,
        return_separator_streams=True,
    )
    first_outputs = _component_outputs(diagnostics.molestreams[0], diagnostics.molefractions[0])

    failures = 0
    failures += _check("diagnostic compute_surface_mole_rates returns surface streams unchanged",
                       np.allclose(_phase_component_rows(surface_default), _phase_component_rows(surface_diag)))
    failures += _check("diagnostics use MRST-style SeparatorStreamResult",
                       isinstance(diagnostics, SeparatorStreamResult))
    failures += _check("diagnostics carry one entry per intermediate separator",
                       len(diagnostics.molestreams) == len(tree.separators)
                       and len(diagnostics.densities) == len(tree.separators)
                       and len(diagnostics.molefractions) == len(tree.separators))
    failures += _check("first separator component split conserves incoming mole stream",
                       np.allclose(first_outputs.reshape(-1), stream, atol=1.0e-12))
    failures += _check("active separator diagnostics contain phase streams/fractions/densities",
                       all(
                           diagnostics.molestreams[i] is not None
                           and diagnostics.molefractions[i] is not None
                           and diagnostics.densities[i] is not None
                           for i in range(len(tree.separators))
                       ))
    return failures


def validate_compute_surface_rates_carries_diagnostics() -> int:
    mixture, info = get_compositional_fluid_case("spe5")
    stream = np.asarray(info.initial, dtype=float)
    _, _, tree = make_separator_tree()
    direct_surface, direct_diag = tree.compute_surface_mole_rates(
        mixture,
        stream,
        return_separator_streams=True,
    )
    rates = tree.compute_surface_rates(mixture, stream)

    failures = 0
    failures += _check("compute_surface_rates exposes separator diagnostics",
                       isinstance(rates.separator_streams, SeparatorStreamResult))
    failures += _check("compute_surface_rates diagnostics match direct diagnostic call",
                       np.allclose(
                           _phase_component_rows(rates.surface_mole_streams),
                           _phase_component_rows(direct_surface),
                       )
                       and np.allclose(
                           np.asarray(rates.separator_streams.molestreams[0][0]),
                           np.asarray(direct_diag.molestreams[0][0]),
                       ))
    failures += _check("single surface separator has no intermediate diagnostics",
                       make_separator_tree()[1].compute_surface_rates(mixture, stream).separator_streams is None)
    return failures


def validate_downstream_start_keeps_upstream_inactive() -> int:
    mixture, info = get_compositional_fluid_case("spe5")
    stream = np.asarray(info.initial, dtype=float)
    _, _, tree = make_separator_tree()
    _, diagnostics = tree.compute_surface_mole_rates(
        mixture,
        stream,
        start=2,
        return_separator_streams=True,
    )

    failures = 0
    failures += _check("downstream start leaves upstream separator diagnostics inactive",
                       diagnostics.molestreams[0] is None
                       and diagnostics.densities[0] is None
                       and diagnostics.molefractions[0] is None)
    failures += _check("downstream start activates selected separator",
                       diagnostics.molestreams[1] is not None
                       and diagnostics.densities[1] is not None
                       and diagnostics.molefractions[1] is not None)
    return failures


def main() -> int:
    print("\nPhase 186: MRST-style separator intermediate stream diagnostics")
    failures = validate_direct_separator_diagnostics()
    failures += validate_compute_surface_rates_carries_diagnostics()
    failures += validate_downstream_start_keeps_upstream_inactive()
    if failures:
        print(f"\nPhase 186 failed ({failures} checks).")
        return 1
    print("\nSeparator trees now expose MRST-style intermediate stream diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
