"""MRST ``ad-core/models/aquifers/selectAquiferFromDeck.m``.

`select_aquifer_from_deck` is the port, for the AD-OO models: no aquifer
unless the deck connects one, and only Fetkovich aquifers are supported.
`select_aquifer_from_deck_ad` is PyMRST's earlier version, returning the
legacy `AquiferModel`, kept for its callers.
"""

from __future__ import annotations

import warnings

from typing import Mapping, Optional

import numpy as np

from pymrst.modules.ad_core.models.aquifers.aquifer import AquiferConnections, AquiferModelAD
from pymrst.modules.deckformat.params.rock.process_aquifer import (
    AquiferDeckOutput,
    process_aquifer,
)


def aquifer_model_from_processed(output: AquiferDeckOutput) -> AquiferModelAD:
    """Build ``AquiferModel`` from ``process_aquifer`` output."""
    a, ind = output.aquifers, output.aquind
    if a.size == 0:
        conn = AquiferConnections(
            aquid=np.zeros(0, dtype=int),
            cells=np.zeros(0, dtype=int),
            j=np.zeros(0),
            c=np.zeros(0),
            alpha=np.zeros(0),
            depth_conn=np.zeros(0),
            depth_aq=np.zeros(0),
            c_aq=np.asarray(output.aquiferprops.C, dtype=float),
        )
    else:
        conn = AquiferConnections(
            aquid=a[:, ind.aquid].astype(int),
            cells=a[:, ind.conn].astype(int),
            j=a[:, ind.J],
            c=a[:, ind.C],
            alpha=a[:, ind.alpha],
            depth_conn=a[:, ind.depthconn],
            depth_aq=a[:, ind.depthaq],
            c_aq=np.asarray(output.aquiferprops.C, dtype=float),
        )
    return AquiferModelAD(
        conn,
        np.asarray(output.initval["pressures"], dtype=float),
        np.asarray(output.initval["volumes"], dtype=float),
    )


def select_aquifer_from_deck_ad(deck: Mapping, model) -> Optional[AquiferModelAD]:
    """Return a Fetkovich aquifer model if the deck contains AQUANCON/AQUFETP."""
    sol = deck.get("SOLUTION", deck)
    if "AQUANCON" not in sol:
        return None
    if "AQUFETP" not in sol:
        return None
    return aquifer_model_from_processed(process_aquifer(deck, model.G))


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.
def select_aquifer_from_deck(deck: Mapping, model):
    """MRST `selectAquiferFromDeck`: a Fetkovich `AquiferModel`, or `None`."""
    aquifers = None
    solution = deck.get("SOLUTION") if isinstance(deck, Mapping) else None
    if solution is not None and "AQUANCON" in solution:
        if "AQUFETP" not in solution:
            warnings.warn("Aquifers disregarded, only Fetkovich type is "
                          "currently implemented")
        else:
            from pymrst.modules.ad_core.models.aquifers.aquifer_model import (
                AquiferModel as AdOoAquiferModel,
            )
            output = process_aquifer(deck, model.G)
            aquifers = AdOoAquiferModel(model, output.aquifers, output.aquind,
                                        output.aquiferprops, output.initval)
    return aquifers


selectAquiferFromDeck = select_aquifer_from_deck

__all__ = ["aquifer_model_from_processed", "select_aquifer_from_deck",
           "select_aquifer_from_deck_ad"]


if __name__ == "__main__":
    def test_no_connected_aquifer_means_none():
        assert select_aquifer_from_deck({}, None) is None
        assert select_aquifer_from_deck({"SOLUTION": {}}, None) is None
        print("ok test_no_connected_aquifer_means_none")

    def test_a_non_fetkovich_aquifer_is_disregarded_with_a_warning():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert select_aquifer_from_deck({"SOLUTION": {"AQUANCON": [[1]]}}, None) is None
        assert any("Fetkovich" in str(w.message) for w in caught)
        print("ok test_a_non_fetkovich_aquifer_is_disregarded_with_a_warning")

    def test_a_fetkovich_aquifer_becomes_an_aquifer_model():
        from types import SimpleNamespace
        from pymrst.gridprocessing import cart_grid, compute_geometry
        from pymrst.modules.ad_core.models.aquifers.aquifer_model import (
            AquiferModel as AdOoAquiferModel,
        )
        G = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
        deck = {"SOLUTION": {"AQUANCON": [[1, 1, 1, 1, 1, 1, 1, "I-", 1.0, 1.0]],
                             "AQUFETP": np.array([[1, 0.5, 2e7, 1e6, 1e-9, 1e-10, 1]])}}
        model = SimpleNamespace(G=G)
        aquifer = select_aquifer_from_deck(deck, model)
        assert isinstance(aquifer, AdOoAquiferModel) and aquifer.ReservoirModel is model
        assert aquifer.aquifers.shape == (1, 8)
        print("ok test_a_fetkovich_aquifer_becomes_an_aquifer_model")

    test_no_connected_aquifer_means_none()
    test_a_non_fetkovich_aquifer_is_disregarded_with_a_warning()
    test_a_fetkovich_aquifer_becomes_an_aquifer_model()
    print("\nAll tests passed!")
