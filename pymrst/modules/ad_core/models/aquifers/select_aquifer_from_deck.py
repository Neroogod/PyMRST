"""MRST ``ad-core/models/aquifers/selectAquiferFromDeck.m``."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from pymrst.modules.ad_core.aquifer import AquiferConnections, AquiferModel
from pymrst.modules.deckformat.params.rock.process_aquifer import (
    AquiferDeckOutput,
    process_aquifer,
)


def aquifer_model_from_processed(output: AquiferDeckOutput) -> AquiferModel:
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
    return AquiferModel(
        conn,
        np.asarray(output.initval["pressures"], dtype=float),
        np.asarray(output.initval["volumes"], dtype=float),
    )


def select_aquifer_from_deck(deck: Mapping, model) -> Optional[AquiferModel]:
    """Return a Fetkovich aquifer model if the deck contains AQUANCON/AQUFETP."""
    sol = deck.get("SOLUTION", deck)
    if "AQUANCON" not in sol:
        return None
    if "AQUFETP" not in sol:
        return None
    return aquifer_model_from_processed(process_aquifer(deck, model.G))


__all__ = ["aquifer_model_from_processed", "select_aquifer_from_deck"]
