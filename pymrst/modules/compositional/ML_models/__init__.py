"""Optional Victory neural-flash experiments for the compositional module.

Nothing in this package is imported by the production compositional runtime.
The public objects are loaded lazily so PyMRST remains usable without Torch.
"""

__all__ = ["VictoryFlashEnsemble", "VictoryPrediction", "victory_mixture"]


def __getattr__(name):
    if name in {"VictoryFlashEnsemble", "VictoryPrediction", "victory_mixture"}:
        from .victory_flash import VictoryFlashEnsemble, VictoryPrediction, victory_mixture

        return {
            "VictoryFlashEnsemble": VictoryFlashEnsemble,
            "VictoryPrediction": VictoryPrediction,
            "victory_mixture": victory_mixture,
        }[name]
    raise AttributeError(name)
