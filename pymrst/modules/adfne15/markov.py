"""Native iterative port of ADFNE15 ``Markov.m``."""

import numpy as np


def markov(probabilities, current_state, count, *, rng=None):
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    state = int(np.asarray(current_state).reshape(-1)[-1])
    output = np.empty(int(count), dtype=int)
    for index in range(int(count)):
        cumulative = np.r_[0, np.cumsum(probabilities[state-1]), 1]
        state = int(np.searchsorted(cumulative, rng.random(), side="right"))
        output[index] = state
    return output
