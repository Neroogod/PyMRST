"""Native port of ADFNE15 ``Cluster.m``."""

import numpy as np


def cluster(groups):
    pending = [set(np.asarray(group).reshape(-1).tolist()) for group in groups]
    output = []
    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            remainder = []
            for candidate in pending:
                if current & candidate:
                    current |= candidate
                    changed = True
                else:
                    remainder.append(candidate)
            pending = remainder
        output.append(np.asarray(sorted(current), dtype=np.int64))
    return output
