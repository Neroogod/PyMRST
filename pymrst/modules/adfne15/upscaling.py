"""Native port of ADFNE15 ``Upscaling.m``."""

import numpy as np


def upscaling(values, method="harmonic", level=1):
    output = np.asarray(values, dtype=float)
    if output.ndim != 2 or any(value < 1 or value & (value-1) for value in output.shape):
        raise ValueError("both matrix dimensions must be powers of two")
    name = str(method).lower()
    while output.shape[0] > level and output.shape[1] > level:
        rows, columns = output.shape
        coarse = np.empty((rows//2, columns//2))
        for i in range(0, rows, 2):
            for j in range(0, columns, 2):
                block = output[i:i+2, j:j+2].reshape(-1)
                if name in {"harmonic", "h"}:
                    coarse[i//2, j//2] = len(block)/np.sum(1/block)
                elif name in {"geometric", "g"}:
                    coarse[i//2, j//2] = np.exp(np.mean(np.log(block)))
                else:
                    coarse[i//2, j//2] = np.mean(block)
        output = coarse
    return output
