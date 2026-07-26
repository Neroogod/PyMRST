"""Adaptive step control from MRST ``re-mpfa/functions/timeStepping.m``."""

import numpy as np


def time_stepping(
    dt, dt_min, dt_max, sim_time, time_cum, iterations,
    print_times, print_counter, lower_opt_iter_range,
    upper_opt_iter_range, lower_mult_factor, upper_mult_factor,
):
    """Choose the next step from iteration count and output-time constraints."""
    step = float(dt)
    completed_iterations = int(iterations) - 1
    if completed_iterations <= lower_opt_iter_range:
        step *= lower_mult_factor
    elif completed_iterations >= upper_opt_iter_range:
        step *= upper_mult_factor
    step = min(max(step, float(dt_min)), float(dt_max))

    targets = np.asarray(print_times, dtype=float).ravel()
    counter = int(print_counter)
    # Accept Python's zero-based counter and MRST's one-based counter.
    index = counter if counter == 0 else counter - 1
    remaining = float(sim_time) - float(time_cum)
    step = min(step, remaining)
    if 0 <= index < targets.size:
        to_print = targets[index] - float(time_cum)
        if to_print >= 0 and step > to_print:
            step = to_print
            counter += 1
    return max(step, 0.0), counter


timeStepping = time_stepping

__all__ = ["time_stepping", "timeStepping"]
