"""MRST ``runAllNorneExamples.m`` compatibility wrapper."""

from pymrst.modules.ad_mechanics.examples.run_norne_example import (
    run_norne_example,
)


def run_all_norne_examples(include_full: bool = False):
    """Run the supported Norne geomechanics examples."""
    results = [run_norne_example("mini Norne")]
    if include_full:
        results.append(run_norne_example("full"))
    return results


__all__ = ["run_all_norne_examples"]
