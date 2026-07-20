"""MRST ``runAll2DCases.m`` compatibility surface."""

from pymrst.modules.ad_mechanics.examples.run_2d_case import (
    TwoDVEMNotPorted,
    run_2d_case,
)


def run_all_2d_cases():
    """Run the implemented MRST-shaped 2D cases.

    The water, oil-water, and black-oil fully coupled/fixed-stress branches
    match MRST's ``run2DCase``.
    """
    cases = [
        ("water", "fully coupled"),
        ("water", "fixed stress splitting"),
        ("oil water", "fully coupled"),
        ("oil water", "fixed stress splitting"),
        ("blackoil", "fully coupled"),
        ("blackoil", "fixed stress splitting"),
    ]
    results = []
    for fluid_model, method in cases:
        try:
            results.append(run_2d_case(fluid_model=fluid_model, method=method))
        except TwoDVEMNotPorted as exc:
            results.append({
                "fluid_model": fluid_model,
                "method": method,
                "skipped": True,
                "reason": str(exc),
            })
    return results


__all__ = ["run_all_2d_cases", "TwoDVEMNotPorted"]


if __name__ == "__main__":
    out = run_all_2d_cases()
    done = sum(not isinstance(r, dict) for r in out)
    skipped = len(out) - done
    print(f"ok run_all_2d_cases: {done} ran, {skipped} skipped")
