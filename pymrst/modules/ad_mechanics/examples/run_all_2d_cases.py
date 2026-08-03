"""MRST ``runAll2DCases.m`` compatibility surface."""

from pymrst.modules.ad_mechanics.examples.run_2d_case import run_2d_case


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
    return [run_2d_case(fluid_model=fluid_model, method=method)
            for fluid_model, method in cases]


__all__ = ["run_all_2d_cases"]


if __name__ == "__main__":
    out = run_all_2d_cases()
    print(f"ok run_all_2d_cases: {len(out)} ran")
