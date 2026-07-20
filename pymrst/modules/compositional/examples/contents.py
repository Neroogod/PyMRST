"""Catalog for the MRST compositional example ports."""

MRST_CONTENTS_EXAMPLES = (
    (
        "compositionalBoundaryConditionsExample",
        "Example comparing natural variables and overall composition with boundary conditions",
    ),
    (
        "compositionalExample3DSixComponents",
        "Example demonstrating a three dimensional, six component problem",
    ),
    (
        "compositionalLargerProblemTutorial",
        "Conceptual example on how to run larger compositional problems",
    ),
    (
        "compositionalValidationSimple",
        "Compare MRST results with other simulators",
    ),
    (
        "runCompositionalSPE1",
        "Example demonstrating compositional solvers with K-values",
    ),
)

EXAMPLES = {
    "compositionalBoundaryConditionsExample": (
        "Runnable natural/overall CO2 boundary-condition displacement with MRST mixture and schedule setup."
    ),
    "compositionalExample3DSixComponents": (
        "Runnable lumped_1 six-component 3D quarter five-spot with MRST schedule structure."
    ),
    "compositionalLargerProblemTutorial": (
        "Runnable 3D SPE5 larger-problem backend tutorial with MRST switches, timing table, and full-size option."
    ),
    "compositionalValidationSimple": (
        "Runnable packed 1D injector-producer validation port with MRST/E300/AD-GPRS comparison slots."
    ),
    "runCompositionalSPE1": (
        "Runnable SPE1 natural/overall/reference comparison with pseudo-component rates and iteration plot data."
    ),
    "book-ii/introToVaporLiquidEquilibrium": (
        "Runnable flash and phase-split example."
    ),
    "book-ii/compositionalAccuracyExample": (
        "Runnable Explicit/Natural/Overall accuracy comparison with MRST plotting-data sections."
    ),
    "book-ii/compositionalMiscibilityExample": (
        "Runnable MRST CO2/C1/C5/C10 miscibility setup with pressure-front and formulation iteration sections."
    ),
    "book-ii/compositionalPerformanceExample": (
        "Runnable small-grid performance benchmark."
    ),
    "book-ii/compositionalSeparatorExample": (
        "Runnable EOS separator flash example."
    ),
    "book-ii/showNaturalOverall": (
        "Runnable natural/overall formulation comparison with MRST-style flux/pressure BCs."
    ),
    "book-ii/showNaturalOverall2": (
        "Runnable benchmark-mixture natural/overall comparison with MRST-style bad initial guess."
    ),
}


EXAMPLE_PORTS = {
    "Contents.m": {
        "python": "pymrst.modules.compositional.examples.contents",
        "file": "contents.py",
        "status": "catalog",
    },
    "compositionalBoundaryConditionsExample.m": {
        "python": "pymrst.modules.compositional.examples.compositional_boundary_conditions_example",
        "file": "compositional_boundary_conditions_example.py",
        "status": "runnable",
    },
    "compositionalExample3DSixComponents.m": {
        "python": "pymrst.modules.compositional.examples.compositional_example_3d_six_components",
        "file": "compositional_example_3d_six_components.py",
        "status": "runnable",
    },
    "compositionalLargerProblemTutorial.m": {
        "python": "pymrst.modules.compositional.examples.compositional_larger_problem_tutorial",
        "file": "compositional_larger_problem_tutorial.py",
        "status": "runnable",
    },
    "compositionalValidationSimple.m": {
        "python": "pymrst.modules.compositional.examples.compositional_validation_simple",
        "file": "compositional_validation_simple.py",
        "status": "runnable",
    },
    "runCompositionalSPE1.m": {
        "python": "pymrst.modules.compositional.examples.run_compositional_spe1",
        "file": "run_compositional_spe1.py",
        "status": "runnable",
    },
    "book-ii/compositionalAccuracyExample.m": {
        "python": "pymrst.modules.compositional.examples.book_ii.compositional_accuracy_example",
        "file": "book_ii/compositional_accuracy_example.py",
        "status": "runnable",
    },
    "book-ii/compositionalMiscibilityExample.m": {
        "python": "pymrst.modules.compositional.examples.book_ii.compositional_miscibility_example",
        "file": "book_ii/compositional_miscibility_example.py",
        "status": "runnable",
    },
    "book-ii/compositionalPerformanceExample.m": {
        "python": "pymrst.modules.compositional.examples.book_ii.compositional_performance_example",
        "file": "book_ii/compositional_performance_example.py",
        "status": "runnable",
    },
    "book-ii/compositionalSeparatorExample.m": {
        "python": "pymrst.modules.compositional.examples.book_ii.compositional_separator_example",
        "file": "book_ii/compositional_separator_example.py",
        "status": "runnable",
    },
    "book-ii/introToVaporLiquidEquilibrium.m": {
        "python": "pymrst.modules.compositional.examples.book_ii.intro_to_vapor_liquid_equilibrium",
        "file": "book_ii/intro_to_vapor_liquid_equilibrium.py",
        "status": "runnable",
    },
    "book-ii/showNaturalOverall.m": {
        "python": "pymrst.modules.compositional.examples.book_ii.show_natural_overall",
        "file": "book_ii/show_natural_overall.py",
        "status": "runnable",
    },
    "book-ii/showNaturalOverall2.m": {
        "python": "pymrst.modules.compositional.examples.book_ii.show_natural_overall2",
        "file": "book_ii/show_natural_overall2.py",
        "status": "runnable",
    },
    "utils/setupSimpleCompositionalExample.m": {
        "python": "pymrst.modules.compositional.examples.utils.setup_simple_compositional_example",
        "file": "utils/setup_simple_compositional_example.py",
        "status": "utility",
    },
}


def list_examples() -> dict[str, str]:
    """Return the MRST-name to PyMRST-status example catalog."""
    return dict(EXAMPLES)


def list_mrst_contents_examples() -> tuple[tuple[str, str], ...]:
    """Return the exact example entries listed by MRST's Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_EXAMPLES)


def list_example_ports() -> dict[str, dict[str, str]]:
    """Return MRST relative paths mapped to PyMRST port metadata."""
    return {k: dict(v) for k, v in EXAMPLE_PORTS.items()}


if __name__ == "__main__":
    def test_catalog_contains_mrst_examples():
        assert [name for name, _ in MRST_CONTENTS_EXAMPLES] == [
            "compositionalBoundaryConditionsExample",
            "compositionalExample3DSixComponents",
            "compositionalLargerProblemTutorial",
            "compositionalValidationSimple",
            "runCompositionalSPE1",
        ]
        assert "compositionalValidationSimple" in EXAMPLES
        assert "book-ii/introToVaporLiquidEquilibrium" in EXAMPLES
        assert "book-ii/showNaturalOverall" in EXAMPLES
        assert "runCompositionalSPE1" in EXAMPLES
        assert "compositionalValidationSimple.m" in EXAMPLE_PORTS
        assert EXAMPLE_PORTS["book-ii/showNaturalOverall.m"]["status"] == "runnable"
        print("ok test_catalog_contains_mrst_examples")

    test_catalog_contains_mrst_examples()
    print("\nAll tests passed!")


__all__ = [
    "EXAMPLES",
    "EXAMPLE_PORTS",
    "MRST_CONTENTS_EXAMPLES",
    "list_example_ports",
    "list_examples",
    "list_mrst_contents_examples",
]
