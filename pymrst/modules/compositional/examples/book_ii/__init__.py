"""MRST `modules/compositional/examples/book-ii` example ports."""

from pymrst.modules.compositional.examples.book_ii.compositional_accuracy_example import (
    accuracy_component_profiles,
    accuracy_cumulative_iterations,
    accuracy_gas_saturation_profiles,
    accuracy_phase_saturations,
    accuracy_plot_sections,
    accuracy_property_profiles,
    run_compositional_accuracy_example,
)
from pymrst.modules.compositional.examples.book_ii.compositional_miscibility_example import (
    miscibility_formulation_iterations,
    miscibility_front_animation,
    miscibility_plot_sections,
    run_compositional_miscibility_example,
)
from pymrst.modules.compositional.examples.book_ii.compositional_performance_example import (
    mrst_performance_plot_table,
    run_compositional_performance_example,
)
from pymrst.modules.compositional.examples.book_ii.compositional_separator_example import (
    run_compositional_separator_example,
)
from pymrst.modules.compositional.examples.book_ii.intro_to_vapor_liquid_equilibrium import (
    run_intro_to_vapor_liquid_equilibrium,
    spe5_flash_diagnostics_section,
)
from pymrst.modules.compositional.examples.book_ii.show_natural_overall import (
    run_show_natural_overall,
    show_natural_overall_convergence_plots,
)
from pymrst.modules.compositional.examples.book_ii.show_natural_overall2 import (
    run_show_natural_overall2,
    show_natural_overall2_runtime_plot,
)

__all__ = [
    "accuracy_component_profiles",
    "accuracy_cumulative_iterations",
    "accuracy_gas_saturation_profiles",
    "accuracy_phase_saturations",
    "accuracy_plot_sections",
    "accuracy_property_profiles",
    "miscibility_formulation_iterations",
    "miscibility_front_animation",
    "miscibility_plot_sections",
    "mrst_performance_plot_table",
    "run_compositional_accuracy_example",
    "run_compositional_miscibility_example",
    "run_compositional_performance_example",
    "run_compositional_separator_example",
    "run_intro_to_vapor_liquid_equilibrium",
    "run_show_natural_overall",
    "run_show_natural_overall2",
    "show_natural_overall_convergence_plots",
    "show_natural_overall2_runtime_plot",
    "spe5_flash_diagnostics_section",
]
