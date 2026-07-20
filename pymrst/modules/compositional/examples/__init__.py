"""MRST `modules/compositional/examples` layout for PyMRST."""

from pymrst.modules.compositional.examples.contents import EXAMPLES
from pymrst.modules.compositional.examples.contents import list_examples
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (
    boundary_condition_iteration_plot,
    boundary_condition_phase_saturations,
    boundary_condition_plot_fields,
    boundary_condition_plot_sections,
    run_compositional_boundary_conditions_example,
)
from pymrst.modules.compositional.examples.compositional_example_3d_six_components import (
    extract_plot_fields,
    run_compositional_example_3d_six_components,
    setup_3d_six_component_case,
    setup_3d_six_component_state,
    six_component_plot_frames,
    six_component_plot_sections,
    six_component_state_plot_fields,
)
from pymrst.modules.compositional.examples.compositional_larger_problem_tutorial import (
    larger_problem_plot_sections,
    larger_problem_tick_labels,
    larger_problem_time_matrix,
    make_larger_problem_case,
    run_compositional_larger_problem_tutorial,
)
from pymrst.modules.compositional.examples.compositional_validation_simple import (
    compare_natural_overall,
    run_compositional_validation_packed,
    run_compositional_validation_simple,
    setup_compositional_validation_simple_problem,
    ternary_displacement_paths,
    validation_comparison_data,
    validation_plot_sections,
)
from pymrst.modules.compositional.examples.run_compositional_spe1 import (
    run_compositional_spe1,
    run_compositional_spe1_comparison,
    spe1_cumulative_iterations,
    spe1_plot_sections,
    spe1_reference_pseudo_component_rates,
    spe1_well_solution_datasets,
)

__all__ = [
    "EXAMPLES",
    "list_examples",
    "boundary_condition_iteration_plot",
    "boundary_condition_phase_saturations",
    "boundary_condition_plot_fields",
    "boundary_condition_plot_sections",
    "compare_natural_overall",
    "run_compositional_boundary_conditions_example",
    "run_compositional_example_3d_six_components",
    "run_compositional_larger_problem_tutorial",
    "run_compositional_spe1",
    "run_compositional_spe1_comparison",
    "run_compositional_validation_packed",
    "run_compositional_validation_simple",
    "extract_plot_fields",
    "larger_problem_plot_sections",
    "larger_problem_tick_labels",
    "larger_problem_time_matrix",
    "make_larger_problem_case",
    "setup_3d_six_component_case",
    "setup_3d_six_component_state",
    "six_component_plot_frames",
    "six_component_plot_sections",
    "six_component_state_plot_fields",
    "setup_compositional_validation_simple_problem",
    "spe1_cumulative_iterations",
    "spe1_plot_sections",
    "spe1_reference_pseudo_component_rates",
    "spe1_well_solution_datasets",
    "ternary_displacement_paths",
    "validation_comparison_data",
    "validation_plot_sections",
]
