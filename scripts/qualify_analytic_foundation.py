#!/usr/bin/env python3
"""Generate the complete M2.4 analytic/manufactured evidence graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pymrst.research_studio.validation import (
    EvidenceEdge, EvidenceGraph, EvidenceNode, EvidenceNodeKind, Independence,
    foundation_analytic_registry,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/research_studio/qualification/analytic_foundation_evidence_graph.json"


def _edge(edge_id, source, target, claim, accepted, metrics, evidence_class):
    return EvidenceEdge(
        edge_id, source, target, evidence_class, claim,
        Independence.INDEPENDENT, mappings=("cell_centroid:exact", "SI_units:exact"),
        accepted=bool(accepted), converged=True, metrics=metrics,
    )


def qualify():
    results = foundation_analytic_registry().run_all()
    hydro = results["hydrostatic_tpfa_column"]
    diffusion = results["pressure_diffusion_1d"]
    radial = results["radial_pressure_bessel_transient"]
    buckley = results["buckley_leverett_equal_mobility"]
    terzaghi = results["terzaghi_consolidation"]
    mandel = results["mandel_cryer_coupling"]
    elasticity = results["elasticity_linear_patch"]
    flash = results["peng_robinson_flash_identities"]
    balance = results["manufactured_accumulation_flux_balance"]
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode(
        "analytic_hydrostatic", EvidenceNodeKind.ANALYTIC,
        "p(z) = p_top + rho g z; zero Darcy flux",
        metadata=hydro.spec.to_dict(),
    ))
    graph.add_node(EvidenceNode(
        "pymrst_hydrostatic", EvidenceNodeKind.PYMRST,
        "PyMRST incompressible TPFA gravity column",
        metadata={"refinement_levels": len(hydro.levels)},
    ))
    graph.add_node(EvidenceNode(
        "analytic_diffusion", EvidenceNodeKind.ANALYTIC,
        "Fourier-series constant-coefficient 1D pressure diffusion",
        metadata=diffusion.spec.to_dict(),
    ))
    graph.add_node(EvidenceNode(
        "pymrst_diffusion", EvidenceNodeKind.PYMRST,
        "PyMRST cell-centred TPFA backward-Euler diffusion",
        metadata={"refinement_levels": len(diffusion.levels)},
    ))
    graph.add_node(EvidenceNode(
        "analytic_radial", EvidenceNodeKind.ANALYTIC,
        "Bessel J0 eigenmode of the axisymmetric diffusion operator",
        metadata=radial.spec.to_dict(),
    ))
    graph.add_node(EvidenceNode(
        "pymrst_radial", EvidenceNodeKind.PYMRST,
        "PyMRST radial finite-volume backward-Euler transient",
        metadata={"refinement_levels": len(radial.levels)},
    ))
    graph.add_node(EvidenceNode(
        "analytic_buckley_leverett", EvidenceNodeKind.ANALYTIC,
        "Welge shock and rarefaction solution",
        metadata=buckley.spec.to_dict(),
    ))
    graph.add_node(EvidenceNode(
        "pymrst_buckley_leverett", EvidenceNodeKind.PYMRST,
        "PyMRST incompressible TPFA plus explicit transport",
        metadata={"refinement_levels": len(buckley.levels)},
    ))
    graph.add_node(EvidenceNode(
        "analytic_terzaghi", EvidenceNodeKind.ANALYTIC,
        "Terzaghi Fourier-series pressure and settlement solution",
        metadata=terzaghi.spec.to_dict(),
    ))
    graph.add_node(EvidenceNode(
        "pymrst_terzaghi", EvidenceNodeKind.PYMRST,
        "PyMRST Terzaghi poroelastic consolidation workflow",
        metadata={"refinement_levels": len(terzaghi.levels)},
    ))
    graph.add_node(EvidenceNode(
        "analytic_mandel", EvidenceNodeKind.ANALYTIC,
        "Mandel series solution with Mandel--Cryer pressure overshoot",
        metadata=mandel.spec.to_dict(),
    ))
    graph.add_node(EvidenceNode(
        "pymrst_mandel", EvidenceNodeKind.PYMRST,
        "PyMRST fully coupled constant-load Mandel workflow",
        metadata={"refinement_levels": len(mandel.levels)},
    ))
    for prefix, result, analytic_label, pymrst_label in (
        ("elasticity_patch", elasticity, "Affine elasticity and rigid-body identities",
         "PyMRST VEM displacement and stiffness assembly"),
        ("flash_equilibrium", flash, "Thermodynamic phase-equilibrium identities",
         "PyMRST Peng--Robinson stability and two-phase flash"),
        ("accumulation_flux", balance, "Manufactured finite-volume balance and Jacobian",
         "PyMRST ADI accumulation and TPFA flux assembly"),
    ):
        graph.add_node(EvidenceNode(
            f"analytic_{prefix}", EvidenceNodeKind.ANALYTIC, analytic_label,
            metadata=result.spec.to_dict(),
        ))
        graph.add_node(EvidenceNode(
            f"pymrst_{prefix}", EvidenceNodeKind.PYMRST, pymrst_label,
            metadata={"refinement_levels": len(result.levels)},
        ))

    max_pressure = max(level.errors["linf_pressure_Pa"] for level in hydro.levels)
    max_flux = max(level.invariants["maximum_absolute_face_flux_m3_s"]
                   for level in hydro.levels)
    graph.add_edge(_edge(
        "hydrostatic_pressure", "analytic_hydrostatic", "pymrst_hydrostatic",
        "TPFA pressure reproduces the exact hydrostatic profile",
        max_pressure <= hydro.acceptance["pressure_absolute_tolerance_Pa"],
        {"maximum_absolute_error_Pa": max_pressure,
         "absolute_tolerance_Pa": hydro.acceptance["pressure_absolute_tolerance_Pa"]},
        "analytic_solution",
    ))
    graph.add_edge(_edge(
        "hydrostatic_zero_flux", "analytic_hydrostatic", "pymrst_hydrostatic",
        "Hydrostatic equilibrium has zero face flux",
        max_flux <= hydro.acceptance["flux_absolute_tolerance_m3_s"],
        {"maximum_absolute_flux_m3_s": max_flux,
         "absolute_tolerance_m3_s": hydro.acceptance["flux_absolute_tolerance_m3_s"]},
        "physical_invariant",
    ))

    for norm in ("l1", "l2", "linf"):
        order = diffusion.observed_orders[norm]
        graph.add_edge(_edge(
            f"diffusion_{norm}_convergence", "analytic_diffusion", "pymrst_diffusion",
            f"Diffusion {norm.upper()} error converges at the declared order",
            order >= diffusion.spec.minimum_order,
            {"observed_order": order, "minimum_order": diffusion.spec.minimum_order,
             "finest_error": diffusion.levels[-1].errors[norm]},
            "grid_time_refinement",
        ))
    max_balance = max(level.invariants["storage_boundary_balance"]
                      for level in diffusion.levels)
    graph.add_edge(_edge(
        "diffusion_conservation", "analytic_diffusion", "pymrst_diffusion",
        "Accumulated pressure storage equals integrated net boundary flux",
        max_balance <= diffusion.acceptance["maximum_storage_balance"],
        {"maximum_absolute_imbalance": max_balance,
         "absolute_tolerance": diffusion.acceptance["maximum_storage_balance"]},
        "conservation",
    ))
    minimum = min(level.invariants["minimum_pressure"] for level in diffusion.levels)
    maximum = max(level.invariants["maximum_pressure"] for level in diffusion.levels)
    graph.add_edge(_edge(
        "diffusion_bounds", "analytic_diffusion", "pymrst_diffusion",
        "Implicit diffusion respects the zero-to-one maximum principle",
        minimum >= -1e-14 and maximum <= 1 + 1e-14,
        {"minimum_pressure": minimum, "maximum_pressure": maximum},
        "physical_invariant",
    ))

    for norm in ("l1", "l2"):
        order = radial.observed_orders[norm]
        graph.add_edge(_edge(
            f"radial_{norm}_convergence", "analytic_radial", "pymrst_radial",
            f"Radial transient {norm.upper()} error converges at near-second order",
            order >= radial.spec.minimum_order,
            {"observed_order": order, "minimum_order": radial.spec.minimum_order,
             "finest_error": radial.levels[-1].errors[norm]},
            "grid_time_refinement",
        ))
    radial_balance = max(level.invariants["storage_boundary_balance"]
                         for level in radial.levels)
    graph.add_edge(_edge(
        "radial_conservation", "analytic_radial", "pymrst_radial",
        "Radial storage change equals integrated inner and outer boundary flux",
        radial_balance <= radial.acceptance["maximum_storage_balance"],
        {"maximum_absolute_imbalance": radial_balance,
         "absolute_tolerance": radial.acceptance["maximum_storage_balance"]},
        "conservation",
    ))
    graph.add_edge(_edge(
        "radial_profile_accuracy", "analytic_radial", "pymrst_radial",
        "Finest radial pressure profile meets the absolute error gate",
        radial.levels[-1].errors["l2"] <= radial.acceptance["maximum_finest_l2_error"],
        {"finest_l2_error": radial.levels[-1].errors["l2"],
         "absolute_tolerance": radial.acceptance["maximum_finest_l2_error"],
         "finest_linf_error": radial.levels[-1].errors["linf"]},
        "analytic_solution",
    ))

    graph.add_edge(_edge(
        "buckley_l1_convergence", "analytic_buckley_leverett",
        "pymrst_buckley_leverett",
        "Shock-capturing transport has the declared L1 convergence rate",
        buckley.observed_orders["l1"] >= buckley.acceptance["minimum_l1_order"],
        {"observed_order": buckley.observed_orders["l1"],
         "minimum_order": buckley.acceptance["minimum_l1_order"]},
        "grid_refinement",
    ))
    graph.add_edge(_edge(
        "buckley_profile_accuracy", "analytic_buckley_leverett",
        "pymrst_buckley_leverett", "The finest saturation profile meets its L1 gate",
        buckley.levels[-1].errors["l1"] <= buckley.acceptance["maximum_finest_l1_error"],
        {"finest_l1_error": buckley.levels[-1].errors["l1"],
         "absolute_tolerance": buckley.acceptance["maximum_finest_l1_error"]},
        "analytic_solution",
    ))
    graph.add_edge(_edge(
        "buckley_shock_position", "analytic_buckley_leverett",
        "pymrst_buckley_leverett", "The numerical water front tracks the Welge shock",
        buckley.levels[-1].errors["shock_position"]
        <= buckley.acceptance["maximum_finest_shock_error"],
        {"finest_shock_position_error": buckley.levels[-1].errors["shock_position"],
         "absolute_tolerance": buckley.acceptance["maximum_finest_shock_error"]},
        "analytic_solution",
    ))
    buckley_balance = max(level.invariants["water_volume_balance_m3"]
                          for level in buckley.levels)
    graph.add_edge(_edge(
        "buckley_conservation", "analytic_buckley_leverett",
        "pymrst_buckley_leverett", "Injected water equals accumulated water",
        buckley_balance <= buckley.acceptance["maximum_water_balance_m3"],
        {"maximum_absolute_imbalance_m3": buckley_balance,
         "absolute_tolerance_m3": buckley.acceptance["maximum_water_balance_m3"]},
        "conservation",
    ))
    bl_min = min(level.invariants["minimum_saturation"] for level in buckley.levels)
    bl_max = max(level.invariants["maximum_saturation"] for level in buckley.levels)
    graph.add_edge(_edge(
        "buckley_saturation_bounds", "analytic_buckley_leverett",
        "pymrst_buckley_leverett", "Explicit transport preserves saturation bounds",
        bl_min >= -1e-12 and bl_max <= 1 + 1e-12,
        {"minimum_saturation": bl_min, "maximum_saturation": bl_max},
        "physical_invariant",
    ))

    graph.add_edge(_edge(
        "terzaghi_pressure_convergence", "analytic_terzaghi", "pymrst_terzaghi",
        "Terzaghi maximum pressure error converges at backward-Euler order",
        terzaghi.observed_orders["pressure_linf"]
        >= terzaghi.acceptance["minimum_pressure_order"],
        {"observed_order": terzaghi.observed_orders["pressure_linf"],
         "minimum_order": terzaghi.acceptance["minimum_pressure_order"]},
        "grid_time_refinement",
    ))
    graph.add_edge(_edge(
        "terzaghi_pressure_accuracy", "analytic_terzaghi", "pymrst_terzaghi",
        "Finest Terzaghi pressure history meets its profile gate",
        terzaghi.levels[-1].errors["pressure_linf"]
        <= terzaghi.acceptance["maximum_finest_pressure_linf"],
        {"finest_relative_linf_error": terzaghi.levels[-1].errors["pressure_linf"],
         "tolerance": terzaghi.acceptance["maximum_finest_pressure_linf"]},
        "analytic_solution",
    ))
    graph.add_edge(_edge(
        "terzaghi_settlement", "analytic_terzaghi", "pymrst_terzaghi",
        "Terzaghi top settlement agrees with the analytic consolidation degree",
        terzaghi.levels[-1].errors["settlement_linf_relative"]
        <= terzaghi.acceptance["maximum_finest_settlement_relative_error"],
        {"finest_relative_error": terzaghi.levels[-1].errors["settlement_linf_relative"],
         "tolerance": terzaghi.acceptance["maximum_finest_settlement_relative_error"]},
        "analytic_solution",
    ))
    tz_min = min(level.invariants["minimum_pressure_ratio"] for level in terzaghi.levels)
    tz_max = max(level.invariants["maximum_pressure_ratio"] for level in terzaghi.levels)
    graph.add_edge(_edge(
        "terzaghi_pressure_bounds", "analytic_terzaghi", "pymrst_terzaghi",
        "Terzaghi excess pressure remains between drained and undrained limits",
        tz_min >= -1e-12 and tz_max <= 1 + 1e-12,
        {"minimum_pressure_ratio": tz_min, "maximum_pressure_ratio": tz_max},
        "physical_invariant",
    ))
    min_settlement_increment = min(
        level.invariants["minimum_settlement_increment"] for level in terzaghi.levels)
    graph.add_edge(_edge(
        "terzaghi_monotone_settlement", "analytic_terzaghi", "pymrst_terzaghi",
        "Settlement increases monotonically as excess pressure dissipates",
        min_settlement_increment > 0,
        {"minimum_settlement_increment_m": min_settlement_increment},
        "physical_invariant",
    ))

    graph.add_edge(_edge(
        "mandel_pressure_convergence", "analytic_mandel", "pymrst_mandel",
        "Mandel maximum pressure error converges at the declared order",
        mandel.observed_orders["pressure_linf"] >= mandel.acceptance["minimum_pressure_order"],
        {"observed_order": mandel.observed_orders["pressure_linf"],
         "minimum_order": mandel.acceptance["minimum_pressure_order"]},
        "grid_time_refinement",
    ))
    graph.add_edge(_edge(
        "mandel_pressure_accuracy", "analytic_mandel", "pymrst_mandel",
        "Finest Mandel pressure history meets the analytic profile gate",
        mandel.levels[-1].errors["pressure_linf"]
        <= mandel.acceptance["maximum_finest_pressure_linf"],
        {"finest_relative_linf_error": mandel.levels[-1].errors["pressure_linf"],
         "tolerance": mandel.acceptance["maximum_finest_pressure_linf"]},
        "analytic_solution",
    ))
    mandel_peak = max(level.invariants["centre_peak_pressure_ratio"] for level in mandel.levels)
    graph.add_edge(_edge(
        "mandel_cryer_overshoot", "analytic_mandel", "pymrst_mandel",
        "The coupled solution reproduces the Mandel--Cryer pressure overshoot",
        mandel_peak >= mandel.acceptance["minimum_overshoot_ratio"],
        {"maximum_centre_pressure_ratio": mandel_peak,
         "minimum_required_ratio": mandel.acceptance["minimum_overshoot_ratio"]},
        "coupling_invariant",
    ))
    mandel_spread = max(level.invariants["maximum_column_pressure_spread"]
                        for level in mandel.levels)
    graph.add_edge(_edge(
        "mandel_column_uniformity", "analytic_mandel", "pymrst_mandel",
        "Plane-strain Mandel pressure is uniform through slab thickness",
        mandel_spread <= mandel.acceptance["maximum_column_pressure_spread"],
        {"maximum_pressure_ratio_spread": mandel_spread,
         "tolerance": mandel.acceptance["maximum_column_pressure_spread"]},
        "physical_invariant",
    ))
    force_error = max(level.invariants["final_plate_force_relative_error"]
                      for level in mandel.levels)
    graph.add_edge(_edge(
        "mandel_constant_plate_load", "analytic_mandel", "pymrst_mandel",
        "The moving plate maintains its prescribed total load",
        force_error <= mandel.acceptance["maximum_plate_force_relative_error"],
        {"maximum_relative_force_error": force_error,
         "tolerance": mandel.acceptance["maximum_plate_force_relative_error"]},
        "mechanical_invariant",
    ))
    graph.add_edge(_edge(
        "mandel_two_way_coupling", "analytic_mandel", "pymrst_mandel",
        "Pressure both rises above its initial value and later dissipates",
        mandel_peak > 1.0 and mandel.levels[-1].errors["pressure_l2"] < 0.005,
        {"peak_pressure_ratio": mandel_peak,
         "finest_l2_profile_error": mandel.levels[-1].errors["pressure_l2"]},
        "coupling_invariant",
    ))

    patch_error = max(level.errors["maximum_relative_displacement"]
                      for level in elasticity.levels)
    rigid_energy = max(level.invariants["maximum_normalized_rigid_energy"]
                       for level in elasticity.levels)
    stiffness_asymmetry = max(level.invariants["relative_stiffness_asymmetry"]
                              for level in elasticity.levels)
    graph.add_edge(_edge(
        "elasticity_affine_recovery", "analytic_elasticity_patch",
        "pymrst_elasticity_patch", "VEM exactly reproduces affine displacement fields",
        patch_error <= elasticity.acceptance["maximum_relative_displacement"],
        {"maximum_relative_error": patch_error,
         "tolerance": elasticity.acceptance["maximum_relative_displacement"]},
        "patch_test",
    ))
    graph.add_edge(_edge(
        "elasticity_rigid_modes", "analytic_elasticity_patch",
        "pymrst_elasticity_patch", "Translations and rotations have zero elastic energy",
        rigid_energy <= elasticity.acceptance["maximum_normalized_rigid_energy"],
        {"maximum_normalized_energy": rigid_energy,
         "tolerance": elasticity.acceptance["maximum_normalized_rigid_energy"]},
        "mechanical_invariant",
    ))
    graph.add_edge(_edge(
        "elasticity_stiffness_symmetry", "analytic_elasticity_patch",
        "pymrst_elasticity_patch", "Assembled elastic stiffness is symmetric",
        stiffness_asymmetry
        <= elasticity.acceptance["maximum_relative_stiffness_asymmetry"],
        {"maximum_relative_asymmetry": stiffness_asymmetry,
         "tolerance": elasticity.acceptance["maximum_relative_stiffness_asymmetry"]},
        "operator_identity",
    ))

    flash_metrics = {
        "material_balance": max(level.errors["material_balance"] for level in flash.levels),
        "fugacity": max(level.errors["fugacity_log_ratio"] for level in flash.levels),
        "closure": max(max(level.errors["liquid_closure"], level.errors["vapour_closure"])
                       for level in flash.levels),
        "two_phase": min(level.invariants["two_phase_fraction"] for level in flash.levels),
        "minimum_composition": min(level.invariants["minimum_composition"]
                                   for level in flash.levels),
    }
    for edge_id, claim, value, limit, comparison, evidence_class in (
        ("flash_material_balance", "Phase split reconstructs the overall feed",
         flash_metrics["material_balance"], flash.acceptance["maximum_material_balance"],
         "maximum", "thermodynamic_identity"),
        ("flash_fugacity_equality", "Liquid and vapour component fugacities are equal",
         flash_metrics["fugacity"], flash.acceptance["maximum_log_fugacity_ratio"],
         "maximum", "thermodynamic_identity"),
        ("flash_phase_closure", "Both phase compositions sum to unity",
         flash_metrics["closure"], flash.acceptance["maximum_composition_closure"],
         "maximum", "composition_identity"),
    ):
        graph.add_edge(_edge(
            edge_id, "analytic_flash_equilibrium", "pymrst_flash_equilibrium", claim,
            value <= limit, {comparison + "_error": value, "tolerance": limit},
            evidence_class,
        ))
    graph.add_edge(_edge(
        "flash_state_coverage", "analytic_flash_equilibrium", "pymrst_flash_equilibrium",
        "The qualification ensemble remains in the two-phase region",
        flash_metrics["two_phase"] >= flash.acceptance["minimum_two_phase_fraction"],
        {"minimum_two_phase_fraction": flash_metrics["two_phase"]}, "state_coverage",
    ))
    graph.add_edge(_edge(
        "flash_physical_bounds", "analytic_flash_equilibrium", "pymrst_flash_equilibrium",
        "Phase fractions and component compositions remain physical",
        flash_metrics["minimum_composition"] >= 0.0,
        {"minimum_component_fraction": flash_metrics["minimum_composition"]},
        "physical_invariant",
    ))

    residual = max(level.errors["maximum_residual"] for level in balance.levels)
    jacobian = max(level.errors["maximum_jacobian_error"] for level in balance.levels)
    global_balance = max(level.invariants["global_source_storage_balance"]
                         for level in balance.levels)
    structure_matches = all(level.invariants["jacobian_nnz"]
                            == level.invariants["expected_jacobian_nnz"]
                            for level in balance.levels)
    graph.add_edge(_edge(
        "manufactured_local_residual", "analytic_accumulation_flux",
        "pymrst_accumulation_flux", "Manufactured source cancels accumulation and flux",
        residual <= balance.acceptance["maximum_residual"],
        {"maximum_absolute_residual": residual,
         "tolerance": balance.acceptance["maximum_residual"]}, "manufactured_solution",
    ))
    graph.add_edge(_edge(
        "manufactured_global_conservation", "analytic_accumulation_flux",
        "pymrst_accumulation_flux", "Global source equals storage on a closed domain",
        global_balance <= balance.acceptance["maximum_global_balance"],
        {"maximum_absolute_imbalance": global_balance,
         "tolerance": balance.acceptance["maximum_global_balance"]}, "conservation",
    ))
    graph.add_edge(_edge(
        "manufactured_ad_jacobian", "analytic_accumulation_flux",
        "pymrst_accumulation_flux", "ADI Jacobian equals the analytic TPFA matrix",
        jacobian <= balance.acceptance["maximum_jacobian_error"],
        {"maximum_absolute_entry_error": jacobian,
         "tolerance": balance.acceptance["maximum_jacobian_error"]}, "derivative_identity",
    ))
    graph.add_edge(_edge(
        "manufactured_jacobian_structure", "analytic_accumulation_flux",
        "pymrst_accumulation_flux", "ADI preserves the expected tridiagonal sparsity",
        structure_matches, {"all_levels_match": structure_matches}, "sparsity_identity",
    ))
    summary = {
        "cases": {key: value.to_dict() for key, value in results.items()},
        "passed": all(value.passed for value in results.values())
                  and all(edge.accepted for edge in graph.edges),
        "accepted_edges": sum(edge.accepted for edge in graph.edges),
        "total_edges": len(graph.edges),
    }
    return graph, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    graph, summary = qualify()
    payload = {"artifact": "pymrst_analytic_foundation_evidence_graph",
               "artifact_schema_version": 1, "graph": graph.to_dict(),
               "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"{'PASS' if summary['passed'] else 'FAIL'}: "
          f"{summary['accepted_edges']}/{summary['total_edges']} analytic edges")
    print(args.output)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
