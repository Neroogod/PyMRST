"""Catalog for the MRST ad-blackoil example ports."""

EXAMPLES = {
    "aquifertest": "Aquifer deck-processing and model smoke example.",
    "blackoilSectorModelExample": "Requires sector deck import not present in PyMRST.",
    "blackoilTimeIntegrationExample": "SPE1 fully implicit run plus time-integration notes.",
    "blackoilTutorialGravSeg": "Runnable two-phase gravity segregation port.",
    "blackoilTutorialMexAcceleration": "MEX acceleration note; PyMRST uses SciPy/Python.",
    "blackoilTutorialOnePhase": "Runnable WaterModel depletion/open-boundary port.",
    "blackoilTutorialPlotHook": "Runnable schedule with Python callback hook.",
    "egg/eggExample": "Requires EGG dataset import not present in PyMRST.",
    "fieldModelNorneExample": "Norne deck/geomech workflow lives in examples/run_norne_geomech.py.",
    "immiscibleTimeIntegrationExample": "Runnable fully implicit 1D heterogeneous waterflood port.",
    "multisegmentWellExample": "Runnable reduced MS-well packing port.",
    "simulatorWorkflowExample": "Runnable basic model/schedule/solver workflow.",
    "spe1/blackoilTutorialSPE1": "Runnable SPE1 graph-model port.",
    "spe1/simulateSPE1": "Runnable SPE1 helper.",
    "spe1/timestepControlDemo": "Runnable timestep-selector demonstration.",
    "spe3/simulateSPE3": "Requires SPE3 deck support not present in PyMRST.",
    "spe9/blackoilTutorialSPE9": "Requires SPE9 deck support not present in PyMRST.",
    "spe9/simulateSPE9": "Requires SPE9 deck support not present in PyMRST.",
    "spe10/tenthCSP_Model_I": "Requires SPE10 dataset support not present in PyMRST.",
}


def list_examples() -> dict[str, str]:
    return dict(EXAMPLES)


__all__ = ["EXAMPLES", "list_examples"]
