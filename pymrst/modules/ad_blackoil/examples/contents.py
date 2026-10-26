"""Catalog for the MRST ad-blackoil example ports."""

EXAMPLES = {
    "aquifertest": "Aquifer deck-processing and model smoke example.",
    "blackoilSectorModelExample": "AD-OO port (pressure-boundary sector of SPE1); matches MRST to round-off.",
    "blackoilTimeIntegrationExample": "SPE1 fully implicit run plus time-integration notes.",
    "blackoilTutorialGravSeg": "AD-OO port of the gravity segregation tutorial; matches MRST.",
    "blackoilTutorialMexAcceleration": "MEX acceleration note; PyMRST uses SciPy/Python.",
    "blackoilTutorialOnePhase": "AD-OO port (closed and open compartment); matches MRST to round-off.",
    "blackoilTutorialPlotHook": "AD-OO port with an after-step plot hook; matches MRST to round-off.",
    "egg/eggExample": "AD-OO port; matches MRST to round-off with a direct solver, to the GMRES tolerance as run.",
    "fieldModelNorneExample": "Full Norne deck/output import with field-scale regression tiers.",
    "immiscibleTimeIntegrationExample": "AD-OO port (implicit, explicit, adaptive implicit, sequential); matches MRST.",
    "multisegmentWellExample": "AD-OO port with MRST's MultisegmentWell; matches MRST to round-off.",
    "simulatorWorkflowExample": "Runnable basic model/schedule/solver workflow.",
    "spe1/blackoilTutorialSPE1": "AD-OO port; matches MRST to round-off.",
    "spe1/simulateSPE1": "AD-OO packed-problem port; matches MRST to round-off.",
    "spe1/timestepControlDemo": "AD-OO port of the timestep-selector demonstration; matches MRST.",
    "spe3/simulateSPE3": "AD-OO packed-problem port; matches MRST (BHP-limit knife edge at step 21).",
    "spe9/blackoilTutorialSPE9": "AD-OO port with CPR; compared with MRST and ECLIPSE.",
    "spe9/simulateSPE9": "AD-OO packed-problem port (diagonal AD, GMRES-ILU(0)); matches MRST.",
    "spe10/tenthCSP_Model_I": "Official SPE10 Model I oil-gas benchmark workflow.",
}


def list_examples() -> dict[str, str]:
    return dict(EXAMPLES)


__all__ = ["EXAMPLES", "list_examples"]


if __name__ == "__main__":
    def test_the_spe_examples_are_listed_as_ports():
        examples = list_examples()
        for name in ("spe1/simulateSPE1", "spe3/simulateSPE3", "spe9/simulateSPE9",
                     "spe9/blackoilTutorialSPE9"):
            assert "port" in examples[name], (name, examples[name])
        print("ok test_the_spe_examples_are_listed_as_ports")

    def test_the_listing_is_a_copy():
        listing = list_examples()
        listing.clear()
        assert list_examples(), "callers must not be able to empty the catalogue"
        print("ok test_the_listing_is_a_copy")

    test_the_spe_examples_are_listed_as_ports()
    test_the_listing_is_a_copy()
    print("\nAll tests passed!")
