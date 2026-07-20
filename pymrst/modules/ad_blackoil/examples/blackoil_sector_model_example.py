"""Layout port of MRST ``blackoilSectorModelExample.m``."""

from pymrst.modules.ad_blackoil.examples._unsupported import unsupported_example


def run_blackoil_sector_model_example():
    unsupported_example(
        "blackoilSectorModelExample",
        "the sector-model deck reader and schedule importer have not been ported",
    )


if __name__ == "__main__":
    run_blackoil_sector_model_example()
