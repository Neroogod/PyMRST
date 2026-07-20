"""Layout port of MRST ``spe10/tenthCSP_Model_I.m``."""

from pymrst.modules.ad_blackoil.examples._unsupported import unsupported_example


def run_tenth_csp_model_i():
    unsupported_example(
        "tenthCSP_Model_I",
        "SPE10 benchmark property loading and deck setup are not present in PyMRST",
    )


if __name__ == "__main__":
    run_tenth_csp_model_i()
