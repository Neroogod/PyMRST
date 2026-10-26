"""Discrete TPFA operators for this port's models.

MRST's ``setupOperatorsTPFA.m`` is ported in full in
`setup_operators_tpfa.py`; this module keeps the import path the models have
always used.
"""

from pymrst.modules.ad_core.utils.setup_operators_tpfa import (
    Operators,
    setupOperatorsTPFA,
    setup_operators_tpfa,
)

__all__ = ["Operators", "setupOperatorsTPFA", "setup_operators_tpfa"]


if __name__ == "__main__":
    def test_the_import_path_reaches_the_port():
        import importlib
        module = importlib.import_module("pymrst.modules.ad_core.utils.setup_operators_tpfa")
        assert setup_operators_tpfa is module.setup_operators_tpfa
        print("ok test_the_import_path_reaches_the_port")

    test_the_import_path_reaches_the_port()
    print("\nAll tests passed!")
