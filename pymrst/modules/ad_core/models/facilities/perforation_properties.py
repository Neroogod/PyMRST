"""Pack and unpack well-perforation properties.

MRST's `packPerforationProperties.m` and `unpackPerforationProperties.m` are
ported in full in `pack_perforation_properties.py` and
`unpack_perforation_properties.py`; this module keeps the import path the
facility models have always used.
"""

from pymrst.modules.ad_core.models.facilities.pack_perforation_properties import (
    packPerforationProperties,
    pack_perforation_properties,
)
from pymrst.modules.ad_core.models.facilities.unpack_perforation_properties import (
    unpackPerforationProperties,
    unpack_perforation_properties,
)

__all__ = ["packPerforationProperties", "pack_perforation_properties",
           "unpackPerforationProperties", "unpack_perforation_properties"]


if __name__ == "__main__":
    def test_the_import_path_reaches_the_ports():
        import importlib
        module = importlib.import_module(
            "pymrst.modules.ad_core.models.facilities.pack_perforation_properties")
        assert pack_perforation_properties is module.pack_perforation_properties
        print("ok test_the_import_path_reaches_the_ports")

    test_the_import_path_reaches_the_ports()
    print("\nAll tests passed!")
