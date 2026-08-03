"""The 10th SPE Comparative Solution Project dataset.

Mirrors MRST's top-level ``modules/spe10``. PyMRST also carries SPE10 material
under ``modules/deckformat/spe10`` (model 1) and
``modules/ad_blackoil/examples/spe10``; this module holds the model 2 rock,
which is what MRST's ``spe10`` module is for.

Consolidating the three locations is a layout question recorded in
``examples_implementation.md`` rather than settled here.
"""

from pymrst.modules.spe10.get_spe10_rock import get_spe10_rock

__all__ = ["get_spe10_rock"]
