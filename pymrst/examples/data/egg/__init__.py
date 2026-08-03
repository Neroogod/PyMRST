"""Egg model examples (MRST ``examples/data/Egg/.../MRST/``).

The Egg model is a 60x60x7 channelised reservoir with 18553 active cells, eight
injectors and four producers. It is small enough to run quickly and awkward
enough to be a real test: the deck is block-centred rather than corner-point,
splits its arrays across INCLUDE files, and builds permeability with `COPY` and
`MULTIPLY` operators.

Reading it end to end exercised four defects in PyMRST, all of which are fixed
and covered by regressions: comma separators in the tokenizer, the missing
GRID-section reader, the reversed `COPY` direction, and the ECLIPSE box
operators being ignored entirely.
"""

__all__: list[str] = []
