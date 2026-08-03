"""Examples that demonstrate how to work with external datasets.

Port of MRST's ``examples/datasets/``. Each file loads a published reservoir
model and inspects its grid, rock and structure.

These are the examples most exposed to the outside world: they need real
payloads, which may or may not be installed, and they read production decks
rather than hand-written fixtures. Every module here therefore locates its data
without triggering a download, reports absence rather than failing, and asserts
the model's published dimensions instead of merely that something loaded. A
grid built from a mis-read ACTNUM still looks entirely reasonable.
"""

__all__: list[str] = []
