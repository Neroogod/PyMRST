# Parameters

This package is the PyMRST counterpart of MRST's `params/` tree. It defines
petrophysical parameters and external forcing through three public areas:

- `pymrst.params.rock`: rock-property construction and conversion;
- `pymrst.params.wells_and_bc`: wells, sources, and boundary conditions;
- `pymrst.params.gravity`: MRST-compatible gravity state and explicit vectors.

The most common functions are re-exported from `pymrst.params`:

```python
from pymrst.params import (
    add_bc, add_source, add_well, gravity, make_rock, pore_volume,
    vertical_well,
)
```

PyMRST uses zero-based cell and face indices. Numerical arrays use NumPy, and
grid/rock structures may be mapping-like objects or typed containers.

See [rock/README.md](rock/README.md) and
[wells_and_bc/README.md](wells_and_bc/README.md) for focused guidance. Full
parity evidence is recorded in the repository-level `params_parity.md` and
`docs/params/parity.csv`.

