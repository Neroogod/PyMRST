# Wells, sources, and boundary conditions

This directory mirrors MRST's `params/wells_and_bc` package.

Public helpers include `add_well`, `vertical_well`, `add_bc`, `add_source`,
`pside`, `psideh`, and `fluxside`. Typed `Well`, `BoundaryCondition`, and
`Source` containers are available to downstream solvers. Well-index
calculation is owned by `pymrst.utils`, matching MRST 2024a, and re-exported
here for convenience.

Example:

```python
from pymrst.params.wells_and_bc import add_source, add_well, pside

wells = add_well(
    [], grid, rock, cells=[0, 1, 2],
    well_type="rate", val=1.0e-4, radius=0.1, name="I1",
)
bc = pside(None, grid, "xmin", pressure=2.0e7, sat=[1.0, 0.0])
src = add_source(None, cells=[10], rates=[-1.0e-4], sat=[0.0, 1.0])
```

Indices are zero-based. Composition arrays normalize to one row per
completion, face, or source. For `vertical_well`, the logical-column form is
preferred; use `top_cell=True` for MRST's alternate top-cell signature.
Multisegment topology extensions live in `multisegment.py`.
