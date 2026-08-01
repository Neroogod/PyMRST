# Rock properties

This directory mirrors MRST's `params/rock` package.

Public functions:

- `make_rock`: construct rock data from compact or per-cell properties;
- `perm_tensor`: expand compact permeability representations;
- `pore_volume`: compute bulk-volume, porosity, and net-to-gross products;
- `grdecl2rock`: convert GRDECL properties, active maps, and multipliers;
- `gaussian_field` and `log_norm_layers`: generate stochastic fields;
- `permeability_converter`: transform permeability tensors.

Example:

```python
from pymrst.params.rock import make_rock, perm_tensor, pore_volume

rock = make_rock(grid, permeability=100.0, porosity=0.25)
K, row, col = perm_tensor(rock, dim=3)
pv = pore_volume(grid, rock)
```

Scalar properties are broadcast over the grid. PyMRST uses SI units unless a
calling workflow explicitly converts deck units.
`permeability_converter(..., off_diagonal_mode="mrst")` preserves MRST's
observable unsupported off-diagonal behavior; `"corrected"` requests the
usable symmetric-tensor transform.
