"""Index-based tensor assembly (MRST ``utils/tensorAssembly``).

A small algebra for building sparse operators by contracting *named* indices
rather than juggling positions and reshapes. `IndexArray` lists multi-indices;
`TensorMap` and `TensorProd` describe how two of them line up; `SparseTensor`
turns the result into a matrix.

**This is not `SparseMultiArray`.** MRST has two unrelated tensor classes -
that one lives in `modules/vemmech/ad_version/`, is four times the size, and
does Einstein-style contraction with AD support. PyMRST ports it separately, at
its own MRST path. The two are easy to confuse by name and share no code.
"""

from pymrst.utils.tensor_assembly.cross_index_array import cross_index_array
from pymrst.utils.tensor_assembly.cross_index_array_generator import (
    CrossIndexArrayGenerator,
)
from pymrst.utils.tensor_assembly.index_array import IndexArray
from pymrst.utils.tensor_assembly.proj_index_array import proj_index_array
from pymrst.utils.tensor_assembly.sort_index_array import sort_index_array
from pymrst.utils.tensor_assembly.sparse_tensor import SparseTensor
from pymrst.utils.tensor_assembly.tensor_map import TensorMap
from pymrst.utils.tensor_assembly.tensor_prod import TensorProd

__all__ = ["CrossIndexArrayGenerator", "IndexArray", "SparseTensor",
           "TensorMap", "TensorProd", "cross_index_array", "proj_index_array",
           "sort_index_array"]
