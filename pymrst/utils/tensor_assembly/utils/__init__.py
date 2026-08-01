"""Helpers for the tensor-assembly classes (MRST ``tensorAssembly/utils``)."""

from pymrst.utils.tensor_assembly.utils.comp_index_array import comp_index_array
from pymrst.utils.tensor_assembly.utils.concat_index_array import (
    concat_index_array,
)
from pymrst.utils.tensor_assembly.utils.crossindconstruct import crossindconstruct
from pymrst.utils.tensor_assembly.utils.replacefield import replacefield
from pymrst.utils.tensor_assembly.utils.tbldispatch1 import tbldispatch1
from pymrst.utils.tensor_assembly.utils.tbldispatch2 import tbldispatch2

__all__ = ["comp_index_array", "concat_index_array", "crossindconstruct",
           "replacefield", "tbldispatch1", "tbldispatch2"]
