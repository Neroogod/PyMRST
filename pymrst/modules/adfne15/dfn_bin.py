"""Native Bin-Wang legacy-option DFN owner corresponding to ``DFN_bin.m``."""
from .dfn_orig import dfn_orig
from .polyclip import polyclip
from dataclasses import replace
def dfn_bin(**options):
    domain=options.pop('polyDomain',None); result=dfn_orig(**options)
    if domain is not None and result.dimension==3: result=replace(result,fractures=tuple(polyclip(result.fractures,domain)))
    return result
