"""Native legacy-option port of ADFNE15 ``DFN_orig.m``."""
from .dfn import generate_dfn
_MAP={'dim':'dimension','n':'count','minl':'minimum_length','mu':'mean_length','maxl':'maximum_length','bbx':'box','dir':'direction','ddir':'direction_variation','dip':'dip','ddip':'dip_variation','q':'polygon_segments'}
def dfn_orig(**options):
    translated={_MAP.get(k,k):v for k,v in options.items() if k not in {'asep','polyDomain'}}
    return generate_dfn(**translated)
