"""Create MRST's aggregate ``NORNE.GRDECL`` include deck."""

from pymrst.utils.dataset_manager import get_dataset_path
from pymrst.utils.dataset_manager.datasets.download_fcn.make_norne_subset_available import make_norne_subset_available

BASE = """INCLUDE
  'INCLUDE/IRAP_1005.GRDECL' /
INCLUDE
  'INCLUDE/ACTNUM_0704.prop' /
INCLUDE
  'INCLUDE/PERM_0704.prop' /
INCLUDE
  'INCLUDE/PORO_0704.prop' /
INCLUDE
  'INCLUDE/NTG_0704.prop' /
COPY
   'PERMX' 'PERMY' /
   'PERMX' 'PERMZ' /
/
"""


def make_norne_grdecl():
    root = get_dataset_path("Norne", skip_available_check=True)
    target = root / "NORNE.GRDECL"
    if target.is_file():
        return True
    if not make_norne_subset_available():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    text = BASE
    if (root / "INCLUDE/MULTZ_HM_1.INC").is_file():
        text += "INCLUDE\n  'INCLUDE/MULTZ_HM_1.INC' /\n"
    if (root / "INCLUDE/MULTZ_JUN_05_MOD.INC").is_file():
        text += "INCLUDE\n  'INCLUDE/MULTZ_JUN_05_MOD.INC' /\n"
    target.write_text(text, encoding="utf-8")
    return True


makeNorneGRDECL = make_norne_grdecl

