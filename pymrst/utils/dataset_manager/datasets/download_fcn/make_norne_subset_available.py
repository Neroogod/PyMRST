"""Install the pinned OPM Norne geometry/property subset."""

from pymrst.utils.dataset_manager import get_dataset_path
from pymrst.utils.dataset_manager.datasets.download_fcn._github_files import download_github_files

FILES = ("norne/INCLUDE/GRID/IRAP_1005.GRDECL",
         "norne/INCLUDE/GRID/ACTNUM_0704.prop",
         "norne/INCLUDE/PETRO/PERM_0704.prop",
         "norne/INCLUDE/PETRO/PORO_0704.prop",
         "norne/INCLUDE/PETRO/NTG_0704.prop",
         "norne/INCLUDE/PETRO/MULTZ_HM_1.INC",
         "norne/INCLUDE/PETRO/MULTZ_JUN_05_MOD.INC")


def make_norne_subset_available():
    root = get_dataset_path("Norne", skip_available_check=True) / "INCLUDE"
    return download_github_files("OPM/opm-tests", "5969ec63b", FILES, root)


makeNorneSubsetAvailable = make_norne_subset_available

