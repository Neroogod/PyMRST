"""Install the pinned Norne initial-ensemble input files."""

from pymrst.utils.dataset_manager import get_dataset_path
from pymrst.utils.dataset_manager.datasets.download_fcn._github_files import download_github_files

FILES = ("ACTNUM_0704.prop", "FAULTMULT_AUG-2006.INC", "NTG_0704.prop",
         "PERM_0704.prop", "PORO_0704.prop", "FastGaussian.m",
         "NorneGeostat.m")


def download_norne_ensemble_files():
    root = get_dataset_path("Norne_Ensemble", skip_available_check=True) / "data"
    return download_github_files("rolfjl/Norne-Initial-Ensemble", "56ed2be",
                                 FILES, root)


downloadNorneEnsembleFiles = download_norne_ensemble_files

