"""Ensure the prebuilt SPE10 Model 1/2 property files are installed."""

from pymrst.utils.dataset_manager import download_dataset, get_dataset_path


def make_spe10_data_available(do_build=False):
    root = get_dataset_path("SPE10", skip_available_check=True)
    required = ("model1_data.mat", "krog.txt", "spe10_rock.mat")
    if all(any(root.rglob(name)) for name in required):
        return True
    if do_build:
        # Building is only possible from already downloaded raw CSP files.
        raw = ("perm_case1.dat", "krog.txt", "spe_perm.dat", "spe_phi.dat")
        return all(any(root.rglob(name)) for name in raw)
    _, ok = download_dataset("SPE10", ask_first=False)
    return ok and all(any(root.rglob(name)) for name in required)


makeSPE10DataAvailable = make_spe10_data_available

