"""Download Norne include files, matching MRST `makeNorneSubsetAvailable`."""

import shutil
import ssl
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from pymrst.modules.deckformat.norne.constants import FILES, REPO, REVISION
from pymrst.modules.deckformat.norne.data_directory import data_directory


def fetch_norne(dest: Optional[Path] = None, verbose: bool = True) -> Path:
    """Download the Norne include files if they are not already present."""
    dest = Path(dest) if dest else data_directory()
    dest.mkdir(parents=True, exist_ok=True)
    for remote in FILES:
        local = dest / Path(remote).name
        if local.exists() and local.stat().st_size > 0:
            continue
        url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/norne/INCLUDE/{remote}"
        if verbose:
            print(f"  downloading {Path(remote).name} ...", flush=True)
        _download(url, local)
    return dest


def _download(url: str, local: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=180) as response:
            local.write_bytes(response.read())
        return
    except urllib.error.URLError as exc:
        if not isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise

    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=180, context=context) as response:
            local.write_bytes(response.read())
        return
    except ImportError:
        pass

    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError(
            f"Cannot verify HTTPS certificates and curl is not available. "
            f"Download {url} to {local} by hand."
        )
    subprocess.run([curl, "-fsSL", url, "-o", str(local)], check=True)


if __name__ == "__main__":

    def test_fetch_constants_are_available() -> None:
        assert FILES
        assert REPO == "OPM/opm-tests"
        assert REVISION == "5969ec63b"
        print("ok test_fetch_constants_are_available")

    test_fetch_constants_are_available()
    print("\nAll tests passed!")

