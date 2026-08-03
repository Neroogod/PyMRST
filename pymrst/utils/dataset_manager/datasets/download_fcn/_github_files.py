"""Download a pinned set of raw GitHub files without requiring git."""

import urllib.request
from pathlib import Path


def download_github_files(repository, revision, files, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for source_name in files:
        target = destination / Path(source_name).name
        if target.is_file():
            continue
        url = f"https://raw.githubusercontent.com/{repository}/{revision}/{source_name}"
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(url) as source, temporary.open("wb") as sink:
                while chunk := source.read(1024 * 1024):
                    sink.write(chunk)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            return False
    return all((destination / Path(name).name).is_file() for name in files)

