"""Dataset location for Norne include files."""

import os
from pathlib import Path


def data_directory() -> Path:
    """Return the directory holding Norne include files."""
    root = os.environ.get("MRST_DATA_DIR")
    base = Path(root) if root else Path.home() / ".pymrst" / "datasets"
    return base / "norne" / "INCLUDE"


if __name__ == "__main__":

    def test_data_directory() -> None:
        path = data_directory()
        assert path.name == "INCLUDE"
        assert path.parent.name == "norne"
        print("ok test_data_directory")

    test_data_directory()
    print("\nAll tests passed!")

