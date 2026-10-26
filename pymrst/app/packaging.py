"""Cross-platform managed-sidecar build planning and qualification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile

from .schemas._common import utc_now
from .services.example_service import ExampleCatalogService


SUPPORTED_TARGETS = frozenset({
    "aarch64-apple-darwin", "x86_64-apple-darwin",
    "x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc",
    "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu",
})


def native_target() -> str:
    system, machine = platform.system().casefold(), platform.machine().casefold()
    architecture = {"arm64": "aarch64", "aarch64": "aarch64",
                    "amd64": "x86_64", "x86_64": "x86_64"}.get(machine)
    suffix = {"darwin": "apple-darwin", "windows": "pc-windows-msvc",
              "linux": "unknown-linux-gnu"}.get(system)
    if not architecture or not suffix:
        raise ValueError(f"unsupported sidecar build host: {system}/{machine}")
    target = f"{architecture}-{suffix}"
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported sidecar build target: {target}")
    return target


def sidecar_filename(target: str) -> str:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported sidecar target: {target}")
    extension = ".exe" if "windows" in target else ""
    return f"pymrst-sidecar-{target}{extension}"


@dataclass(frozen=True, slots=True)
class SidecarBuildReceipt:
    target: str
    relative_path: str
    sha256: str
    size_bytes: int
    python_version: str
    created_at: str

    def to_dict(self):
        return {
            "schema_version": 1, "target": self.target,
            "relative_path": self.relative_path, "sha256": self.sha256,
            "size_bytes": self.size_bytes, "python_version": self.python_version,
            "created_at": self.created_at,
        }


def build_sidecar(
    repository_root: str | Path, *, target: str | None = None,
    python_executable: str | Path = sys.executable,
) -> SidecarBuildReceipt:
    root, target = Path(repository_root).resolve(), target or native_target()
    python_path = Path(python_executable).expanduser()
    if not python_path.is_absolute():
        resolved = shutil.which(str(python_path))
        if not resolved:
            raise FileNotFoundError(f"Python executable was not found: {python_executable}")
        python_path = Path(resolved)
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable was not found: {python_path}")
    if target != native_target():
        raise ValueError(
            "managed sidecars must be built natively on their target operating system and architecture"
        )
    output_root = root / "ui" / "src-tauri" / "binaries"
    output_root.mkdir(parents=True, exist_ok=True)
    filename = sidecar_filename(target)
    # Each catalog entry's target module is only reached through a runtime
    # importlib.import_module() call inside example_entry.py, invisible to
    # PyInstaller's static import analysis — same reason the worker owner
    # modules below need explicit hidden-imports.
    example_modules = sorted({item.module for item in ExampleCatalogService().list()})
    with tempfile.TemporaryDirectory(prefix="pymrst-sidecar-build-") as temporary:
        temporary = Path(temporary)
        command = [
            # Preserve virtual-environment launchers instead of resolving their
            # symlink to the base interpreter.  Python uses the launcher path to
            # discover pyvenv.cfg and the environment's installed packages.
            str(python_path), "-m", "PyInstaller",
            "--noconfirm", "--clean", "--onefile", "--noupx",
            "--name", filename.removesuffix(".exe"),
            "--distpath", str(temporary / "dist"),
            "--workpath", str(temporary / "work"),
            "--specpath", str(temporary / "spec"),
            "--paths", str(root),
            # Worker owners are selected from a closed registry at runtime, so
            # PyInstaller cannot discover them through static imports.
            "--hidden-import", "pymrst.app.workers.process_entry",
            "--hidden-import", "pymrst.app.workers.protocol_jobs",
            "--hidden-import", "pymrst.app.workers.simulation_jobs",
            "--hidden-import", "pymrst.app.workers.ensemble_jobs",
            "--hidden-import", "pymrst.app.workers.workload_entry",
            "--hidden-import", "pymrst.app.workers.construction_jobs",
            "--hidden-import", "pymrst.app.workers.example_entry",
            *[arg for module in example_modules for arg in ("--hidden-import", module)],
            "--hidden-import", "uvicorn.logging",
            "--hidden-import", "uvicorn.loops.auto",
            "--hidden-import", "uvicorn.protocols.http.auto",
            "--hidden-import", "uvicorn.protocols.websockets.auto",
            "--hidden-import", "uvicorn.lifespan.on",
            # The base desktop distribution executes the NumPy/SciPy backend.
            # Optional ML and native-rendering profiles are separate artifacts;
            # accidentally discovering them through broad package initializers
            # makes the sidecar enormous and non-reproducible across hosts.
            "--exclude-module", "torch",
            "--exclude-module", "tensorflow",
            "--exclude-module", "PyQt5",
            "--exclude-module", "pyvista",
            "--exclude-module", "vtk",
            "--exclude-module", "vtkmodules",
            "--exclude-module", "matplotlib",
            "--exclude-module", "plotly",
            "--exclude-module", "streamlit",
            str(root / "scripts" / "studio_sidecar_entry.py"),
        ]
        build_environment = os.environ.copy()
        # Keep builder caches inside the disposable build root. This prevents
        # cross-release contamination and works in locked-down CI runners where
        # the user's Library/AppData cache is intentionally not writable.
        build_environment["PYINSTALLER_CONFIG_DIR"] = str(temporary / "cache")
        build_environment["MPLCONFIGDIR"] = str(temporary / "matplotlib")
        completed = subprocess.run(
            command, cwd=root, check=False, env=build_environment,
        )
        if completed.returncode:
            raise RuntimeError(f"PyInstaller sidecar build failed with exit code {completed.returncode}")
        built = temporary / "dist" / filename
        if not built.is_file():
            raise RuntimeError(f"PyInstaller did not produce {filename}")
        target_path = output_root / filename
        os.replace(built, target_path)
    if os.name != "nt":
        target_path.chmod(target_path.stat().st_mode | stat.S_IXUSR)
    digest = hashlib.sha256()
    with target_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    version_probe = subprocess.run(
        [str(python_path), "-c", "import platform; print(platform.python_version())"],
        cwd=root, check=True, capture_output=True, text=True,
    )
    receipt = SidecarBuildReceipt(
        target, target_path.relative_to(root).as_posix(), digest.hexdigest(),
        target_path.stat().st_size, version_probe.stdout.strip(), utc_now(),
    )
    receipt_path = output_root / f"{filename}.receipt.json"
    receipt_path.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n")
    return receipt


__all__ = [
    "SUPPORTED_TARGETS", "SidecarBuildReceipt", "build_sidecar", "native_target",
    "sidecar_filename",
]
