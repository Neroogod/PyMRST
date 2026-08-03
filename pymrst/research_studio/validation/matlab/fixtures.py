"""Creation, persistence, and strict validation of MATLAB fixtures."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .execution import discover_matlab, generation_command, run_generator
from .provenance import infer_mrst_version, matlab_release_from_output, sha256_file
from .schemas import MatlabFixtureManifest, MatlabFixtureSpec


class StaleMatlabFixtureError(RuntimeError):
    """A reference or its generator no longer matches recorded provenance."""


def load_manifest(path: Path) -> MatlabFixtureManifest:
    return MatlabFixtureManifest.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8")))


def write_manifest(path: Path, manifest: MatlabFixtureManifest) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def validate_fixture(spec: MatlabFixtureSpec, *, require_manifest: bool = True
                     ) -> MatlabFixtureManifest | None:
    missing = [path for path in (spec.reference_path, spec.generator_path, spec.mrst_root,
                                 *spec.input_artifacts.values())
               if not path.exists()]
    if missing:
        raise FileNotFoundError(f"MATLAB fixture dependencies are missing: {missing}")
    if not spec.manifest_path.exists():
        if require_manifest:
            raise StaleMatlabFixtureError(
                f"fixture manifest is missing: {spec.manifest_path}")
        return None
    manifest = load_manifest(spec.manifest_path)
    mismatches = []
    checks = {
        "fixture_id": (manifest.fixture_id, spec.fixture_id),
        "case_name": (manifest.case_name, spec.case_name),
        "modules": (manifest.modules, spec.modules),
        "mappings": (manifest.mappings, spec.mappings),
        "tolerances": (manifest.tolerances, spec.tolerances),
        "solver_settings": (manifest.solver_settings, spec.solver_settings),
        "generator_function": (manifest.generator_function, spec.generator_function),
        "generator_sha256": (manifest.generator_sha256, sha256_file(spec.generator_path)),
        "reference_sha256": (manifest.reference_sha256, sha256_file(spec.reference_path)),
        "reference_bytes": (manifest.reference_bytes, spec.reference_path.stat().st_size),
        "mrst_root": (Path(manifest.mrst_root).resolve(), spec.mrst_root),
    }
    current_inputs = {
        name: {"path": str(path), "sha256": sha256_file(path),
               "bytes": path.stat().st_size}
        for name, path in spec.input_artifacts.items()
    }
    checks["input_artifacts"] = (manifest.input_artifacts, current_inputs)
    for name, (recorded, current) in checks.items():
        if recorded != current:
            mismatches.append(name)
    if mismatches:
        raise StaleMatlabFixtureError(
            "MATLAB fixture provenance mismatch: " + ", ".join(mismatches))
    return manifest


def generate_fixture(spec: MatlabFixtureSpec, *, executable: Path | str | None = None,
                     timeout_seconds: float = 1800.0) -> MatlabFixtureManifest:
    matlab = discover_matlab(executable)
    before = spec.reference_path.stat().st_mtime_ns if spec.reference_path.exists() else None
    result = run_generator(spec, executable=matlab, timeout_seconds=timeout_seconds)
    if not result.succeeded:
        raise RuntimeError(
            f"MATLAB generator failed with status {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    if not spec.reference_path.is_file() or spec.reference_path.stat().st_size == 0:
        raise RuntimeError("MATLAB generator did not create a nonempty reference fixture")
    after = spec.reference_path.stat().st_mtime_ns
    if before is not None and after == before:
        raise RuntimeError("MATLAB generator succeeded but did not rewrite the reference fixture")
    manifest = MatlabFixtureManifest(
        fixture_id=spec.fixture_id, case_name=spec.case_name,
        matlab_executable=str(matlab),
        matlab_release=matlab_release_from_output(result.stdout),
        mrst_root=str(spec.mrst_root), mrst_version=infer_mrst_version(spec.mrst_root),
        modules=spec.modules, generator_function=spec.generator_function,
        generator_sha256=sha256_file(spec.generator_path),
        reference_sha256=sha256_file(spec.reference_path),
        reference_bytes=spec.reference_path.stat().st_size,
        mappings=spec.mappings, tolerances=spec.tolerances,
        input_artifacts={
            name: {"path": str(path), "sha256": sha256_file(path),
                   "bytes": path.stat().st_size}
            for name, path in spec.input_artifacts.items()
        },
        solver_settings=spec.solver_settings,
        regeneration_command=generation_command(spec, matlab),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        metadata={"elapsed_seconds": result.elapsed_seconds,
                  "stdout_tail": result.stdout[-2000:],
                  "stderr_tail": result.stderr[-2000:]},
    )
    write_manifest(spec.manifest_path, manifest)
    return validate_fixture(spec)


__all__ = ["StaleMatlabFixtureError", "generate_fixture", "load_manifest",
           "validate_fixture", "write_manifest"]
