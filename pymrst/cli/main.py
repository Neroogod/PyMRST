"""The `pymrst` command: report what the installation can do, and fetch data.

Two subcommands, deliberately kept apart:

    pymrst doctor            what this installation can do
    pymrst datasets ...      what data is available and installed

They are separate because "installed correctly" and "benchmark data
downloaded" are different states, and running them together produces a report
where a missing 470 MB optional dataset looks like a broken installation.

This is purely additive. Nothing here is imported by any PyMRST module, and
running a simulation never touches it.
"""

from __future__ import annotations

import argparse
import sys

from pymrst.cli.capabilities import (
    CAPABILITIES, cuda_status, environment, writable_paths,
)

_OK, _NO = "OK", "not installed"


def _status_column(label: str, status: str, note: str = "") -> str:
    return f"  {label:<26} {status:<22} {note}".rstrip()


def doctor(argv=None) -> int:
    """Report capability status. Exits non-zero only if the core is broken."""
    print("Environment")
    for key, value in environment().items():
        print(_status_column(key, value))

    print("\nCapabilities")
    core_ok = True
    for capability in CAPABILITIES:
        available = capability.available
        if capability.extra is None:
            core_ok = core_ok and available
        status = _OK if available else _NO
        note = capability.detail if available else capability.remedy
        print(_status_column(capability.name, status, note))

    print(_status_column("CUDA", cuda_status()))

    print("\nDirectories")
    for label, path, ok, note in writable_paths():
        print(_status_column(label, "writable" if ok else "PROBLEM",
                             f"{path}  ({note})"))

    if not core_ok:
        print("\nThe core dependencies are missing; PyMRST cannot run.",
              file=sys.stderr)
        return 1
    print("\nDatasets are reported separately: pymrst datasets list")
    return 0


def _load_datasets():
    """Imported lazily so `doctor` still works if the package is broken."""
    from pymrst.utils.dataset_manager import get_available_datasets
    return get_available_datasets()


def datasets_list(argv=None) -> int:
    infos, present = _load_datasets()
    print(f"{len(infos)} datasets known "
          f"({sum(bool(p) for p in present)} installed)\n")
    print(f"  {'name':<26} {'installed':<11} {'size':>8}  source")
    for info, installed in zip(infos, present):
        size = f"{info.filesize:.1f} MB" if info.filesize and info.filesize == info.filesize else "-"
        origin = "direct download" if info.fileurl else "manual - see fetch"
        print(f"  {info.name:<26} {'yes' if installed else 'no':<11} {size:>8}  {origin}")
    print("\n  pymrst datasets fetch <name>     terms are shown before download")
    return 0


def datasets_fetch(name: str, assume_yes: bool = False) -> int:
    """Show provenance and terms, then download only on agreement.

    A benchmark deck is not automatically redistributable, and several here
    are not: their `fileurl` is empty and their `instructions` say where to
    obtain them. Those must never be fetched silently, and the ones that *can*
    be downloaded should still say where they came from first.
    """
    infos, present = _load_datasets()
    match = next((i for i in infos if i.name.lower() == name.lower()), None)
    if match is None:
        print(f"unknown dataset {name!r}; try: pymrst datasets list",
              file=sys.stderr)
        return 2
    installed = dict(zip((i.name for i in infos), present)).get(match.name, False)

    print(f"{match.name}\n  {match.description or '(no description)'}")
    if match.modelType:
        print(f"  model: {match.modelType}")
    if match.cells and match.cells == match.cells:
        print(f"  cells: {match.cells:,}")
    if match.website:
        print(f"  origin: {match.website}")
    print(f"  terms: {_terms(match)}")

    if installed:
        print("\nAlready installed.")
        return 0
    if not match.fileurl:
        print("\nThis dataset is not redistributed by PyMRST and must be "
              "obtained from its origin:")
        print(f"  {match.instructions or 'see the website above'}")
        return 3

    print(f"\nSource: {match.fileurl}")
    from pymrst.utils.dataset_manager import download_dataset
    try:
        destination, ok = download_dataset(match.name, ask_first=not assume_yes)
    except EOFError:
        # `download_dataset` prompts on stdin. With no terminal - a CI job, a
        # pipeline - that raises rather than declining, which reads as a crash.
        print("\nNo terminal to confirm on; re-run with --yes to accept.",
              file=sys.stderr)
        return 4
    if not ok:
        print(f"\nNot downloaded. Target was {destination}", file=sys.stderr)
        return 5
    print(f"Installed to {destination}")
    return 0


def _terms(info) -> str:
    """What is known about redistribution, stated rather than implied.

    `DatasetInfo` carries no licence field, so this reports the distinction
    the data already makes - a dataset PyMRST can download directly versus one
    whose owner requires you to obtain it yourself - and says plainly that the
    licence itself is unrecorded. Overstating this would be worse than
    reporting nothing.
    """
    licence = getattr(info, "licence", None) or getattr(info, "license", None)
    if licence:
        return str(licence)
    if info.fileurl:
        return ("redistributed by the upstream MRST dataset host; licence not "
                "recorded in DatasetInfo - check the origin before publishing")
    return "not redistributable by PyMRST; obtain from the origin"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pymrst",
        description="PyMRST installation and dataset utilities.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="report what this installation can do")

    data = sub.add_parser("datasets", help="list or fetch benchmark datasets")
    data_sub = data.add_subparsers(dest="action")
    data_sub.add_parser("list", help="list known datasets")
    fetch = data_sub.add_parser("fetch", help="download a dataset")
    fetch.add_argument("name")
    fetch.add_argument("--yes", "-y", action="store_true",
                       help="accept the download without a prompt; terms are "
                            "still printed first")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "datasets":
        if args.action == "list":
            return datasets_list()
        if args.action == "fetch":
            return datasets_fetch(args.name, assume_yes=args.yes)
        data.print_help()
        return 2
    parser.print_help()
    return 2


__all__ = ["datasets_fetch", "datasets_list", "doctor", "main"]


if __name__ == "__main__":
    # Self-tests, not the command. The console script is declared in
    # pyproject as `pymrst.cli.main:main`, so this guard is not what makes
    # `pymrst` work - and making it run the CLI meant `python3 main.py`
    # printed usage and exited 2, which `tests/test_all.py` reads as a failing
    # module. Running a file in this tree runs its tests; that is the rule the
    # sweep enforces, and this file broke it.

    def test_doctor_reports_and_succeeds():
        assert doctor() == 0
        print("ok test_doctor_reports_and_succeeds")

    def test_an_unknown_dataset_is_refused():
        assert datasets_fetch("no_such_dataset_xyz") == 2
        print("ok test_an_unknown_dataset_is_refused")

    def test_bare_invocation_prints_help_without_raising():
        assert main([]) == 2
        print("ok test_bare_invocation_prints_help_without_raising")

    def test_datasets_list_runs():
        assert datasets_list() == 0
        print("ok test_datasets_list_runs")

    test_doctor_reports_and_succeeds()
    test_an_unknown_dataset_is_refused()
    test_bare_invocation_prints_help_without_raising()
    test_datasets_list_runs()
    print("\nAll tests passed!")
