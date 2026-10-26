"""Entrypoint for both the managed API and its frozen isolated workers."""

import sys


def main() -> int:
    # A one-file executable cannot launch ``sys.executable -m ...``: its
    # executable is the sidecar bootloader, not a general Python interpreter.
    # Keep worker execution isolated while routing it through a private mode.
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        from pymrst.app.workers.process_entry import main as worker_main
        return worker_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "--workload-worker":
        from pymrst.app.workers.workload_entry import main as workload_main
        return workload_main(sys.argv[2:])
    from pymrst.app.desktop_sidecar import main as sidecar_main
    return sidecar_main()


if __name__ == "__main__":
    raise SystemExit(main())
