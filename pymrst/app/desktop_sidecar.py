"""Private loopback API process owned by the PyMRST Studio desktop shell."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import threading

from .application import create_application
from .schemas.deployment import ServerDeploymentSettings


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="pymrst-studio-sidecar")
    result.add_argument("--workspace", type=Path, required=True)
    result.add_argument("--port", type=int, default=0)
    return result


def _token_from_environment() -> str:
    token = os.environ.get("PYMRST_DESKTOP_BOOTSTRAP_TOKEN", "").strip()
    if len(token) < 32:
        raise RuntimeError("missing or invalid desktop bootstrap credential")
    return token


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if not 0 <= arguments.port <= 65535:
        raise SystemExit("sidecar port must lie in [0, 65535]")
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit("desktop sidecar requires the PyMRST app dependency profile") from error
    token = _token_from_environment()
    context = create_application(arguments.workspace)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", arguments.port))
        listener.listen(128)
        port = int(listener.getsockname()[1])
        settings = ServerDeploymentSettings(
            allowed_hosts=("127.0.0.1", "localhost"),
            allowed_origins=("tauri://localhost", "http://tauri.localhost"),
            expose_api_documentation=False,
        )
        from .api.application import create_fastapi_app
        application = create_fastapi_app(
            context, settings, desktop_bootstrap_token=token,
        )
        async def emit_readiness():
            # Emitted from the application startup lifecycle, after Uvicorn has
            # initialized the serving loop around the already-bound socket.
            print(json.dumps({
                "event": "ready", "endpoint": f"http://127.0.0.1:{port}",
                "pid": os.getpid(), "database_schema": context.database.schema_version,
            }, sort_keys=True), flush=True)

        application.router.add_event_handler("startup", emit_readiness)
        configuration = uvicorn.Config(
            application, log_level="warning", access_log=False,
            server_header=False, proxy_headers=False,
        )
        server = uvicorn.Server(configuration)
        def request_shutdown():
            try:
                command = sys.stdin.readline().strip()
            except (OSError, ValueError):
                command = ""
            if command in {"", "shutdown"}:
                server.should_exit = True

        threading.Thread(
            target=request_shutdown, name="pymrst-desktop-shutdown", daemon=True,
        ).start()
        server.run(sockets=[listener])
    finally:
        listener.close()
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parser"]
