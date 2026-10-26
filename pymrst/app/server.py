"""Conservative command-line launcher for PyMRST Studio's optional HTTP API."""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path

from .application import create_application
from .schemas.deployment import DeploymentEnvironment, ServerDeploymentSettings
from .schemas.security import DeploymentMode


def _loopback(host: str) -> bool:
    normalized = str(host).strip().casefold()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="pymrst-studio",
        description="Launch the governed PyMRST Studio application API.",
    )
    result.add_argument("workspace", type=Path, help="Studio workspace root")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    result.add_argument("--ssl-certfile")
    result.add_argument("--ssl-keyfile")
    result.add_argument(
        "--allow-development-network", action="store_true",
        help="Explicitly permit non-loopback binding in shared development mode",
    )
    return result


def validate_launch(
    *, host: str, mode: DeploymentMode, settings: ServerDeploymentSettings,
    ssl_certfile: str = "", ssl_keyfile: str = "",
    allow_development_network: bool = False,
) -> None:
    if not 1 <= len(str(host)) <= 253:
        raise ValueError("invalid server host")
    if bool(ssl_certfile) != bool(ssl_keyfile):
        raise ValueError("TLS certificate and key must be supplied together")
    if not _loopback(host):
        if mode is not DeploymentMode.SHARED:
            raise ValueError("non-loopback binding requires shared deployment mode")
        if (settings.environment is not DeploymentEnvironment.PRODUCTION
                and not allow_development_network):
            raise ValueError(
                "non-loopback development binding requires --allow-development-network"
            )
    if mode is DeploymentMode.SHARED:
        settings.validate_shared_mode()
    if settings.require_https and not ssl_certfile and not settings.trusted_proxy_networks:
        raise ValueError(
            "HTTPS enforcement requires direct TLS files or a configured trusted TLS proxy"
        )


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit("PyMRST Studio requires `pip install pymrst[app]`") from error
    settings = ServerDeploymentSettings.from_environment()
    context = create_application(arguments.workspace)
    try:
        validate_launch(
            host=arguments.host, mode=context.security.mode, settings=settings,
            ssl_certfile=arguments.ssl_certfile or "",
            ssl_keyfile=arguments.ssl_keyfile or "",
            allow_development_network=arguments.allow_development_network,
        )
        from .api.application import create_fastapi_app
        application = create_fastapi_app(context, settings)
        uvicorn.run(
            application, host=arguments.host, port=arguments.port, workers=1,
            proxy_headers=bool(settings.trusted_proxy_networks),
            forwarded_allow_ips=",".join(settings.trusted_proxy_networks),
            ssl_certfile=arguments.ssl_certfile, ssl_keyfile=arguments.ssl_keyfile,
            server_header=False,
        )
    finally:
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parser", "validate_launch"]
