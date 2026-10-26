"""Validated server-deployment settings for local and shared PyMRST Studio."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import ipaddress
import os
from typing import Mapping
from urllib.parse import urlparse


class DeploymentEnvironment(StrEnum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _boolean(value: str, name: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


@dataclass(frozen=True, slots=True)
class ServerDeploymentSettings:
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "testserver")
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173", "http://localhost:5173",
    )
    trusted_proxy_networks: tuple[str, ...] = ()
    require_https: bool = False
    maximum_request_bytes: int = 16 * 1024 * 1024
    authentication_failures_per_minute: int = 30
    hsts_seconds: int = 31_536_000
    expose_api_documentation: bool = True

    def __post_init__(self):
        object.__setattr__(self, "environment", DeploymentEnvironment(self.environment))
        hosts = tuple(str(item).strip().casefold() for item in self.allowed_hosts)
        if not hosts or any(not item for item in hosts) or len(hosts) != len(set(hosts)):
            raise ValueError("allowed hosts must be non-empty and unique")
        object.__setattr__(self, "allowed_hosts", hosts)
        origins = tuple(str(item).strip().rstrip("/") for item in self.allowed_origins)
        if len(origins) != len(set(origins)):
            raise ValueError("allowed origins must be unique")
        for origin in origins:
            parsed = urlparse(origin)
            if origin != "*" and (parsed.scheme not in {"http", "https", "tauri"} or not parsed.netloc):
                raise ValueError(f"invalid allowed origin: {origin}")
        object.__setattr__(self, "allowed_origins", origins)
        networks = []
        for value in self.trusted_proxy_networks:
            networks.append(str(ipaddress.ip_network(str(value).strip(), strict=False)))
        if len(networks) != len(set(networks)):
            raise ValueError("trusted proxy networks must be unique")
        object.__setattr__(self, "trusted_proxy_networks", tuple(networks))
        for name, minimum, maximum in (
            ("maximum_request_bytes", 1024, 10 * 1024**3),
            ("authentication_failures_per_minute", 1, 100_000),
            ("hsts_seconds", 0, 10 * 365 * 24 * 3600),
        ):
            value = int(getattr(self, name))
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
            object.__setattr__(self, name, value)

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None,
    ) -> "ServerDeploymentSettings":
        values = os.environ if environment is None else environment
        defaults = cls()
        return cls(
            environment=values.get("PYMRST_APP_ENVIRONMENT", defaults.environment.value),
            allowed_hosts=_csv(values.get(
                "PYMRST_APP_ALLOWED_HOSTS", ",".join(defaults.allowed_hosts),
            )),
            allowed_origins=_csv(values.get(
                "PYMRST_APP_ALLOWED_ORIGINS", ",".join(defaults.allowed_origins),
            )),
            trusted_proxy_networks=_csv(values.get("PYMRST_APP_TRUSTED_PROXIES", "")),
            require_https=_boolean(values.get(
                "PYMRST_APP_REQUIRE_HTTPS", str(defaults.require_https),
            ), "PYMRST_APP_REQUIRE_HTTPS"),
            maximum_request_bytes=int(values.get(
                "PYMRST_APP_MAXIMUM_REQUEST_BYTES", defaults.maximum_request_bytes,
            )),
            authentication_failures_per_minute=int(values.get(
                "PYMRST_APP_AUTH_FAILURES_PER_MINUTE",
                defaults.authentication_failures_per_minute,
            )),
            hsts_seconds=int(values.get("PYMRST_APP_HSTS_SECONDS", defaults.hsts_seconds)),
            expose_api_documentation=_boolean(values.get(
                "PYMRST_APP_EXPOSE_DOCS", str(defaults.expose_api_documentation),
            ), "PYMRST_APP_EXPOSE_DOCS"),
        )

    def proxy_is_trusted(self, address: str) -> bool:
        try:
            candidate = ipaddress.ip_address(str(address).strip())
        except ValueError:
            return False
        return any(candidate in ipaddress.ip_network(network)
                   for network in self.trusted_proxy_networks)

    def request_is_secure(
        self, *, peer_address: str, scheme: str, forwarded_proto: str = "",
    ) -> bool:
        if str(scheme).casefold() == "https":
            return True
        if self.proxy_is_trusted(peer_address):
            first = str(forwarded_proto).split(",", 1)[0].strip().casefold()
            return first == "https"
        return False

    def client_address(self, *, peer_address: str, forwarded_for: str = "") -> str:
        """Honor forwarding metadata only across an explicitly trusted hop."""
        peer = str(peer_address).strip() or "unknown"
        if not self.proxy_is_trusted(peer):
            return peer
        candidate = str(forwarded_for).split(",", 1)[0].strip()
        try:
            return str(ipaddress.ip_address(candidate))
        except ValueError:
            return peer

    def validate_shared_mode(self) -> None:
        if "*" in self.allowed_hosts:
            raise ValueError("shared mode forbids wildcard allowed hosts")
        if "*" in self.allowed_origins:
            raise ValueError("shared mode forbids wildcard allowed origins")
        if any(urlparse(origin).scheme not in {"http", "https"}
               for origin in self.allowed_origins):
            raise ValueError("shared mode permits only HTTP(S) browser origins")
        if self.environment is DeploymentEnvironment.PRODUCTION:
            if not self.require_https:
                raise ValueError("production shared mode requires HTTPS enforcement")
            if self.expose_api_documentation:
                raise ValueError("production shared mode must disable interactive API documentation")

    def to_dict(self) -> dict[str, object]:
        return {
            "environment": self.environment.value,
            "allowed_hosts": list(self.allowed_hosts),
            "allowed_origins": list(self.allowed_origins),
            "trusted_proxy_networks": list(self.trusted_proxy_networks),
            "require_https": self.require_https,
            "maximum_request_bytes": self.maximum_request_bytes,
            "authentication_failures_per_minute": self.authentication_failures_per_minute,
            "hsts_seconds": self.hsts_seconds,
            "expose_api_documentation": self.expose_api_documentation,
        }


__all__ = ["DeploymentEnvironment", "ServerDeploymentSettings"]
