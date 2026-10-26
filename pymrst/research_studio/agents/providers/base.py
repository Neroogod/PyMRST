"""Configuration and transport boundaries for optional reasoning providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import json
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .._serialization import frozen_json_mapping


class ProviderKind(StrEnum):
    OPENAI_RESPONSES = "openai_responses"
    OPENAI_COMPATIBLE = "openai_compatible"
    OLLAMA = "ollama"


_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True, slots=True)
class ProviderConfiguration:
    provider_id: str
    kind: ProviderKind
    model: str
    endpoint: str
    credential_env: str = ""
    timeout_seconds: float = 30.0
    maximum_output_tokens: int = 2_048
    maximum_response_bytes: int = 1_000_000
    enabled: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("provider_id", "model", "endpoint"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "kind", ProviderKind(self.kind))
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("provider endpoint must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("provider endpoints cannot contain credentials, query, or fragment")
        if self.kind is ProviderKind.OPENAI_RESPONSES:
            if parsed.scheme != "https" or parsed.hostname.casefold() != "api.openai.com":
                raise ValueError("OpenAI Responses providers are restricted to https://api.openai.com")
            if parsed.port not in {None, 443} or parsed.path.rstrip("/") not in {"", "/v1"}:
                raise ValueError("OpenAI Responses endpoint must be https://api.openai.com/v1")
            if not self.credential_env:
                object.__setattr__(self, "credential_env", "OPENAI_API_KEY")
        elif parsed.hostname.casefold() not in _LOOPBACK:
            raise ValueError("local reasoning-provider endpoints must use a loopback host")
        credential = str(self.credential_env).strip()
        if credential and not credential.replace("_", "").isalnum():
            raise ValueError("credential_env must be an environment-variable name")
        object.__setattr__(self, "credential_env", credential)
        timeout = float(self.timeout_seconds)
        output_tokens = int(self.maximum_output_tokens)
        maximum = int(self.maximum_response_bytes)
        if (not 0 < timeout <= 300 or not 128 <= output_tokens <= 16_384
                or not 1_024 <= maximum <= 10_000_000):
            raise ValueError("provider timeout, output-token, or response-size limit is outside safe bounds")
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "maximum_output_tokens", output_tokens)
        object.__setattr__(self, "maximum_response_bytes", maximum)
        object.__setattr__(self, "options", frozen_json_mapping(self.options, "$.options"))

    def to_dict(self):
        return {
            "provider_id": self.provider_id, "kind": self.kind.value,
            "model": self.model, "endpoint": self.endpoint.rstrip("/"),
            "credential_env": self.credential_env,
            "timeout_seconds": self.timeout_seconds,
            "maximum_output_tokens": self.maximum_output_tokens,
            "maximum_response_bytes": self.maximum_response_bytes,
            "enabled": bool(self.enabled), "options": dict(self.options),
        }


class JsonTransport(Protocol):
    def post_json(
        self, url: str, *, headers: Mapping[str, str], payload: Mapping[str, Any],
        timeout_seconds: float, maximum_response_bytes: int,
    ) -> Mapping[str, Any]: ...


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("reasoning-provider redirects are disabled")


class UrllibJsonTransport:
    """Small dependency-free POST transport with redirect and size controls."""

    def post_json(self, url, *, headers, payload, timeout_seconds, maximum_response_bytes):
        request = Request(
            url, data=json.dumps(payload, allow_nan=False, separators=(",", ":")).encode(),
            headers={"Content-Type": "application/json", **dict(headers)}, method="POST",
        )
        try:
            with build_opener(_NoRedirects()).open(request, timeout=timeout_seconds) as response:
                body = response.read(maximum_response_bytes + 1)
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(f"reasoning provider request failed: {error}") from error
        if len(body) > maximum_response_bytes:
            raise RuntimeError("reasoning provider response exceeded configured byte limit")
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("reasoning provider returned invalid JSON") from error
        if not isinstance(value, dict):
            raise RuntimeError("reasoning provider response must be a JSON object")
        return value


__all__ = [
    "JsonTransport", "ProviderConfiguration", "ProviderKind", "UrllibJsonTransport",
]
