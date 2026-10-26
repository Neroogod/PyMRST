from __future__ import annotations

import asyncio

import pytest

from pymrst.app.api.middleware import (
    AuthenticationFailureLimiter, RequestSizeLimitMiddleware,
    SecureTransportMiddleware, SecurityHeadersMiddleware,
)
from pymrst.app.schemas.deployment import DeploymentEnvironment, ServerDeploymentSettings
from pymrst.app.schemas.security import DeploymentMode
from pymrst.app.server import validate_launch


def _run_asgi(application, *, body_chunks=(b"",), headers=(), scheme="http", peer="127.0.0.1"):
    messages = [
        {"type": "http.request", "body": body,
         "more_body": index < len(body_chunks) - 1}
        for index, body in enumerate(body_chunks)
    ]
    sent = []

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "method": "POST", "path": "/probe", "scheme": scheme,
        "headers": list(headers), "client": (peer, 41000), "server": ("test", 80),
    }
    asyncio.run(application(scope, receive, send))
    return sent


async def _drain_app(scope, receive, send):
    while True:
        message = await receive()
        if message["type"] == "http.disconnect" or not message.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": 204, "headers": []})
    await send({"type": "http.response.body", "body": b""})


def test_environment_parser_is_strict_and_normalizes_networks():
    settings = ServerDeploymentSettings.from_environment({
        "PYMRST_APP_ENVIRONMENT": "production",
        "PYMRST_APP_ALLOWED_HOSTS": "studio.example.test",
        "PYMRST_APP_ALLOWED_ORIGINS": "https://studio.example.test/",
        "PYMRST_APP_TRUSTED_PROXIES": "10.20.1.4/24",
        "PYMRST_APP_REQUIRE_HTTPS": "true",
        "PYMRST_APP_MAXIMUM_REQUEST_BYTES": "4096",
        "PYMRST_APP_AUTH_FAILURES_PER_MINUTE": "7",
        "PYMRST_APP_HSTS_SECONDS": "86400",
        "PYMRST_APP_EXPOSE_DOCS": "false",
    })
    assert settings.environment is DeploymentEnvironment.PRODUCTION
    assert settings.allowed_origins == ("https://studio.example.test",)
    assert settings.trusted_proxy_networks == ("10.20.1.0/24",)
    settings.validate_shared_mode()
    with pytest.raises(ValueError, match="true or false"):
        ServerDeploymentSettings.from_environment({"PYMRST_APP_REQUIRE_HTTPS": "perhaps"})


def test_shared_production_rejects_wildcards_insecure_transport_and_docs():
    with pytest.raises(ValueError, match="wildcard allowed hosts"):
        ServerDeploymentSettings(allowed_hosts=("*",)).validate_shared_mode()
    with pytest.raises(ValueError, match="requires HTTPS"):
        ServerDeploymentSettings(environment="production").validate_shared_mode()
    with pytest.raises(ValueError, match="disable interactive"):
        ServerDeploymentSettings(
            environment="production", require_https=True,
        ).validate_shared_mode()
    desktop = ServerDeploymentSettings(allowed_origins=("tauri://localhost",))
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        desktop.validate_shared_mode()


def test_forwarded_metadata_is_trusted_only_from_configured_network():
    settings = ServerDeploymentSettings(trusted_proxy_networks=("10.0.0.0/8",))
    assert settings.request_is_secure(
        peer_address="10.2.3.4", scheme="http", forwarded_proto="https",
    )
    assert not settings.request_is_secure(
        peer_address="192.0.2.4", scheme="http", forwarded_proto="https",
    )
    assert settings.client_address(
        peer_address="10.2.3.4", forwarded_for="203.0.113.8, 10.2.3.4",
    ) == "203.0.113.8"
    assert settings.client_address(
        peer_address="192.0.2.4", forwarded_for="203.0.113.8",
    ) == "192.0.2.4"


def test_request_limit_rejects_declared_and_streamed_overflow():
    limited = RequestSizeLimitMiddleware(_drain_app, 5)
    declared = _run_asgi(limited, headers=((b"content-length", b"6"),))
    assert declared[0]["status"] == 413
    streamed = _run_asgi(limited, body_chunks=(b"123", b"456"))
    assert streamed[0]["status"] == 413
    accepted = _run_asgi(limited, body_chunks=(b"12", b"345"))
    assert accepted[0]["status"] == 204


def test_https_middleware_rejects_spoofed_forwarding_and_accepts_trusted_proxy():
    settings = ServerDeploymentSettings(
        trusted_proxy_networks=("10.0.0.0/8",), require_https=True,
    )
    guarded = SecureTransportMiddleware(_drain_app, settings)
    rejected = _run_asgi(
        guarded, headers=((b"x-forwarded-proto", b"https"),), peer="192.0.2.1",
    )
    assert rejected[0]["status"] == 400
    accepted = _run_asgi(
        guarded, headers=((b"x-forwarded-proto", b"https"),), peer="10.1.2.3",
    )
    assert accepted[0]["status"] == 204


def test_security_headers_cover_successful_responses():
    protected = SecurityHeadersMiddleware(
        _drain_app, hsts_seconds=3600, documentation_enabled=False,
    )
    sent = _run_asgi(protected, scheme="https")
    headers = dict(sent[0]["headers"])
    assert headers[b"x-content-type-options"] == b"nosniff"
    assert headers[b"x-frame-options"] == b"DENY"
    assert headers[b"cache-control"] == b"no-store"
    assert headers[b"strict-transport-security"] == b"max-age=3600"
    assert b"frame-ancestors 'none'" in headers[b"content-security-policy"]


def test_authentication_failure_limiter_expires_window_and_resets():
    current = [100.0]
    limiter = AuthenticationFailureLimiter(2, clock=lambda: current[0])
    assert limiter.allow("peer")
    limiter.record_failure("peer"); limiter.record_failure("peer")
    assert not limiter.allow("peer")
    current[0] += 61
    assert limiter.allow("peer")
    limiter.record_failure("peer"); limiter.reset("peer")
    assert limiter.allow("peer")


def test_launcher_refuses_unsafe_network_modes_and_incomplete_tls():
    development = ServerDeploymentSettings()
    with pytest.raises(ValueError, match="requires shared"):
        validate_launch(host="0.0.0.0", mode=DeploymentMode.LOCAL, settings=development)
    with pytest.raises(ValueError, match="allow-development-network"):
        validate_launch(host="0.0.0.0", mode=DeploymentMode.SHARED, settings=development)
    with pytest.raises(ValueError, match="supplied together"):
        validate_launch(
            host="127.0.0.1", mode=DeploymentMode.LOCAL, settings=development,
            ssl_certfile="certificate.pem",
        )
    production = ServerDeploymentSettings(
        environment="production", allowed_hosts=("studio.example.test",),
        allowed_origins=("https://studio.example.test",), require_https=True,
        expose_api_documentation=False,
    )
    with pytest.raises(ValueError, match="direct TLS files"):
        validate_launch(host="0.0.0.0", mode=DeploymentMode.SHARED, settings=production)
