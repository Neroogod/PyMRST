"""Dependency-light ASGI production safeguards."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from typing import Callable

from ..schemas.deployment import ServerDeploymentSettings


def _json_response(status: int, detail: str):
    body = json.dumps({"detail": detail}, separators=(",", ":")).encode()

    async def response(scope, receive, send):
        await send({
            "type": "http.response.start", "status": status,
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})
    return response


class RequestSizeLimitMiddleware:
    """Reject oversized fixed-length and chunked HTTP bodies while streaming."""

    def __init__(self, app, maximum_bytes: int):
        self.app, self.maximum_bytes = app, int(maximum_bytes)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {key.lower(): value for key, value in scope.get("headers", ())}
        length = headers.get(b"content-length", b"")
        try:
            declared = int(length) if length else None
        except ValueError:
            return await _json_response(400, "invalid Content-Length")(scope, receive, send)
        if declared is not None and (declared < 0 or declared > self.maximum_bytes):
            return await _json_response(413, "request body exceeds configured limit")(
                scope, receive, send,
            )
        consumed, exceeded = 0, False

        async def limited_receive():
            nonlocal consumed, exceeded
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.maximum_bytes:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        response_started = False

        async def guarded_send(message):
            nonlocal response_started
            if exceeded:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded and not response_started:
            await _json_response(413, "request body exceeds configured limit")(
                scope, receive, send,
            )


class SecureTransportMiddleware:
    def __init__(self, app, settings: ServerDeploymentSettings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.settings.require_https:
            return await self.app(scope, receive, send)
        headers = {key.lower(): value.decode("latin-1")
                   for key, value in scope.get("headers", ())}
        client = scope.get("client") or ("", 0)
        if not self.settings.request_is_secure(
            peer_address=client[0], scheme=scope.get("scheme", "http"),
            forwarded_proto=headers.get(b"x-forwarded-proto", ""),
        ):
            return await _json_response(400, "HTTPS is required")(
                scope, receive, send,
            )
        return await self.app(scope, receive, send)


class SecurityHeadersMiddleware:
    def __init__(
        self, app, *, hsts_seconds: int, documentation_enabled: bool = False,
    ):
        self.app, self.hsts_seconds = app, int(hsts_seconds)
        self.documentation_enabled = bool(documentation_enabled)

    async def __call__(self, scope, receive, send):
        async def secure_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", ()))
                policy = (
                    b"default-src 'self' https://cdn.jsdelivr.net; "
                    b"style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                    b"script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                    b"img-src 'self' data: https://fastapi.tiangolo.com; "
                    b"frame-ancestors 'none'; base-uri 'none'"
                    if self.documentation_enabled else
                    b"default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
                )
                headers.extend((
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    (b"content-security-policy", policy),
                    (b"cache-control", b"no-store"),
                ))
                if scope.get("scheme") == "https" and self.hsts_seconds:
                    headers.append((
                        b"strict-transport-security",
                        f"max-age={self.hsts_seconds}".encode(),
                    ))
                message = {**message, "headers": headers}
            await send(message)
        await self.app(scope, receive, secure_send)


class AuthenticationFailureLimiter:
    """Bound authentication attempts per peer; suitable for one workspace process."""

    def __init__(self, maximum_per_minute: int, *, clock: Callable[[], float] = time.monotonic):
        self.maximum = int(maximum_per_minute)
        self.clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, peer: str) -> bool:
        now = self.clock()
        cutoff = now - 60.0
        with self._lock:
            events = self._events[str(peer)]
            while events and events[0] <= cutoff:
                events.popleft()
            return len(events) < self.maximum

    def record_failure(self, peer: str) -> None:
        with self._lock:
            self._events[str(peer)].append(self.clock())

    def reset(self, peer: str) -> None:
        with self._lock:
            self._events.pop(str(peer), None)


__all__ = [
    "AuthenticationFailureLimiter", "RequestSizeLimitMiddleware",
    "SecureTransportMiddleware", "SecurityHeadersMiddleware",
]
