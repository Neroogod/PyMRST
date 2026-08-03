"""Cooperative cancellation contracts with no asynchronous state mutation."""

from __future__ import annotations

from threading import Event


class CancellationRequested(RuntimeError):
    pass


class CancellationToken:
    """Thread-safe flag checked only at lifecycle-safe boundaries."""

    def __init__(self) -> None:
        self._event = Event()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def request(self) -> None:
        self._event.set()

    def raise_if_requested(self) -> None:
        if self.requested:
            raise CancellationRequested("simulation cancellation requested")


__all__ = ["CancellationRequested", "CancellationToken"]
