"""Synchronous event delivery with explicit handler-failure policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Literal, Protocol, TextIO

from pymrst.research_studio.evidence.events import SimulationEvent


class EventHandler(Protocol):
    def __call__(self, event: SimulationEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class HandlerFailure:
    handler: str
    event_sequence: int
    error_type: str
    message: str


class EventDeliveryError(RuntimeError):
    pass


class EventBus:
    """Deliver immutable events in registration order in the caller thread."""

    def __init__(self, *, error_policy: Literal["isolate", "raise"] = "isolate"):
        if error_policy not in {"isolate", "raise"}:
            raise ValueError("error_policy must be 'isolate' or 'raise'")
        self.error_policy = error_policy
        self._handlers: list[EventHandler] = []
        self.failures: list[HandlerFailure] = []

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)
        return unsubscribe

    def publish(self, event: SimulationEvent) -> None:
        for handler in tuple(self._handlers):
            try:
                handler(event)
            except Exception as error:
                failure = HandlerFailure(
                    handler=getattr(handler, "__qualname__", type(handler).__name__),
                    event_sequence=event.sequence,
                    error_type=type(error).__name__,
                    message=str(error),
                )
                self.failures.append(failure)
                if self.error_policy == "raise":
                    raise EventDeliveryError(
                        f"event handler {failure.handler} failed at sequence "
                        f"{failure.event_sequence}: {failure.message}"
                    ) from error


class JsonLinesHandler:
    """Append one durable, flush-on-event JSON record to a path or stream."""

    def __init__(self, target: str | Path | TextIO):
        self._owns_stream = not hasattr(target, "write")
        self._stream = (Path(target).open("a", encoding="utf-8")
                        if self._owns_stream else target)

    def __call__(self, event: SimulationEvent) -> None:
        self._stream.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if self._owns_stream and not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "JsonLinesHandler":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


__all__ = [
    "EventBus", "EventDeliveryError", "EventHandler", "HandlerFailure",
    "JsonLinesHandler",
]
