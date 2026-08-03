"""Native reentrant port of ADFNE15 ``Ticot.m``."""
from contextlib import contextmanager
from dataclasses import dataclass,field
from time import perf_counter
from .convert import clock
@dataclass(slots=True)
class Ticot:
    stack:list=field(default_factory=list); history:list=field(default_factory=list)
    def start(self,message): self.stack.append((str(message),perf_counter()))
    def stop(self):
        if not self.stack: raise RuntimeError("no active timer")
        message,started=self.stack.pop(); elapsed=perf_counter()-started
        record=(('@' if self.stack else '')+message,clock(elapsed)); self.history.append(record); return record
    @contextmanager
    def measure(self,message):
        self.start(message)
        try: yield self
        finally: self.stop()
