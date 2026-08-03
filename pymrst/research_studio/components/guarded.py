"""M4.3 policy-enforced surrogate execution with complete telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import time
from typing import Callable, Mapping

import numpy as np

from .surrogate import SurrogateContract, SurrogatePrediction


@dataclass(frozen=True, slots=True)
class GuardEvent:
    call_index: int
    batch_size: int
    used_surrogate: bool
    fallback_reason: str
    ood_reasons: tuple[str, ...]
    invariant_failures: tuple[str, ...]
    reference_samples: int
    maximum_reference_error: float
    estimated_accumulated_error: float
    elapsed_seconds: float
    device: str


class GuardedSurrogateRuntime:
    def __init__(self, contract: SurrogateContract, predictor: Callable, *,
                 fallback: Callable | None = None, device="cpu",
                 uncertainty_limit=np.inf):
        if device not in contract.devices:
            raise ValueError(f"surrogate {contract.surrogate_id} does not support {device}")
        self.contract = contract
        self.predictor = predictor
        self.fallback = fallback or self._resolve(contract.physics_fallback)
        self.device = device
        self.uncertainty_limit = float(uncertainty_limit)
        self.events: list[GuardEvent] = []
        self._accumulated_error = 0.0
        self._rng = np.random.Generator(
            np.random.PCG64(contract.reference_sampling.seed))

    @staticmethod
    def _resolve(owner):
        module, symbol = owner.split(":", 1)
        return getattr(importlib.import_module(module), symbol)

    def __call__(self, inputs: Mapping[str, np.ndarray]):
        started = time.perf_counter()
        normalized, ood = self.contract.validate_inputs(inputs)
        batch = next(iter(normalized.values())).shape[0]
        prediction = None
        reason = ""
        invariant_failures = ()
        if ood:
            reason = "out_of_domain"
        else:
            prediction = self.predictor(normalized, device=self.device)
            if not isinstance(prediction, SurrogatePrediction):
                raise TypeError("predictor must return SurrogatePrediction")
            outputs = self.contract.validate_outputs(prediction.outputs)
            invariant_failures = tuple(prediction.invariant_failures)
            if invariant_failures:
                reason = "invariant_failure"
            elif prediction.uncertainty is not None and np.any(
                    np.asarray(prediction.uncertainty) > self.uncertainty_limit):
                reason = "uncertainty_limit"
        used_surrogate = not reason
        outputs = (self.contract.validate_outputs(prediction.outputs)
                   if used_surrogate else self.contract.validate_outputs(
                       self.fallback(normalized)))
        policy = self.contract.reference_sampling
        requested = max(policy.minimum_per_batch,
                        int(np.ceil(policy.fraction*batch)))
        sample_count = min(batch, requested)
        maximum_error = 0.0
        if used_surrogate and sample_count:
            indices = np.sort(self._rng.choice(batch, size=sample_count, replace=False))
            reference = self.contract.validate_outputs(
                self.fallback({name: value[indices] for name, value in normalized.items()}))
            maximum_error = max(float(np.max(np.abs(
                outputs[name][indices]-reference[name]))) for name in outputs)
            self._accumulated_error += maximum_error
        event = GuardEvent(
            len(self.events), batch, used_surrogate, reason, tuple(ood),
            invariant_failures, sample_count if used_surrogate else batch,
            maximum_error, self._accumulated_error,
            time.perf_counter()-started, self.device)
        self.events.append(event)
        return outputs

    def report(self):
        surrogate = sum(event.used_surrogate for event in self.events)
        fallback = len(self.events)-surrogate
        return {"surrogate_id": self.contract.surrogate_id,
                "calls": len(self.events), "surrogate_calls": surrogate,
                "fallback_calls": fallback,
                "ood_events": sum(bool(event.ood_reasons) for event in self.events),
                "invariant_failures": sum(bool(event.invariant_failures)
                                          for event in self.events),
                "estimated_accumulated_error": self._accumulated_error,
                "device": self.device,
                "events": [event.__dict__ if hasattr(event, "__dict__") else {
                    name: getattr(event, name) for name in event.__slots__}
                    for event in self.events]}


__all__ = ["GuardEvent", "GuardedSurrogateRuntime"]
