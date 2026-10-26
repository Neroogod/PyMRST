"""Provider-call telemetry normalized without retaining prompts or responses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderCallMetrics:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self):
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None:
                value = int(value)
                if value < 0:
                    raise ValueError("provider token counts cannot be negative")
                object.__setattr__(self, name, value)

    def to_dict(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


def openai_metrics(response):
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    return ProviderCallMetrics(
        usage.get("input_tokens", usage.get("prompt_tokens")),
        usage.get("output_tokens", usage.get("completion_tokens")),
        usage.get("total_tokens"),
    )


def ollama_metrics(response):
    if not isinstance(response, dict):
        return ProviderCallMetrics()
    input_tokens = response.get("prompt_eval_count")
    output_tokens = response.get("eval_count")
    total = None if input_tokens is None or output_tokens is None else int(input_tokens) + int(output_tokens)
    return ProviderCallMetrics(input_tokens, output_tokens, total)


__all__ = ["ProviderCallMetrics", "ollama_metrics", "openai_metrics"]
