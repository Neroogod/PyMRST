"""Structured-output adapters that may propose—but never execute—tool plans."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..base import ToolCall
from ..narrative import NarrativeProposal
from ..orchestrator import InvestigationPlan
from .base import JsonTransport, ProviderConfiguration, ProviderKind, UrllibJsonTransport
from .metrics import ProviderCallMetrics, ollama_metrics, openai_metrics


_PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "calls": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "call_id": {"type": "string"},
                    "tool": {"type": "string"},
                    "arguments_json": {"type": "string"},
                },
                "required": ["call_id", "tool", "arguments_json"],
            },
        },
    }, "required": ["calls"],
}

_NARRATIVE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "integer", "const": 1},
        "title": {"type": "string", "minLength": 1, "maxLength": 200},
        "executive_summary": {"type": "string", "minLength": 1, "maxLength": 4000},
        "sections": {
            "type": "array", "minItems": 1, "maxItems": 24,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "section_id": {"type": "string", "minLength": 1, "maxLength": 80},
                    "heading": {"type": "string", "minLength": 1, "maxLength": 160},
                    "narrative": {"type": "string", "minLength": 1, "maxLength": 8000},
                    "finding_ids": {
                        "type": "array", "minItems": 1, "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": ["section_id", "heading", "narrative", "finding_ids"],
            },
        },
    },
    "required": ["schema_version", "title", "executive_summary", "sections"],
}


def _contract(response_type):
    if response_type is InvestigationPlan:
        return "pymrst_investigation_plan", _PLAN_SCHEMA, (
            "Return a bounded plan only. Choose exclusively from tools. "
            "arguments_json must be a JSON object string. Do not calculate results, "
            "request files, start workloads, or propose mutations."
        )
    if response_type is NarrativeProposal:
        return "pymrst_evidence_narrative", _NARRATIVE_SCHEMA, (
            "Return an evidence-bound narrative only. Use exclusively the supplied "
            "accepted finding statements. Every section must identify its finding_ids. "
            "Do not add facts, numbers, URLs, recommendations, or evidence references."
        )
    raise TypeError(f"unsupported reasoning-provider response type: {response_type!r}")


def _prompt(messages, tools, response_type):
    _, _, instruction = _contract(response_type)
    return json.dumps({
        "contract": instruction,
        "messages": list(messages), "tools": list(tools),
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decode_plan(value, response_type):
    if response_type is NarrativeProposal:
        if not isinstance(value, dict):
            raise RuntimeError("reasoning provider narrative must be an object")
        try:
            return NarrativeProposal.from_dict(value)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("reasoning provider emitted an invalid narrative") from error
    if response_type is not InvestigationPlan:
        raise TypeError("unsupported reasoning-provider response type")
    if not isinstance(value, dict) or not isinstance(value.get("calls"), list):
        raise RuntimeError("reasoning provider omitted the structured calls array")
    calls = []
    for item in value["calls"]:
        if not isinstance(item, dict):
            raise RuntimeError("reasoning provider emitted an invalid plan item")
        try:
            arguments = json.loads(item["arguments_json"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("reasoning provider emitted invalid arguments_json") from error
        if not isinstance(arguments, dict):
            raise RuntimeError("planned tool arguments must decode to an object")
        calls.append(ToolCall(item["call_id"], item["tool"], arguments))
    return InvestigationPlan(tuple(calls))


def _openai_output_text(response):
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    for item in response.get("output", ()):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", ()):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    return text
    raise RuntimeError("OpenAI response contained no structured output text")


class OpenAIResponsesPlanningProvider:
    def __init__(self, configuration, *, api_key: str, transport: JsonTransport | None = None):
        if configuration.kind is not ProviderKind.OPENAI_RESPONSES:
            raise ValueError("OpenAI Responses adapter requires openai_responses configuration")
        self.configuration = configuration
        self.api_key = str(api_key).strip()
        if not self.api_key:
            raise ValueError("OpenAI API key is unavailable")
        self.transport = transport or UrllibJsonTransport()
        self._last_metrics = ProviderCallMetrics()

    def take_metrics(self):
        value, self._last_metrics = self._last_metrics, ProviderCallMetrics()
        return value

    def complete(self, messages, tools, response_type):
        config = self.configuration
        schema_name, schema, _ = _contract(response_type)
        response = self.transport.post_json(
            config.endpoint.rstrip("/") + "/responses",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload={
                "model": config.model, "store": False,
                "max_output_tokens": config.maximum_output_tokens,
                "input": _prompt(messages, tools, response_type),
                "text": {"format": {
                    "type": "json_schema", "name": schema_name,
                    "strict": True, "schema": schema,
                }},
            },
            timeout_seconds=config.timeout_seconds,
            maximum_response_bytes=config.maximum_response_bytes,
        )
        self._last_metrics = openai_metrics(response)
        try:
            value = json.loads(_openai_output_text(response))
        except json.JSONDecodeError as error:
            raise RuntimeError("OpenAI structured plan was not valid JSON") from error
        return _decode_plan(value, response_type)


class OpenAICompatiblePlanningProvider:
    """Loopback-only adapter for vLLM and compatible local servers."""

    def __init__(self, configuration, *, api_key="", transport: JsonTransport | None = None):
        if configuration.kind is not ProviderKind.OPENAI_COMPATIBLE:
            raise ValueError("compatible adapter requires openai_compatible configuration")
        self.configuration, self.api_key = configuration, str(api_key).strip()
        self.transport = transport or UrllibJsonTransport()
        self._last_metrics = ProviderCallMetrics()

    def take_metrics(self):
        value, self._last_metrics = self._last_metrics, ProviderCallMetrics()
        return value

    def complete(self, messages, tools, response_type):
        config = self.configuration
        schema_name, schema, _ = _contract(response_type)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = self.transport.post_json(
            config.endpoint.rstrip("/") + "/chat/completions", headers=headers,
            payload={
                "model": config.model,
                "max_tokens": config.maximum_output_tokens,
                "messages": [{"role": "user", "content": _prompt(messages, tools, response_type)}],
                "temperature": 0,
                "response_format": {
                    "type": "json_schema", "json_schema": {
                        "name": schema_name, "strict": True,
                        "schema": schema,
                    },
                },
            }, timeout_seconds=config.timeout_seconds,
            maximum_response_bytes=config.maximum_response_bytes,
        )
        self._last_metrics = openai_metrics(response)
        try:
            content = response["choices"][0]["message"]["content"]
            value = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("compatible provider returned no valid structured plan") from error
        return _decode_plan(value, response_type)


class OllamaPlanningProvider:
    def __init__(self, configuration, *, transport: JsonTransport | None = None):
        if configuration.kind is not ProviderKind.OLLAMA:
            raise ValueError("Ollama adapter requires ollama configuration")
        self.configuration, self.transport = configuration, transport or UrllibJsonTransport()
        self._last_metrics = ProviderCallMetrics()

    def take_metrics(self):
        value, self._last_metrics = self._last_metrics, ProviderCallMetrics()
        return value

    def complete(self, messages, tools, response_type):
        config = self.configuration
        _, schema, _ = _contract(response_type)
        response = self.transport.post_json(
            config.endpoint.rstrip("/") + "/api/chat", headers={},
            payload={
                "model": config.model, "stream": False, "format": schema,
                "messages": [{"role": "user", "content": _prompt(messages, tools, response_type)}],
                "options": {"temperature": 0, "num_predict": config.maximum_output_tokens},
            }, timeout_seconds=config.timeout_seconds,
            maximum_response_bytes=config.maximum_response_bytes,
        )
        self._last_metrics = ollama_metrics(response)
        try:
            value = json.loads(response["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("Ollama returned no valid structured plan") from error
        return _decode_plan(value, response_type)


__all__ = [
    "OllamaPlanningProvider", "OpenAICompatiblePlanningProvider",
    "OpenAIResponsesPlanningProvider",
]
