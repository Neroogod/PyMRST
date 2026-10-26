"""Acceptance tests for credential-safe, independently bounded model adapters."""

import json

import pytest

from pymrst.research_studio.agents.base import ToolDescriptor
from pymrst.research_studio.agents.context import InvestigationContext
from pymrst.research_studio.agents.permissions import AgentPermission, PermissionGrant
from pymrst.research_studio.agents.planning import BoundedInvestigationPlanner
from pymrst.research_studio.agents.narrative import NarrativeProposal
from pymrst.research_studio.agents.providers import (
    OllamaPlanningProvider, OpenAIResponsesPlanningProvider,
    ProviderConfiguration,
)


class CapturingTransport:
    def __init__(self, response):
        self.response, self.calls = response, []

    def post_json(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def context():
    return InvestigationContext(
        "investigation_provider", "Inspect the selected run", "case_provider",
        "a" * 64, PermissionGrant(AgentPermission.ANALYZE), ("run_selected",),
    )


def descriptor():
    return ToolDescriptor(
        "analyze_conservation", "Inspect conservation evidence",
        AgentPermission.ANALYZE,
    )


def encoded_plan(arguments):
    return json.dumps({"calls": [{
        "call_id": "provider_call", "tool": "analyze_conservation",
        "arguments_json": json.dumps(arguments),
    }]})


def test_openai_responses_adapter_uses_nonstored_structured_output_and_bounded_context():
    transport = CapturingTransport({
        "output_text": encoded_plan({"run_id": "run_selected"}),
        "usage": {"input_tokens": 40, "output_tokens": 12, "total_tokens": 52},
    })
    config = ProviderConfiguration(
        "openai_planner", "openai_responses", "configured-model",
        "https://api.openai.com/v1", "PYMRST_TEST_OPENAI_KEY",
    )
    provider = OpenAIResponsesPlanningProvider(
        config, api_key="not-persisted", transport=transport,
    )
    planned = BoundedInvestigationPlanner(
        provider, provider_name=config.provider_id,
    ).plan(context(), (descriptor(),))
    assert planned.plan.calls[0].arguments == {"run_id": "run_selected"}
    url, request = transport.calls[0]
    assert url == "https://api.openai.com/v1/responses"
    assert request["payload"]["store"] is False
    assert request["payload"]["max_output_tokens"] == 2_048
    assert request["payload"]["text"]["format"]["type"] == "json_schema"
    assert request["payload"]["text"]["format"]["strict"] is True
    prompt = json.loads(request["payload"]["input"])
    assert prompt["messages"][-1]["content"]["run_ids"] == ["run_selected"]
    assert "arrays" not in request["payload"]["input"].casefold()
    assert provider.take_metrics().total_tokens == 52
    assert provider.take_metrics().total_tokens is None


def test_independent_planner_rejects_provider_attempt_to_escape_run_context():
    transport = CapturingTransport({"output_text": encoded_plan({"run_id": "run_other"})})
    config = ProviderConfiguration(
        "openai_planner", "openai_responses", "configured-model",
        "https://api.openai.com/v1", "PYMRST_TEST_OPENAI_KEY",
    )
    provider = OpenAIResponsesPlanningProvider(config, api_key="memory-only", transport=transport)
    with pytest.raises(ValueError, match="out-of-context run_id"):
        BoundedInvestigationPlanner(provider).plan(context(), (descriptor(),))


def test_local_adapters_reject_non_loopback_endpoints_and_parse_ollama_schema():
    with pytest.raises(ValueError, match="loopback"):
        ProviderConfiguration(
            "remote_local", "ollama", "model", "http://example.com:11434",
        )
    transport = CapturingTransport({
        "message": {"content": encoded_plan({"run_id": "run_selected"})},
    })
    config = ProviderConfiguration(
        "ollama_local", "ollama", "local-model", "http://127.0.0.1:11434",
    )
    provider = OllamaPlanningProvider(config, transport=transport)
    planned = BoundedInvestigationPlanner(provider).plan(context(), (descriptor(),))
    assert planned.plan.calls[0].tool == "analyze_conservation"
    assert transport.calls[0][0] == "http://127.0.0.1:11434/api/chat"
    assert transport.calls[0][1]["payload"]["stream"] is False


def test_openai_adapter_uses_a_separate_strict_narrative_contract():
    narrative = {
        "schema_version": 1, "title": "Evidence answer",
        "executive_summary": "Accepted findings support the answer.",
        "sections": [{
            "section_id": "pressure", "heading": "Pressure",
            "narrative": "Pressure agrees.", "finding_ids": ["finding_pressure"],
        }],
    }
    transport = CapturingTransport({"output_text": json.dumps(narrative)})
    config = ProviderConfiguration(
        "openai_narrative", "openai_responses", "configured-model",
        "https://api.openai.com/v1", "PYMRST_TEST_OPENAI_KEY",
    )
    provider = OpenAIResponsesPlanningProvider(config, api_key="memory-only", transport=transport)
    result = provider.complete(
        ({"role": "user", "content": {"accepted_findings": []}},), (), NarrativeProposal,
    )
    assert result.sections[0].finding_ids == ("finding_pressure",)
    format_spec = transport.calls[0][1]["payload"]["text"]["format"]
    assert format_spec["name"] == "pymrst_evidence_narrative"
    assert format_spec["strict"] is True
