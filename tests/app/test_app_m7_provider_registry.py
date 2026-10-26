"""Application persistence tests for optional reasoning providers."""

import json

import pytest

from pymrst.app import create_application
from pymrst.app.services.provider_service import ReasoningProviderService
from pymrst.research_studio.agents.providers import ProviderConfiguration
from pymrst.research_studio.agents.planning import DeterministicPlanningProvider


class HealthTransport:
    def post_json(self, url, **kwargs):
        return {"output_text": json.dumps({"calls": [{
            "call_id": "health", "tool": "provider_health_check",
            "arguments_json": json.dumps({"probe": "pymrst-provider-connectivity"}),
        }]})}


def test_provider_registry_persists_environment_reference_but_never_secret(tmp_path):
    root = tmp_path / "workspace"
    with create_application(root) as app:
        service = ReasoningProviderService(
            app.database, environment={"PYMRST_PROVIDER_SECRET": "super-secret-value"},
        )
        registered = service.register(ProviderConfiguration(
            "openai_research", "openai_responses", "configured-model",
            "https://api.openai.com/v1", "PYMRST_PROVIDER_SECRET",
        ))
        assert registered.credential_env == "PYMRST_PROVIDER_SECRET"
        row = app.database.fetch_one(
            "SELECT configuration_json FROM app_reasoning_providers WHERE provider_id=?",
            (registered.provider_id,),
        )
        assert "super-secret-value" not in row["configuration_json"]
        assert json.loads(row["configuration_json"])["credential_env"] == (
            "PYMRST_PROVIDER_SECRET"
        )
        assert service.build("openai_research").api_key == "super-secret-value"
        check = service.check("openai_research", transport=HealthTransport())
        assert check.status == "passed"
        assert check.details == {
            "kind": "openai_responses", "model": "configured-model",
            "endpoint": "https://api.openai.com/v1",
            "probe": "synthetic_plan_only", "contract_valid": True,
        }
        assert service.checks("openai_research") == (check,)
        disabled = service.set_enabled("openai_research", False)
        assert disabled.enabled is False
        with pytest.raises(ValueError, match="disabled"):
            service.build("openai_research")
        assert service.set_enabled("openai_research", True).enabled is True
        reregistered = service.register(ProviderConfiguration(
            "openai_research", "openai_responses", "configured-model",
            "https://api.openai.com/v1", "PYMRST_PROVIDER_SECRET",
        ))
        assert reregistered.provider_id == registered.provider_id
        assert reregistered.content_hash == registered.content_hash
        assert reregistered.created_at == registered.created_at
        assert reregistered.enabled is True
        with pytest.raises(ValueError, match="immutable"):
            service.register(ProviderConfiguration(
                "openai_research", "openai_responses", "different-model",
                "https://api.openai.com/v1", "PYMRST_PROVIDER_SECRET",
            ))


def test_provider_options_reject_embedded_credentials(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        with pytest.raises(ValueError, match="cannot persist credentials"):
            app.providers.register(ProviderConfiguration(
                "unsafe", "ollama", "local", "http://localhost:11434",
                options={"api_key": "must-not-be-stored"},
            ))


def test_provider_check_failure_is_audited_without_raising_or_storing_secret(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        service = ReasoningProviderService(app.database, environment={})
        service.register(ProviderConfiguration(
            "missing_key", "openai_responses", "configured-model",
            "https://api.openai.com/v1", "MISSING_TEST_KEY",
        ))
        check = service.check("missing_key")
        assert check.status == "failed"
        assert check.details["error_type"] == "ValueError"
        encoded = json.dumps(check.to_dict())
        assert "MISSING_TEST_KEY" not in encoded
        assert "api key is unavailable" in check.details["error"].casefold()


def test_provider_evaluations_are_durable_comparable_and_synthetic(tmp_path):
    root = tmp_path / "workspace"
    with create_application(root) as app:
        evaluation = app.providers.evaluate("deterministic-v1")
        assert evaluation.status == "qualified"
        assert evaluation.report["summary"]["pass_rate"] == 1.0
        assert evaluation.report["summary"]["fixture_count"] == 3
        assert app.providers.evaluations("deterministic-v1") == (evaluation,)
        assert app.providers.evaluations() == (evaluation,)
        encoded = json.dumps(evaluation.to_dict()).casefold()
        assert "synthetic_reference" not in encoded
        assert "reservoir" not in encoded

    with create_application(root) as reopened:
        assert reopened.providers.evaluations("deterministic-v1") == (evaluation,)
