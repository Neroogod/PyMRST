"""Credential-safe registry for optional planning providers."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Mapping
from uuid import uuid4

from pymrst.research_studio.agents.providers import (
    OllamaPlanningProvider, OpenAICompatiblePlanningProvider,
    OpenAIResponsesPlanningProvider, ProviderConfiguration, ProviderKind,
)
from pymrst.research_studio.agents.orchestrator import InvestigationPlan
from pymrst.research_studio.agents.planning import DeterministicPlanningProvider
from pymrst.research_studio.agents.providers.evaluation import (
    PlanningProviderEvaluator, default_planning_evaluation_fixtures,
)

from ..database.models import AppDatabase
from ..schemas._common import utc_now
from ..schemas.agent import (
    ReasoningProviderRecord, ReasoningProviderCheckRecord,
    ReasoningProviderEvaluationRecord,
)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _record(row):
    return ReasoningProviderRecord(
        row["provider_id"], row["kind"], row["model"], row["endpoint"],
        row["credential_env"], bool(row["enabled"]), row["content_hash"],
        json.loads(row["configuration_json"]), row["created_at"], row["updated_at"],
    )


def _check(row):
    return ReasoningProviderCheckRecord(
        row["check_id"], row["provider_id"], row["status"],
        float(row["latency_seconds"]), row["configuration_hash"],
        json.loads(row["details_json"]), row["created_at"],
    )


def _evaluation(row):
    return ReasoningProviderEvaluationRecord(
        row["evaluation_id"], row["provider_id"], row["configuration_hash"],
        row["suite_hash"], row["status"], row["content_hash"],
        json.loads(row["report_json"]), row["created_at"],
    )


def _sensitive_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if any(token in normalized for token in ("secret", "password", "api_key", "token")):
                yield str(key)
            yield from _sensitive_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _sensitive_keys(child)


class ReasoningProviderService:
    def __init__(self, database: AppDatabase, *, environment: Mapping[str, str] | None = None):
        self.database = database
        self.environment = os.environ if environment is None else environment

    def register(self, configuration: ProviderConfiguration) -> ReasoningProviderRecord:
        if not isinstance(configuration, ProviderConfiguration):
            raise TypeError("provider registration requires ProviderConfiguration")
        value = configuration.to_dict()
        forbidden = tuple(_sensitive_keys(value.get("options", {})))
        if forbidden:
            raise ValueError(f"provider configuration cannot persist credentials: {forbidden}")
        encoded = _encoded(value)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        now = utc_now()
        existing = self.database.fetch_one(
            "SELECT * FROM app_reasoning_providers WHERE provider_id=?",
            (configuration.provider_id,),
        )
        if existing is not None and existing["content_hash"] != digest:
            raise ValueError("provider identifiers are immutable; register a new provider id")
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO app_reasoning_providers VALUES(?,?,?,?,?,?,?,?,?,?)",
                (configuration.provider_id, configuration.kind.value,
                 configuration.model, configuration.endpoint.rstrip("/"),
                 configuration.credential_env, int(configuration.enabled),
                 digest, encoded, now, now),
            )
        return self.get(configuration.provider_id)

    def get(self, provider_id: str) -> ReasoningProviderRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_reasoning_providers WHERE provider_id=?", (str(provider_id),),
        )
        if row is None:
            raise KeyError(f"unknown reasoning provider: {provider_id}")
        return _record(row)

    def list(self) -> tuple[ReasoningProviderRecord, ...]:
        return tuple(_record(row) for row in self.database.fetch_all(
            "SELECT * FROM app_reasoning_providers ORDER BY provider_id",
        ))

    def set_enabled(self, provider_id: str, enabled: bool) -> ReasoningProviderRecord:
        current = self.get(provider_id)
        value = dict(current.configuration)
        value["enabled"] = bool(enabled)
        configuration = ProviderConfiguration(**value)
        encoded = _encoded(configuration.to_dict())
        digest, now = hashlib.sha256(encoded.encode()).hexdigest(), utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_reasoning_providers SET enabled=?,content_hash=?,"
                "configuration_json=?,updated_at=? WHERE provider_id=?",
                (int(bool(enabled)), digest, encoded, now, provider_id),
            )
        return self.get(provider_id)

    def build(self, provider_id: str, *, transport=None):
        record = self.get(provider_id)
        if not record.enabled:
            raise ValueError("reasoning provider is disabled")
        config = ProviderConfiguration(**record.configuration)
        credential = (
            str(self.environment.get(config.credential_env, "")).strip()
            if config.credential_env else ""
        )
        if config.kind is ProviderKind.OPENAI_RESPONSES:
            return OpenAIResponsesPlanningProvider(
                config, api_key=credential, transport=transport,
            )
        if config.kind is ProviderKind.OPENAI_COMPATIBLE:
            return OpenAICompatiblePlanningProvider(
                config, api_key=credential, transport=transport,
            )
        if config.kind is ProviderKind.OLLAMA:
            return OllamaPlanningProvider(config, transport=transport)
        raise ValueError(f"unsupported reasoning provider kind: {config.kind}")

    def check(self, provider_id: str, *, transport=None) -> ReasoningProviderCheckRecord:
        """Run a synthetic plan handshake; no project or simulator data is supplied."""
        record = self.get(provider_id)
        started = time.monotonic()
        status, details = "passed", {
            "kind": record.kind, "model": record.model,
            "endpoint": record.endpoint, "probe": "synthetic_plan_only",
        }
        try:
            provider = self.build(provider_id, transport=transport)
            plan = provider.complete(
                ({"role": "system", "content": "Return exactly one synthetic health-check call."},
                 {"role": "user", "content": {"probe": "pymrst-provider-connectivity"}}),
                ({
                    "name": "provider_health_check",
                    "description": "Return the supplied synthetic probe unchanged",
                    "permission": "analyze", "mutates_case": False,
                    "starts_workload": False, "requires_approval": False,
                },),
                InvestigationPlan,
            )
            calls = plan.calls
            if (len(calls) != 1 or calls[0].tool != "provider_health_check"
                    or calls[0].arguments != {"probe": "pymrst-provider-connectivity"}):
                raise RuntimeError("provider did not honor the synthetic health-check contract")
            details["contract_valid"] = True
        except Exception as error:
            status = "failed"
            details.update({
                "contract_valid": False, "error_type": type(error).__name__,
                "error": str(error) or type(error).__name__,
            })
        latency, now = max(0.0, time.monotonic() - started), utc_now()
        item = ReasoningProviderCheckRecord(
            f"provider_check_{uuid4().hex}", provider_id, status, latency,
            record.content_hash, details, now,
        )
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_reasoning_provider_checks VALUES(?,?,?,?,?,?,?)",
                (item.check_id, provider_id, status, latency,
                 record.content_hash, _encoded(details), now),
            )
        return item

    def checks(self, provider_id: str) -> tuple[ReasoningProviderCheckRecord, ...]:
        self.get(provider_id)
        return tuple(_check(row) for row in self.database.fetch_all(
            "SELECT * FROM app_reasoning_provider_checks WHERE provider_id=? "
            "ORDER BY created_at", (provider_id,),
        ))

    def evaluate(self, provider_id: str, *, provider=None) -> ReasoningProviderEvaluationRecord:
        provider_id = str(provider_id).strip()
        if provider_id == "deterministic-v1":
            configuration_hash = hashlib.sha256(b"deterministic-v1").hexdigest()
            adapter = provider or DeterministicPlanningProvider()
        else:
            registered = self.get(provider_id)
            configuration_hash = registered.content_hash
            adapter = provider or self.build(provider_id)
        report = PlanningProviderEvaluator(
            default_planning_evaluation_fixtures(),
        ).evaluate(adapter, provider_name=provider_id)
        value = report.to_dict()
        digest, now = hashlib.sha256(_encoded(value).encode()).hexdigest(), utc_now()
        item = ReasoningProviderEvaluationRecord(
            f"provider_evaluation_{uuid4().hex}", provider_id,
            configuration_hash, report.suite_hash, report.status,
            digest, value, now,
        )
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_reasoning_provider_evaluations VALUES(?,?,?,?,?,?,?,?)",
                (item.evaluation_id, provider_id, configuration_hash,
                 report.suite_hash, report.status, digest, _encoded(value), now),
            )
        return item

    def evaluations(self, provider_id: str | None = None):
        if provider_id is None:
            rows = self.database.fetch_all(
                "SELECT * FROM app_reasoning_provider_evaluations ORDER BY created_at",
            )
        else:
            rows = self.database.fetch_all(
                "SELECT * FROM app_reasoning_provider_evaluations "
                "WHERE provider_id=? ORDER BY created_at", (str(provider_id),),
            )
        return tuple(_evaluation(row) for row in rows)


__all__ = ["ReasoningProviderService"]
