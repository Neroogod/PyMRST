"""SQLite application database with explicit forward-only migrations."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading


APP_DATABASE_SCHEMA_VERSION = 20


_MIGRATION_1 = """
CREATE TABLE app_projects (
  project_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  tags_json TEXT NOT NULL,
  root_path TEXT NOT NULL UNIQUE,
  content_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  schema_version INTEGER NOT NULL
);
CREATE TABLE app_assets (
  asset_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  kind TEXT NOT NULL,
  original_name TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
  created_at TEXT NOT NULL,
  UNIQUE(project_id, content_hash)
);
CREATE TABLE app_tasks (
  task_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  progress REAL NOT NULL CHECK(progress >= 0 AND progress <= 1),
  message TEXT NOT NULL,
  request_json TEXT NOT NULL,
  result_json TEXT,
  error_json TEXT,
  cancellation_requested INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX app_tasks_project_status ON app_tasks(project_id, status);
CREATE TABLE app_notifications (
  notification_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL DEFAULT '',
  task_id TEXT NOT NULL DEFAULT '',
  severity TEXT NOT NULL,
  title TEXT NOT NULL,
  message TEXT NOT NULL,
  is_read INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE INDEX app_notifications_project_read ON app_notifications(project_id, is_read);
CREATE TABLE app_settings (
  setting_key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


_MIGRATION_2 = """
CREATE TABLE app_models (
  model_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  asset_id TEXT NOT NULL REFERENCES app_assets(asset_id),
  name TEXT NOT NULL,
  kind TEXT NOT NULL,
  unit_system TEXT NOT NULL,
  status TEXT NOT NULL,
  source_hash TEXT NOT NULL,
  inspection_hash TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(project_id, asset_id)
);
CREATE INDEX app_models_project ON app_models(project_id, created_at);
CREATE TABLE app_input_patches (
  patch_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  model_id TEXT NOT NULL REFERENCES app_models(model_id),
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  base_hash TEXT NOT NULL,
  patch_hash TEXT NOT NULL,
  operations_json TEXT NOT NULL,
  status TEXT NOT NULL,
  variant_relative_path TEXT NOT NULL DEFAULT '',
  variant_hash TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  materialized_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX app_input_patches_model ON app_input_patches(model_id, created_at);
CREATE TABLE app_grid_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  model_id TEXT REFERENCES app_models(model_id),
  name TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX app_grid_snapshots_project ON app_grid_snapshots(project_id, created_at);
"""


_MIGRATION_3 = """
CREATE TABLE app_run_configurations (
  configuration_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  model_id TEXT REFERENCES app_models(model_id),
  name TEXT NOT NULL,
  runner_kind TEXT NOT NULL,
  configuration_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(project_id, configuration_hash)
);
CREATE INDEX app_run_configurations_project ON app_run_configurations(project_id, created_at);
CREATE TABLE app_runs (
  run_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  configuration_id TEXT NOT NULL REFERENCES app_run_configurations(configuration_id),
  task_id TEXT NOT NULL REFERENCES app_tasks(task_id),
  attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
  parent_run_id TEXT REFERENCES app_runs(run_id),
  resume_checkpoint_id TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL,
  run_relative_path TEXT NOT NULL,
  process_id INTEGER,
  started_at TEXT NOT NULL DEFAULT '',
  finished_at TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  UNIQUE(configuration_id, attempt_number)
);
CREATE INDEX app_runs_project_status ON app_runs(project_id, status, created_at);
CREATE TABLE app_run_events (
  run_id TEXT NOT NULL REFERENCES app_runs(run_id),
  sequence INTEGER NOT NULL CHECK(sequence >= 0),
  event_type TEXT NOT NULL,
  elapsed_seconds REAL NOT NULL CHECK(elapsed_seconds >= 0),
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(run_id, sequence)
);
CREATE INDEX app_run_events_type ON app_run_events(run_id, event_type, sequence);
CREATE TABLE app_run_checkpoints (
  checkpoint_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES app_runs(run_id),
  sequence INTEGER NOT NULL,
  relative_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(run_id, sequence)
);
"""


_MIGRATION_4 = """
CREATE TABLE app_result_states (
  state_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES app_runs(run_id),
  report_step INTEGER NOT NULL CHECK(report_step > 0),
  simulated_time_seconds REAL NOT NULL CHECK(simulated_time_seconds >= 0),
  relative_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  fields_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(run_id, report_step)
);
CREATE INDEX app_result_states_run ON app_result_states(run_id, report_step);
CREATE TABLE app_result_comparisons (
  comparison_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  reference_run_id TEXT NOT NULL REFERENCES app_runs(run_id),
  candidate_run_id TEXT NOT NULL REFERENCES app_runs(run_id),
  content_hash TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(reference_run_id, candidate_run_id, content_hash)
);
CREATE INDEX app_result_comparisons_project ON app_result_comparisons(project_id, created_at);
CREATE TABLE app_result_exports (
  export_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES app_runs(run_id),
  kind TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX app_result_exports_run ON app_result_exports(run_id, created_at);
"""


_MIGRATION_5 = """
CREATE TABLE app_model_drafts (
  draft_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  grid_snapshot_id TEXT NOT NULL REFERENCES app_grid_snapshots(snapshot_id),
  name TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version > 0),
  status TEXT NOT NULL,
  document_hash TEXT NOT NULL,
  document_json TEXT NOT NULL,
  issues_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX app_model_drafts_project ON app_model_drafts(project_id, updated_at);
CREATE TABLE app_model_draft_events (
  draft_id TEXT NOT NULL REFERENCES app_model_drafts(draft_id),
  version INTEGER NOT NULL,
  document_hash TEXT NOT NULL,
  document_json TEXT NOT NULL,
  actor TEXT NOT NULL,
  reason TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(draft_id, version)
);
CREATE TABLE app_model_revisions (
  revision_id TEXT PRIMARY KEY,
  draft_id TEXT NOT NULL REFERENCES app_model_drafts(draft_id),
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  revision_number INTEGER NOT NULL CHECK(revision_number > 0),
  content_hash TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(draft_id, revision_number),
  UNIQUE(draft_id, content_hash)
);
CREATE INDEX app_model_revisions_project ON app_model_revisions(project_id, created_at);
"""


_MIGRATION_6 = """
CREATE TABLE app_experiments (
  experiment_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  source_kind TEXT NOT NULL,
  source_id TEXT NOT NULL,
  name TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  definition_json TEXT NOT NULL,
  matrix_json TEXT NOT NULL,
  materialized_json TEXT NOT NULL,
  member_count INTEGER NOT NULL CHECK(member_count > 0),
  created_at TEXT NOT NULL,
  UNIQUE(project_id, content_hash)
);
CREATE INDEX app_experiments_project ON app_experiments(project_id, created_at);
CREATE TABLE app_experiment_executions (
  execution_id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES app_experiments(experiment_id),
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  target_kind TEXT NOT NULL,
  configuration_id TEXT REFERENCES app_run_configurations(configuration_id),
  run_id TEXT REFERENCES app_runs(run_id),
  status TEXT NOT NULL,
  target_json TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX app_experiment_executions_experiment ON app_experiment_executions(experiment_id, created_at);
CREATE TABLE app_experiment_qualifications (
  qualification_id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES app_experiments(experiment_id),
  execution_id TEXT NOT NULL REFERENCES app_experiment_executions(execution_id),
  reference_kind TEXT NOT NULL,
  reference_id TEXT NOT NULL,
  status TEXT NOT NULL,
  tolerances_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(execution_id, reference_kind, reference_id)
);
CREATE INDEX app_experiment_qualifications_experiment ON app_experiment_qualifications(experiment_id, created_at);
CREATE TABLE app_experiment_branches (
  branch_id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES app_experiments(experiment_id),
  execution_id TEXT NOT NULL REFERENCES app_experiment_executions(execution_id),
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  parent_member_id TEXT NOT NULL,
  configuration_id TEXT NOT NULL REFERENCES app_run_configurations(configuration_id),
  run_id TEXT REFERENCES app_runs(run_id),
  status TEXT NOT NULL,
  definition_json TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX app_experiment_branches_experiment ON app_experiment_branches(experiment_id, created_at);
"""


_MIGRATION_7 = """
CREATE TABLE app_investigations (
  investigation_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  question TEXT NOT NULL,
  status TEXT NOT NULL,
  context_json TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(project_id, content_hash)
);
CREATE INDEX app_investigations_project ON app_investigations(project_id, created_at);
CREATE TABLE app_investigation_findings (
  finding_id TEXT PRIMARY KEY,
  investigation_id TEXT NOT NULL REFERENCES app_investigations(investigation_id),
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  finding_json TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  review_status TEXT NOT NULL,
  review_note TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(investigation_id, content_hash)
);
CREATE INDEX app_investigation_findings_owner ON app_investigation_findings(investigation_id, created_at);
CREATE TABLE app_engineering_reports (
  report_id TEXT PRIMARY KEY,
  investigation_id TEXT NOT NULL REFERENCES app_investigations(investigation_id),
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  content_hash TEXT NOT NULL,
  report_json TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(investigation_id, content_hash)
);
CREATE INDEX app_engineering_reports_project ON app_engineering_reports(project_id, created_at);
"""


_MIGRATION_8 = """
CREATE TABLE app_variant_proposals (
  proposal_id TEXT PRIMARY KEY,
  investigation_id TEXT NOT NULL REFERENCES app_investigations(investigation_id),
  finding_id TEXT NOT NULL REFERENCES app_investigation_findings(finding_id),
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  name TEXT NOT NULL,
  proposal_kind TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  proposal_json TEXT NOT NULL,
  status TEXT NOT NULL,
  decision_note TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(investigation_id, content_hash)
);
CREATE INDEX app_variant_proposals_investigation ON app_variant_proposals(investigation_id, created_at);
"""


_MIGRATION_9 = """
CREATE TABLE app_investigation_plans (
  plan_id TEXT PRIMARY KEY,
  investigation_id TEXT NOT NULL REFERENCES app_investigations(investigation_id),
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  provider TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  plan_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(investigation_id, content_hash)
);
CREATE INDEX app_investigation_plans_owner ON app_investigation_plans(investigation_id, created_at);
"""


_MIGRATION_10 = """
CREATE TABLE app_investigation_executions (
  execution_id TEXT PRIMARY KEY,
  investigation_id TEXT NOT NULL REFERENCES app_investigations(investigation_id),
  plan_id TEXT NOT NULL REFERENCES app_investigation_plans(plan_id),
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  plan_hash TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  status TEXT NOT NULL,
  execution_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  completed_at TEXT NOT NULL,
  UNIQUE(investigation_id, plan_id, content_hash)
);
CREATE INDEX app_investigation_executions_owner
  ON app_investigation_executions(investigation_id, created_at);
"""


_MIGRATION_11 = """
CREATE TABLE app_conflict_decisions (
  decision_id TEXT PRIMARY KEY,
  investigation_id TEXT NOT NULL REFERENCES app_investigations(investigation_id),
  conflict_id TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  decision_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(investigation_id, conflict_id)
);
CREATE INDEX app_conflict_decisions_owner
  ON app_conflict_decisions(investigation_id, created_at);
CREATE TABLE app_investigation_syntheses (
  synthesis_id TEXT PRIMARY KEY,
  investigation_id TEXT NOT NULL REFERENCES app_investigations(investigation_id),
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  finding_set_hash TEXT NOT NULL,
  decision_set_hash TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  synthesis_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(investigation_id, finding_set_hash, decision_set_hash)
);
CREATE INDEX app_investigation_syntheses_owner
  ON app_investigation_syntheses(investigation_id, created_at);
"""


_MIGRATION_12 = """
CREATE TABLE app_reasoning_providers (
  provider_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  model TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  credential_env TEXT NOT NULL,
  enabled INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  configuration_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX app_reasoning_providers_enabled
  ON app_reasoning_providers(enabled, provider_id);
"""


_MIGRATION_13 = """
CREATE TABLE app_reasoning_provider_checks (
  check_id TEXT PRIMARY KEY,
  provider_id TEXT NOT NULL REFERENCES app_reasoning_providers(provider_id),
  status TEXT NOT NULL,
  latency_seconds REAL NOT NULL CHECK(latency_seconds >= 0),
  configuration_hash TEXT NOT NULL,
  details_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX app_reasoning_provider_checks_owner
  ON app_reasoning_provider_checks(provider_id, created_at);
"""


_MIGRATION_14 = """
CREATE TABLE app_reasoning_provider_evaluations (
  evaluation_id TEXT PRIMARY KEY,
  provider_id TEXT NOT NULL,
  configuration_hash TEXT NOT NULL,
  suite_hash TEXT NOT NULL,
  status TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  report_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX app_reasoning_provider_evaluations_owner
  ON app_reasoning_provider_evaluations(provider_id, created_at);
"""


_MIGRATION_15 = """
CREATE TABLE app_investigation_narratives (
  narrative_id TEXT PRIMARY KEY,
  investigation_id TEXT NOT NULL REFERENCES app_investigations(investigation_id),
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  provider TEXT NOT NULL,
  provider_configuration_hash TEXT NOT NULL,
  source_findings_hash TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  narrative_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('draft', 'approved', 'rejected')),
  reviewer TEXT NOT NULL DEFAULT '',
  review_note TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(investigation_id, provider, source_findings_hash, content_hash)
);
CREATE INDEX app_investigation_narratives_owner
  ON app_investigation_narratives(investigation_id, created_at);
"""


_MIGRATION_16 = """
CREATE TABLE app_workspace_backups (
  backup_id TEXT PRIMARY KEY,
  relative_path TEXT NOT NULL UNIQUE,
  content_hash TEXT NOT NULL,
  size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
  project_count INTEGER NOT NULL CHECK(project_count >= 0),
  file_count INTEGER NOT NULL CHECK(file_count >= 0),
  status TEXT NOT NULL CHECK(status IN ('created', 'verified', 'failed')),
  manifest_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  verified_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX app_workspace_backups_created ON app_workspace_backups(created_at);
"""


_MIGRATION_17 = """
ALTER TABLE app_workspace_backups ADD COLUMN pruned_at TEXT NOT NULL DEFAULT '';
CREATE TABLE app_backup_jobs (
  job_id TEXT PRIMARY KEY,
  status TEXT NOT NULL CHECK(status IN (
    'queued','running','completed','failed','cancelled','interrupted'
  )),
  progress REAL NOT NULL CHECK(progress >= 0 AND progress <= 1),
  message TEXT NOT NULL,
  backup_id TEXT REFERENCES app_workspace_backups(backup_id),
  error_json TEXT,
  cancellation_requested INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX app_backup_jobs_status ON app_backup_jobs(status, created_at);
CREATE TABLE app_backup_qualifications (
  qualification_id TEXT PRIMARY KEY,
  backup_id TEXT NOT NULL REFERENCES app_workspace_backups(backup_id),
  status TEXT NOT NULL CHECK(status IN ('passed','failed')),
  checks_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX app_backup_qualifications_owner
  ON app_backup_qualifications(backup_id, created_at);
"""


_MIGRATION_18 = """
CREATE TABLE IF NOT EXISTS app_users (
  user_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE COLLATE NOCASE,
  system_role TEXT NOT NULL CHECK(system_role IN ('admin','engineer','viewer','auditor')),
  active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS app_project_memberships (
  project_id TEXT NOT NULL REFERENCES app_projects(project_id),
  user_id TEXT NOT NULL REFERENCES app_users(user_id),
  project_role TEXT NOT NULL CHECK(project_role IN ('owner','editor','runner','analyst','viewer')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(project_id, user_id)
);
CREATE INDEX IF NOT EXISTS app_project_memberships_user
  ON app_project_memberships(user_id, project_id);
CREATE TABLE IF NOT EXISTS app_access_tokens (
  token_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES app_users(user_id),
  label TEXT NOT NULL,
  token_prefix TEXT NOT NULL UNIQUE,
  token_hash TEXT NOT NULL UNIQUE,
  scopes_json TEXT NOT NULL,
  expires_at TEXT NOT NULL DEFAULT '',
  revoked_at TEXT NOT NULL DEFAULT '',
  last_used_at TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS app_access_tokens_user ON app_access_tokens(user_id, created_at);
CREATE TABLE IF NOT EXISTS app_security_audit (
  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  actor_id TEXT NOT NULL,
  action TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  resource_id TEXT NOT NULL DEFAULT '',
  project_id TEXT NOT NULL DEFAULT '',
  outcome TEXT NOT NULL CHECK(outcome IN ('allowed','denied','succeeded','failed')),
  details_json TEXT NOT NULL,
  previous_hash TEXT NOT NULL,
  event_hash TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS app_security_audit_project
  ON app_security_audit(project_id, sequence);
CREATE INDEX IF NOT EXISTS app_security_audit_actor
  ON app_security_audit(actor_id, sequence);
"""


_MIGRATION_19 = """
CREATE TABLE IF NOT EXISTS app_environment_qualifications (
  qualification_id TEXT PRIMARY KEY,
  status TEXT NOT NULL CHECK(status IN ('ready','degraded','blocked')),
  content_hash TEXT NOT NULL,
  platform_json TEXT NOT NULL,
  checks_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS app_environment_qualifications_created
  ON app_environment_qualifications(created_at);
"""


_MIGRATION_20 = """
CREATE TABLE IF NOT EXISTS app_workload_qualifications (
  qualification_id TEXT PRIMARY KEY,
  workload_id TEXT NOT NULL,
  tier TEXT NOT NULL CHECK(tier IN ('smoke','engineering','field_scale','endurance')),
  status TEXT NOT NULL CHECK(status IN ('queued','running','passed','failed','cancelled','interrupted')),
  progress REAL NOT NULL CHECK(progress >= 0 AND progress <= 1),
  message TEXT NOT NULL,
  request_json TEXT NOT NULL,
  budgets_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  comparison_json TEXT NOT NULL,
  evidence_relative_path TEXT NOT NULL DEFAULT '',
  content_hash TEXT NOT NULL DEFAULT '',
  process_id INTEGER,
  cancellation_requested INTEGER NOT NULL DEFAULT 0,
  error_json TEXT,
  started_at TEXT NOT NULL DEFAULT '',
  finished_at TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS app_workload_qualifications_kind
  ON app_workload_qualifications(workload_id, tier, created_at);
CREATE INDEX IF NOT EXISTS app_workload_qualifications_status
  ON app_workload_qualifications(status, created_at);
"""


class AppDatabase:
    """Thread-safe metadata boundary; large numerical data never enters SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            with self._lock:
                self._connection.execute("PRAGMA foreign_keys=ON")
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=FULL")
                self._connection.execute("PRAGMA busy_timeout=5000")
                self._migrate()
        except Exception:
            self._connection.close()
            raise

    def _migrate(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version > APP_DATABASE_SCHEMA_VERSION:
            raise RuntimeError(
                f"application database schema {version} is newer than supported "
                f"{APP_DATABASE_SCHEMA_VERSION}"
            )
        if version == 0:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_1
                    + "\nPRAGMA user_version=1;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 1
        if version == 1:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_2
                    + "\nPRAGMA user_version=2;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 2
        if version == 2:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_3
                    + "\nPRAGMA user_version=3;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 3
        if version == 3:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_4
                    + "\nPRAGMA user_version=4;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 4
        if version == 4:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_5
                    + "\nPRAGMA user_version=5;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 5
        if version == 5:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_6
                    + "\nPRAGMA user_version=6;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 6
        if version == 6:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_7
                    + "\nPRAGMA user_version=7;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 7
        if version == 7:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_8
                    + "\nPRAGMA user_version=8;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 8
        if version == 8:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_9
                    + "\nPRAGMA user_version=9;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 9
        if version == 9:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_10
                    + "\nPRAGMA user_version=10;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 10
        if version == 10:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_11
                    + "\nPRAGMA user_version=11;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 11
        if version == 11:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_12
                    + "\nPRAGMA user_version=12;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 12
        if version == 12:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_13
                    + "\nPRAGMA user_version=13;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 13
        if version == 13:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_14
                    + "\nPRAGMA user_version=14;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 14
        if version == 14:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_15
                    + "\nPRAGMA user_version=15;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 15
        if version == 15:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_16
                    + "\nPRAGMA user_version=16;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 16
        if version == 16:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_17
                    + "\nPRAGMA user_version=17;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 17
        if version == 17:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_18
                    + "\nPRAGMA user_version=18;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 18
        if version == 18:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_19
                    + "\nPRAGMA user_version=19;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = 19
        if version == 19:
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + _MIGRATION_20
                    + "\nPRAGMA user_version=20;\nCOMMIT;"
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    @contextmanager
    def transaction(self):
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def fetch_one(self, query: str, parameters=()):
        with self._lock:
            return self._connection.execute(query, parameters).fetchone()

    def fetch_all(self, query: str, parameters=()):
        with self._lock:
            return tuple(self._connection.execute(query, parameters).fetchall())

    def backup_to(self, target: str | Path) -> Path:
        """Create a transactionally consistent SQLite snapshot."""
        target = Path(target).resolve()
        if target == self.path:
            raise ValueError("database backup target must differ from source")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        destination = sqlite3.connect(target)
        try:
            with self._lock:
                self._connection.backup(destination)
            result = destination.execute("PRAGMA integrity_check").fetchone()[0]
            if result != "ok":
                raise RuntimeError(f"database snapshot failed integrity check: {result}")
        except Exception:
            destination.close()
            target.unlink(missing_ok=True)
            raise
        destination.close()
        return target

    @property
    def schema_version(self) -> int:
        with self._lock:
            return int(self._connection.execute("PRAGMA user_version").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


__all__ = ["APP_DATABASE_SCHEMA_VERSION", "AppDatabase"]
