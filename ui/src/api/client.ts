export interface ProjectRecord {
  schema_version: number;
  project_id: string;
  name: string;
  description: string;
  tags: string[];
  root_path: string;
  content_hash: string;
  created_at: string;
  updated_at: string;
}

export interface NotificationRecord {
  notification_id: string;
  project_id: string;
  task_id: string;
  severity: "info" | "success" | "warning" | "error";
  title: string;
  message: string;
  created_at: string;
  read: boolean;
}

export interface HealthRecord {
  status: string;
  database_schema: number;
}
export type Capability = "system_admin" | "view" | "edit" | "execute" | "publish" | "audit";
export interface SecurityActor {
  user_id: string; display_name: string; system_role: "admin" | "engineer" | "viewer" | "auditor";
  token_id: string; token_scopes: Capability[]; local_session: boolean; capabilities?: Capability[];
}
export interface SecurityUser {
  user_id: string; display_name: string; email: string;
  system_role: "admin" | "engineer" | "viewer" | "auditor";
  active: boolean; created_at: string; updated_at: string;
}
export interface SecurityToken {
  token_id: string; user_id: string; label: string; token_prefix: string;
  scopes: Capability[]; expires_at: string; revoked_at: string;
  last_used_at: string; created_at: string; token?: string; shown_once?: boolean;
}
export interface ProjectMembership {
  project_id: string; user_id: string;
  project_role: "owner" | "editor" | "runner" | "analyst" | "viewer";
  created_at: string; updated_at: string;
}
export interface SecurityAuditEvent {
  sequence: number; event_id: string; actor_id: string; action: string;
  resource_type: string; resource_id: string; project_id: string;
  outcome: "allowed" | "denied" | "succeeded" | "failed";
  details: Record<string, unknown>; previous_hash: string; event_hash: string; created_at: string;
}
export interface EnvironmentCheck {
  check_id: string; category: string; title: string;
  status: "passed" | "warning" | "failed"; required: boolean;
  message: string; details: Record<string, unknown>; remediation: string;
}
export interface EnvironmentQualification {
  qualification_id: string; status: "ready" | "degraded" | "blocked";
  content_hash: string; platform: Record<string, string>;
  checks: EnvironmentCheck[]; summary: { passed: number; warning: number; failed: number };
  created_at: string;
}
export type WorkloadTier = "smoke" | "engineering" | "field_scale" | "endurance";
export interface WorkloadBudget {
  maximum_wall_seconds: number; maximum_cpu_seconds: number;
  maximum_peak_rss_bytes: number; maximum_disk_bytes: number;
  maximum_wall_regression_ratio: number; maximum_memory_regression_ratio: number;
  maximum_wall_relative_mad: number;
}
export interface ExampleDefinition {
  example_id: string; title: string; description: string; category: string;
  runtime_class: "fast" | "medium" | "slow";
}
export interface WorkloadDefinition {
  workload_id: string; title: string; description: string; tiers: WorkloadTier[];
  default_budgets: Record<WorkloadTier, WorkloadBudget>;
  available: boolean; unavailable_reason: string; requirements: string[];
}
export interface WorkloadQualification {
  qualification_id: string; workload_id: string; tier: WorkloadTier;
  status: "queued" | "running" | "passed" | "failed" | "cancelled" | "interrupted";
  progress: number; message: string; request: Record<string, unknown>;
  budgets: WorkloadBudget; metrics: Record<string, unknown>;
  comparison: { baseline_qualification_id?: string; ratios?: Record<string, { actual: number; maximum: number; passed: boolean }>; checks?: Record<string, { actual: number; maximum: number; passed: boolean }>; correctness?: Record<string, boolean> };
  evidence_relative_path: string; content_hash: string; process_id: number | null;
  cancellation_requested: boolean; error: Record<string, unknown> | null;
  started_at: string; finished_at: string; created_at: string; updated_at: string;
}
export interface BackupPlan {
  project_count: number; file_count: number; estimated_bytes: number;
  database_schema_version: number; includes: string[];
}
export interface BackupRecord {
  backup_id: string; relative_path: string; content_hash: string; size_bytes: number;
  project_count: number; file_count: number; status: "created" | "verified" | "failed";
  manifest: Record<string, unknown>; created_at: string; verified_at: string; pruned_at: string;
}
export interface BackupJobRecord {
  job_id: string; status: "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted";
  progress: number; message: string; backup_id: string; error: Record<string, unknown> | null;
  cancellation_requested: boolean; created_at: string; updated_at: string;
}
export interface BackupPolicy {
  enabled: boolean; interval_hours: number; retained_verified_backups: number;
  auto_verify: boolean; recovery_drill: boolean;
}
export interface BackupQualificationRecord {
  qualification_id: string; backup_id: string; status: "passed" | "failed";
  checks: Record<string, unknown>; created_at: string;
}
export interface RestoreReceipt {
  backup_id: string; target_root: string; archive_hash: string; file_count: number;
  restored_bytes: number; database_integrity: string; completed_at: string;
}

export interface ProjectAsset {
  asset_id: string; project_id: string; kind: string; original_name: string;
  relative_path: string; content_hash: string; size_bytes: number; created_at: string;
}

export interface SourceLocation {
  asset_id: string; path: string; line: number; column: number;
}

export interface ValidationIssue {
  code: string; severity: "error" | "warning" | "info"; category: string;
  message: string; location: SourceLocation | null; details: Record<string, unknown>;
}

export interface KeywordOccurrence {
  occurrence_id: string; keyword: string; section: string; location: SourceLocation;
  end_line: number; record_count: number; value_count: number; defaulted_value_count: number;
}

export interface ModelInspection {
  model_id: string; project_id: string; asset_id: string; name: string;
  kind: string; unit_system: string; status: "ready" | "invalid";
  source_hash: string; inspection_hash: string; sections: string[];
  files: Array<{ path: string; content_hash: string; size_bytes: number; line_count: number }>;
  occurrences?: KeywordOccurrence[]; issues: ValidationIssue[];
  created_at: string; updated_at: string;
}

export interface SourceSlice {
  model_id: string; path: string; start_line: number; end_line: number;
  total_lines: number; content_hash: string; lines: string[];
}

export interface GridArraySummary {
  name: string; shape: number[]; dtype: string; minimum: number | null;
  maximum: number | null; mean: number | null; percentiles: Record<string, number>;
  nonfinite_count: number;
}

export interface GridSnapshot {
  snapshot_id: string; project_id: string; model_id: string; name: string;
  content_hash: string; cell_count: number; face_count: number; node_count: number;
  dimension: number; active_cell_count: number; bounds: number[][];
  arrays: GridArraySummary[]; findings: Array<Record<string, unknown>>; created_at: string;
}

export interface GridValues {
  snapshot_id: string; array: string; offset: number; count: number;
  total: number; stride?: number; shape: number[]; values: number[] | number[][];
}

export interface InputPatch {
  patch_id: string; project_id: string; model_id: string; title: string;
  description: string; base_hash: string; patch_hash: string;
  operations: Array<{ kind: "replace_lines" | "insert_before" | "insert_after" | "delete_lines"; start_line: number; end_line: number; text: string }>;
  status: "draft" | "materialized"; variant_relative_path: string;
  variant_hash: string; created_at: string; materialized_at: string;
}

export interface PatchPreview {
  patch: InputPatch; diff: string; base_line_count: number;
  variant_line_count: number; variant_hash: string; original_unchanged: boolean;
}

export interface RunConfiguration {
  configuration_id: string; project_id: string; model_id: string; name: string;
  runner_kind: string; configuration_hash: string; payload: Record<string, unknown>;
  created_at: string;
}

export type RunStatus = "queued" | "starting" | "running" | "cancelling" | "completed" | "failed" | "cancelled" | "interrupted";
export interface RunRecord {
  run_id: string; project_id: string; configuration_id: string; task_id: string;
  attempt_number: number; parent_run_id: string; resume_checkpoint_id: string;
  status: RunStatus; run_relative_path: string; process_id: number | null;
  started_at: string; finished_at: string; created_at: string;
}

export interface RunEvent {
  run_id: string; sequence: number; event_type: string; elapsed_seconds: number;
  payload: Record<string, unknown>; created_at: string;
}

export interface RunCheckpoint {
  checkpoint_id: string; run_id: string; sequence: number; relative_path: string;
  content_hash: string; metadata: Record<string, unknown>; created_at: string;
}

export interface ResultField { name: string; shape: number[]; dtype: string; minimum: number | null; maximum: number | null; mean: number | null; nonfinite_count: number }
export interface ResultState {
  state_id: string; run_id: string; report_step: number; simulated_time_seconds: number;
  relative_path: string; content_hash: string; fields: ResultField[]; created_at: string;
}
export interface ResultDashboard {
  run_id: string; report_steps: number[]; times: number[];
  field: Record<string, Array<{ minimum: number; mean: number; maximum: number }>>;
  wells: Record<string, number[][]>; well_names: string[];
}
export interface RunComparison {
  comparison_id: string; project_id: string; reference_run_id: string;
  candidate_run_id: string; content_hash: string;
  summary: { common_report_steps: number[]; fields: Record<string, { maximum_absolute_error: number; maximum_relative_error: number; root_mean_square_error: number; worst_report_step: number; compared_values: number }> };
  created_at: string;
}

export interface FieldSource {
  kind: "uniform" | "grid_array"; value: number | null; array: string; scale: number;
}
export interface ConstructionWell {
  name: string; cells: number[]; control_type: string; target_si: number;
  radius_m: number; direction: "x" | "y" | "z"; composition: number[];
  sign: number; status: boolean; limits: Record<string, number>;
}
export interface ModelConstructionDocument {
  schema_version: 1;
  rock: { porosity: FieldSource; permeability_x: FieldSource; permeability_y: FieldSource | null; permeability_z: FieldSource | null; net_to_gross: FieldSource | null };
  fluid: { phases: string[]; viscosity_pa_s: number[]; density_kg_m3: number[]; relperm_exponent: number[]; compressibility_pa_inverse: number[]; reference_pressure_pa: number };
  initial_state: { pressure: FieldSource; saturations: Record<string, FieldSource>; dissolved_gas_ratio: FieldSource | null };
  wells: ConstructionWell[];
  schedule: { timestep_seconds: number[]; step_control: number[]; controls: Array<{ name: string; well_overrides: Record<string, Record<string, unknown>> }> };
}
export interface ConstructionIssue {
  code: string; severity: "error" | "warning" | "info"; section: string;
  message: string; details: Record<string, unknown>;
}
export interface ModelDraft {
  draft_id: string; project_id: string; grid_snapshot_id: string; name: string;
  version: number; status: "ready" | "invalid"; document_hash: string;
  document: ModelConstructionDocument; issues: ConstructionIssue[];
  created_at: string; updated_at: string;
}
export interface ModelRevision {
  revision_id: string; draft_id: string; project_id: string; revision_number: number;
  content_hash: string; relative_path: string; summary: Record<string, unknown>; created_at: string;
}
export interface ExperimentRecord {
  experiment_id: string; project_id: string; source_kind: "model_revision" | "run_configuration";
  source_id: string; name: string; content_hash: string; member_count: number;
  definition: Record<string, unknown>; matrix: Record<string, unknown>;
  coordinates: Array<{ member_id: string; ordinal: number; variant_name: string; comparison_group: string; parameters: Record<string, unknown>; components: Record<string, string> }>;
  created_at: string;
}
export interface ExperimentExecution {
  execution_id: string; experiment_id: string; project_id: string;
  target_kind: "local" | "slurm" | "ssh"; configuration_id: string; run_id: string;
  status: string; target: { kind: string; name: string; max_workers: number; memory_per_worker_bytes: number; configuration: Record<string, unknown> };
  relative_path: string; summary: Record<string, unknown>; created_at: string; updated_at: string;
}
export interface QualificationRecord {
  qualification_id: string; experiment_id: string; execution_id: string;
  reference_kind: string; reference_id: string; status: "passed" | "failed";
  tolerances: Record<string, unknown>; metrics: Record<string, unknown>;
  evidence: Record<string, unknown>; created_at: string;
}
export interface BranchRecord {
  branch_id: string; experiment_id: string; execution_id: string; project_id: string;
  parent_member_id: string; configuration_id: string; run_id: string; status: string;
  definition: Record<string, unknown>; relative_path: string;
  summary: Record<string, unknown>; created_at: string; updated_at: string;
}
export interface InvestigationRecord {
  investigation_id: string; project_id: string; question: string;
  status: "open" | "completed" | "unqualified" | "failed";
  context: { run_ids?: string[]; selection?: Record<string, unknown>; attributes?: Record<string, unknown> };
  content_hash: string; created_at: string; updated_at: string;
}
export interface FindingRecord {
  finding_id: string; investigation_id: string; project_id: string;
  finding: Record<string, unknown> & { agent?: string }; content_hash: string;
  review_status: "pending" | "accepted" | "rejected"; review_note: string;
  created_at: string; updated_at: string;
}
export interface EngineeringReportRecord {
  report_id: string; investigation_id: string; project_id: string;
  content_hash: string; report: Record<string, unknown>; relative_path: string;
  status: "draft" | "qualified"; created_at: string;
}
export interface VariantProposalRecord {
  proposal_id: string; investigation_id: string; finding_id: string; project_id: string;
  name: string; proposal_kind: "input_patch" | "run_configuration" | "checkpoint_branch" | "experiment";
  content_hash: string; proposal: Record<string, unknown>;
  status: "proposed" | "approved" | "rejected"; decision_note: string;
  created_at: string; updated_at: string;
}
export interface InvestigationPlanRecord {
  plan_id: string; investigation_id: string; project_id: string;
  provider: string; content_hash: string;
  plan: { plan?: { calls?: Array<{ call_id: string; tool: string; arguments: Record<string, unknown> }> } };
  created_at: string;
}
export interface InvestigationExecutionRecord {
  execution_id: string; investigation_id: string; plan_id: string; project_id: string;
  plan_hash: string; content_hash: string; status: string;
  execution: { calls?: Array<{ call: { call_id: string; tool: string }; result: { ok: boolean; error?: string | null } }> };
  created_at: string; completed_at: string;
}
export interface InvestigationSynthesisRecord {
  synthesis_id: string; investigation_id: string; project_id: string;
  finding_set_hash: string; decision_set_hash: string; content_hash: string;
  synthesis: {
    finding_ids: string[];
    summary: Record<string, number | Record<string, number>>;
    conflicts: Array<{
      conflict_id: string; finding_ids: [string, string]; category: string;
      shared_objects: string[]; reason: string;
      resolution: null | { action: string; reviewer: string; rationale: string; selected_finding_ids: string[] };
    }>;
    uncertainties: Array<{ finding_id: string; uncertainty_kind: string; confidence: number; explanation: string }>;
    corroborations: Array<{ corroboration_id: string; finding_ids: string[]; category: string; reason: string }>;
  };
  created_at: string;
}
export interface InvestigationNarrativeRecord {
  narrative_id: string; investigation_id: string; project_id: string;
  provider: string; provider_configuration_hash: string;
  source_findings_hash: string; content_hash: string;
  narrative: {
    title: string; executive_summary: string; status: "draft" | "approved" | "rejected";
    sections: Array<{
      section_id: string; heading: string; narrative: string; finding_ids: string[];
      evidence: Array<{ kind: string; uri: string; label: string; checksum?: string | null }>;
      uncertainty_caveats: string[];
    }>;
  };
  status: "draft" | "approved" | "rejected"; reviewer: string; review_note: string;
  created_at: string; updated_at: string;
}
export interface ReasoningProviderRecord {
  provider_id: string; kind: string; model: string; endpoint: string;
  credential_env: string; enabled: boolean; content_hash: string;
  configuration: Record<string, unknown>; created_at: string; updated_at: string;
}
export interface ReasoningProviderCheckRecord {
  check_id: string; provider_id: string; status: "passed" | "failed";
  latency_seconds: number; configuration_hash: string;
  details: Record<string, unknown>; created_at: string;
}
export interface ReasoningProviderEvaluationRecord {
  evaluation_id: string; provider_id: string; configuration_hash: string;
  suite_hash: string; status: "qualified" | "failed"; content_hash: string;
  report: {
    summary: {
      fixture_count: number; passed_count: number; failed_count: number;
      pass_rate: number; mean_f1: number; total_latency_seconds: number;
      reported_total_tokens: number | null;
    };
    results: Array<{ fixture_id: string; status: string; f1: number; latency_seconds: number; error?: string }>;
  };
  created_at: string;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

let accessToken = typeof sessionStorage === "undefined"
  ? "" : sessionStorage.getItem("pymrst.access_token") ?? "";

export function setAccessToken(value: string) {
  accessToken = value.trim();
  if (typeof sessionStorage !== "undefined") {
    if (accessToken) sessionStorage.setItem("pymrst.access_token", accessToken);
    else sessionStorage.removeItem("pymrst.access_token");
  }
}

export function hasAccessToken() { return Boolean(accessToken); }
export function getAccessToken() { return accessToken; }

let apiBaseUrl = "";
let desktopToken = "";

export function configureDesktopBackend(endpoint: string, token: string) {
  const normalized = endpoint.trim().replace(/\/$/, "");
  if (normalized && !/^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(normalized)) {
    throw new Error("Desktop backend must use a loopback HTTP endpoint");
  }
  if (normalized && token.trim().length < 32) {
    throw new Error("Desktop backend supplied an invalid bootstrap credential");
  }
  apiBaseUrl = normalized; desktopToken = token.trim();
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
      ...(desktopToken ? { "X-PyMRST-Desktop-Token": desktopToken } : {}),
      ...init.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(response.status, body.detail ?? `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  environmentQualification: (signal?: AbortSignal) =>
    request<EnvironmentQualification | null>("/api/v1/system/environment-qualification", { signal }),
  runEnvironmentQualification: () =>
    request<EnvironmentQualification>("/api/v1/system/environment-qualifications", { method: "POST" }),
  examples: (signal?: AbortSignal) =>
    request<ExampleDefinition[]>("/api/v1/examples", { signal }),
  workloadCatalog: (signal?: AbortSignal) =>
    request<WorkloadDefinition[]>("/api/v1/system/workload-catalog", { signal }),
  workloadQualifications: (limit = 50, signal?: AbortSignal) =>
    request<WorkloadQualification[]>(`/api/v1/system/workload-qualifications?limit=${limit}`, { signal }),
  workloadQualification: (qualificationId: string, signal?: AbortSignal) =>
    request<WorkloadQualification>(`/api/v1/system/workload-qualifications/${qualificationId}`, { signal }),
  submitWorkloadQualification: (payload: { workload_id: string; tier: WorkloadTier; seed: number; budget: WorkloadBudget | null; warmup_runs: number; measured_runs: number }) =>
    request<WorkloadQualification>("/api/v1/system/workload-qualifications", {
      method: "POST", body: JSON.stringify(payload),
    }),
  cancelWorkloadQualification: (qualificationId: string) =>
    request<WorkloadQualification>(`/api/v1/system/workload-qualifications/${qualificationId}/cancel`, { method: "POST" }),
  verifyWorkloadQualification: (qualificationId: string) =>
    request<{ qualification_id: string; valid: boolean; reason: string; content_hash: string; expected_content_hash?: string }>(`/api/v1/system/workload-qualifications/${qualificationId}/verify`),
  securityMode: (signal?: AbortSignal) =>
    request<{ mode: "local" | "shared" }>("/api/v1/security/mode", { signal }),
  setSecurityMode: (mode: "local" | "shared") =>
    request<{ mode: "local" | "shared" }>("/api/v1/security/mode", {
      method: "PUT", body: JSON.stringify({ mode }),
    }),
  securityIdentity: (signal?: AbortSignal) =>
    request<SecurityActor>("/api/v1/security/me", { signal }),
  securityUsers: (signal?: AbortSignal) =>
    request<SecurityUser[]>("/api/v1/security/users", { signal }),
  createSecurityUser: (payload: { display_name: string; email: string; system_role: SecurityUser["system_role"] }) =>
    request<SecurityUser>("/api/v1/security/users", { method: "POST", body: JSON.stringify(payload) }),
  setSecurityUserActive: (userId: string, active: boolean) =>
    request<SecurityUser>(`/api/v1/security/users/${userId}/active`, {
      method: "POST", body: JSON.stringify({ active }),
    }),
  securityTokens: (userId: string, signal?: AbortSignal) =>
    request<SecurityToken[]>(`/api/v1/security/tokens?user_id=${encodeURIComponent(userId)}`, { signal }),
  issueSecurityToken: (userId: string, payload: { label: string; scopes: Capability[]; expires_at: string }) =>
    request<SecurityToken>(`/api/v1/security/users/${userId}/tokens`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  revokeSecurityToken: (tokenId: string) =>
    request<SecurityToken>(`/api/v1/security/tokens/${tokenId}/revoke`, { method: "POST" }),
  setProjectMembership: (projectId: string, userId: string, projectRole: ProjectMembership["project_role"]) =>
    request<ProjectMembership>(`/api/v1/security/projects/${projectId}/members/${userId}`, {
      method: "PUT", body: JSON.stringify({ project_role: projectRole }),
    }),
  securityAudit: (limit = 20, signal?: AbortSignal) =>
    request<SecurityAuditEvent[]>(`/api/v1/security/audit?limit=${limit}`, { signal }),
  verifySecurityAudit: (signal?: AbortSignal) =>
    request<{ valid: boolean; events: number; head_hash?: string; reason?: string }>(
      "/api/v1/security/audit/verify", { signal },
    ),
  backupPlan: (signal?: AbortSignal) =>
    request<BackupPlan>("/api/v1/system/backups/plan", { signal }),
  backups: (signal?: AbortSignal) =>
    request<BackupRecord[]>("/api/v1/system/backups", { signal }),
  createBackup: () => request<BackupRecord>("/api/v1/system/backups", { method: "POST" }),
  verifyBackup: (backupId: string) =>
    request<BackupRecord>(`/api/v1/system/backups/${backupId}/verify`, { method: "POST" }),
  restoreBackup: (backupId: string, targetRoot: string, confirmArchiveHash: string) =>
    request<RestoreReceipt>(`/api/v1/system/backups/${backupId}/restore`, {
      method: "POST", body: JSON.stringify({ target_root: targetRoot, confirm_archive_hash: confirmArchiveHash }),
    }),
  backupJobs: (signal?: AbortSignal) =>
    request<BackupJobRecord[]>("/api/v1/system/backup-jobs", { signal }),
  submitBackupJob: () =>
    request<BackupJobRecord>("/api/v1/system/backup-jobs", { method: "POST" }),
  backupJob: (jobId: string, signal?: AbortSignal) =>
    request<BackupJobRecord>(`/api/v1/system/backup-jobs/${jobId}`, { signal }),
  cancelBackupJob: (jobId: string) =>
    request<BackupJobRecord>(`/api/v1/system/backup-jobs/${jobId}/cancel`, { method: "POST" }),
  backupPolicy: (signal?: AbortSignal) =>
    request<BackupPolicy>("/api/v1/system/backups/policy", { signal }),
  saveBackupPolicy: (policy: BackupPolicy) =>
    request<BackupPolicy>("/api/v1/system/backups/policy", {
      method: "PUT", body: JSON.stringify(policy),
    }),
  backupQualifications: (backupId?: string, signal?: AbortSignal) =>
    request<BackupQualificationRecord[]>(
      `/api/v1/system/backup-qualifications${backupId ? `?backup_id=${encodeURIComponent(backupId)}` : ""}`,
      { signal },
    ),
  qualifyBackup: (backupId: string) =>
    request<BackupQualificationRecord>(`/api/v1/system/backups/${backupId}/qualify`, { method: "POST" }),
  reasoningProviders: (signal?: AbortSignal) =>
    request<ReasoningProviderRecord[]>("/api/v1/reasoning-providers", { signal }),
  registerReasoningProvider: (payload: Record<string, unknown>) =>
    request<ReasoningProviderRecord>("/api/v1/reasoning-providers", {
      method: "POST", body: JSON.stringify(payload),
    }),
  setReasoningProviderEnabled: (providerId: string, enabled: boolean) =>
    request<ReasoningProviderRecord>(`/api/v1/reasoning-providers/${providerId}/enabled`, {
      method: "POST", body: JSON.stringify({ enabled }),
    }),
  reasoningProviderChecks: (providerId: string, signal?: AbortSignal) =>
    request<ReasoningProviderCheckRecord[]>(`/api/v1/reasoning-providers/${providerId}/checks`, { signal }),
  checkReasoningProvider: (providerId: string) =>
    request<ReasoningProviderCheckRecord>(`/api/v1/reasoning-providers/${providerId}/check`, {
      method: "POST", body: "{}",
    }),
  reasoningProviderEvaluations: (providerId?: string, signal?: AbortSignal) =>
    request<ReasoningProviderEvaluationRecord[]>(
      `/api/v1/reasoning-provider-evaluations${providerId ? `?provider_id=${encodeURIComponent(providerId)}` : ""}`,
      { signal },
    ),
  evaluateReasoningProvider: (providerId: string) =>
    request<ReasoningProviderEvaluationRecord>(`/api/v1/reasoning-providers/${providerId}/evaluate`, {
      method: "POST", body: "{}",
    }),
  health: (signal?: AbortSignal) => request<HealthRecord>("/api/v1/health", { signal }),
  projects: (signal?: AbortSignal) => request<ProjectRecord[]>("/api/v1/projects", { signal }),
  notifications: (signal?: AbortSignal) =>
    request<NotificationRecord[]>("/api/v1/notifications?unread_only=true", { signal }),
  createProject: (payload: { name: string; description: string; tags: string[] }) =>
    request<ProjectRecord>("/api/v1/projects", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  assets: (projectId: string, signal?: AbortSignal) =>
    request<ProjectAsset[]>(`/api/v1/projects/${projectId}/assets`, { signal }),
  importAsset: (projectId: string, payload: { source_path: string; kind: string }) =>
    request<ProjectAsset>(`/api/v1/projects/${projectId}/assets`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  models: (projectId: string, signal?: AbortSignal) =>
    request<ModelInspection[]>(`/api/v1/projects/${projectId}/models`, { signal }),
  inspectModel: (projectId: string, payload: { asset_id: string; name?: string }) =>
    request<ModelInspection>(`/api/v1/projects/${projectId}/models`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  model: (modelId: string, signal?: AbortSignal) =>
    request<ModelInspection>(`/api/v1/models/${modelId}`, { signal }),
  source: (modelId: string, path: string, startLine = 1, lineCount = 500, signal?: AbortSignal) =>
    request<SourceSlice>(`/api/v1/models/${modelId}/source?${new URLSearchParams({
      path, start_line: String(startLine), line_count: String(lineCount),
    })}`, { signal }),
  grids: (projectId: string, signal?: AbortSignal) =>
    request<GridSnapshot[]>(`/api/v1/projects/${projectId}/grids`, { signal }),
  generateGrid: (projectId: string, payload: { name: string; dims: [number, number, number]; physical_dims_m: [number, number, number] }) =>
    request<GridSnapshot>(`/api/v1/projects/${projectId}/grids/generate`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  gridValues: (snapshotId: string, arrayName: string, offset = 0, count = 25000, signal?: AbortSignal, stride = 1) =>
    request<GridValues>(`/api/v1/grids/${snapshotId}/arrays/${arrayName}?${new URLSearchParams({
      offset: String(offset), count: String(count), stride: String(stride),
    })}`, { signal }),
  patches: (modelId: string, signal?: AbortSignal) =>
    request<InputPatch[]>(`/api/v1/models/${modelId}/patches`, { signal }),
  createPatch: (modelId: string, payload: { title: string; description: string; operations: InputPatch["operations"] }) =>
    request<InputPatch>(`/api/v1/models/${modelId}/patches`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  patchPreview: (patchId: string, signal?: AbortSignal) =>
    request<PatchPreview>(`/api/v1/patches/${patchId}/preview`, { signal }),
  materializePatch: (patchId: string) =>
    request<InputPatch>(`/api/v1/patches/${patchId}/materialize`, { method: "POST" }),
  modelDrafts: (projectId: string, signal?: AbortSignal) =>
    request<ModelDraft[]>(`/api/v1/projects/${projectId}/model-drafts`, { signal }),
  createModelDraft: (projectId: string, payload: { name: string; grid_snapshot_id: string; document: ModelConstructionDocument }) =>
    request<ModelDraft>(`/api/v1/projects/${projectId}/model-drafts`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  updateModelDraft: (draftId: string, payload: { expected_version: number; reason: string; document: ModelConstructionDocument }) =>
    request<ModelDraft>(`/api/v1/model-drafts/${draftId}`, {
      method: "PUT", body: JSON.stringify(payload),
    }),
  validateModelDraft: (draftId: string, document?: ModelConstructionDocument) =>
    request<{ draft_id: string; valid: boolean; issues: ConstructionIssue[] }>(`/api/v1/model-drafts/${draftId}/validate`, {
      method: "POST", body: JSON.stringify(document ? { document } : {}),
    }),
  modelDraftHistory: (draftId: string, signal?: AbortSignal) =>
    request<Array<{ draft_id: string; version: number; document_hash: string; actor: string; reason: string; created_at: string }>>(`/api/v1/model-drafts/${draftId}/history`, { signal }),
  modelRevisions: (draftId: string, signal?: AbortSignal) =>
    request<ModelRevision[]>(`/api/v1/model-drafts/${draftId}/revisions`, { signal }),
  projectModelRevisions: (projectId: string, signal?: AbortSignal) =>
    request<ModelRevision[]>(`/api/v1/projects/${projectId}/model-revisions`, { signal }),
  materializeModelDraft: (draftId: string) =>
    request<ModelRevision>(`/api/v1/model-drafts/${draftId}/materialize`, { method: "POST" }),
  experiments: (projectId: string, signal?: AbortSignal) =>
    request<ExperimentRecord[]>(`/api/v1/projects/${projectId}/experiments`, { signal }),
  createExperiment: (projectId: string, payload: Record<string, unknown>) =>
    request<ExperimentRecord>(`/api/v1/projects/${projectId}/experiments`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  experimentExecutions: (experimentId: string, signal?: AbortSignal) =>
    request<ExperimentExecution[]>(`/api/v1/experiments/${experimentId}/executions`, { signal }),
  createExperimentExecution: (experimentId: string, payload: Record<string, unknown>) =>
    request<ExperimentExecution>(`/api/v1/experiments/${experimentId}/executions`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  experimentExecution: (executionId: string, signal?: AbortSignal) =>
    request<ExperimentExecution>(`/api/v1/experiment-executions/${executionId}`, { signal }),
  cancelExperimentExecution: (executionId: string) =>
    request<ExperimentExecution>(`/api/v1/experiment-executions/${executionId}/cancel`, { method: "POST" }),
  qualifications: (experimentId: string, signal?: AbortSignal) =>
    request<QualificationRecord[]>(`/api/v1/experiments/${experimentId}/qualifications`, { signal }),
  createQualification: (executionId: string, payload: Record<string, unknown>) =>
    request<QualificationRecord>(`/api/v1/experiment-executions/${executionId}/qualifications`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  branches: (experimentId: string, signal?: AbortSignal) =>
    request<BranchRecord[]>(`/api/v1/experiments/${experimentId}/branches`, { signal }),
  createBranch: (executionId: string, payload: Record<string, unknown>) =>
    request<BranchRecord>(`/api/v1/experiment-executions/${executionId}/branches`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  branch: (branchId: string, signal?: AbortSignal) =>
    request<BranchRecord>(`/api/v1/experiment-branches/${branchId}`, { signal }),
  investigations: (projectId: string, signal?: AbortSignal) =>
    request<InvestigationRecord[]>(`/api/v1/projects/${projectId}/investigations`, { signal }),
  createInvestigation: (projectId: string, payload: Record<string, unknown>) =>
    request<InvestigationRecord>(`/api/v1/projects/${projectId}/investigations`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  findings: (investigationId: string, signal?: AbortSignal) =>
    request<FindingRecord[]>(`/api/v1/investigations/${investigationId}/findings`, { signal }),
  investigationPlans: (investigationId: string, signal?: AbortSignal) =>
    request<InvestigationPlanRecord[]>(`/api/v1/investigations/${investigationId}/plans`, { signal }),
  investigationExecutions: (investigationId: string, signal?: AbortSignal) =>
    request<InvestigationExecutionRecord[]>(`/api/v1/investigations/${investigationId}/executions`, { signal }),
  investigationSyntheses: (investigationId: string, signal?: AbortSignal) =>
    request<InvestigationSynthesisRecord[]>(`/api/v1/investigations/${investigationId}/syntheses`, { signal }),
  investigationNarratives: (investigationId: string, signal?: AbortSignal) =>
    request<InvestigationNarrativeRecord[]>(`/api/v1/investigations/${investigationId}/narratives`, { signal }),
  draftInvestigationNarrative: (investigationId: string, provider = "deterministic-v1") =>
    request<InvestigationNarrativeRecord>(`/api/v1/investigations/${investigationId}/narratives`, {
      method: "POST", body: JSON.stringify({ provider }),
    }),
  reviewInvestigationNarrative: (
    narrativeId: string, status: "approved" | "rejected", reviewer: string, note: string,
  ) => request<InvestigationNarrativeRecord>(`/api/v1/investigation-narratives/${narrativeId}/review`, {
    method: "POST", body: JSON.stringify({ status, reviewer, note }),
  }),
  resolveInvestigationConflict: (
    investigationId: string, conflictId: string,
    payload: { action: "resolve" | "dismiss"; reviewer: string; rationale: string; selected_finding_ids?: string[] },
  ) => request<InvestigationSynthesisRecord>(
    `/api/v1/investigations/${investigationId}/conflicts/${conflictId}/resolve`,
    { method: "POST", body: JSON.stringify(payload) },
  ),
  planInvestigation: (investigationId: string, provider = "deterministic-v1") =>
    request<InvestigationPlanRecord>(`/api/v1/investigations/${investigationId}/plan`, {
      method: "POST", body: JSON.stringify({ provider, maximum_calls: 32 }),
    }),
  analyzeInvestigation: (investigationId: string, planId?: string) =>
    request<FindingRecord[]>(`/api/v1/investigations/${investigationId}/analyze`, {
      method: "POST", body: JSON.stringify(planId ? { plan_id: planId } : {}),
    }),
  reviewFinding: (findingId: string, status: "accepted" | "rejected", note = "") =>
    request<FindingRecord>(`/api/v1/findings/${findingId}/review`, {
      method: "POST", body: JSON.stringify({ status, note }),
    }),
  engineeringReports: (projectId: string, signal?: AbortSignal) =>
    request<EngineeringReportRecord[]>(`/api/v1/projects/${projectId}/engineering-reports`, { signal }),
  buildEngineeringReport: (investigationId: string, title: string, qualified = false) =>
    request<EngineeringReportRecord>(`/api/v1/investigations/${investigationId}/reports`, {
      method: "POST", body: JSON.stringify({ title, qualified }),
    }),
  variantProposals: (investigationId: string, signal?: AbortSignal) =>
    request<VariantProposalRecord[]>(`/api/v1/investigations/${investigationId}/variant-proposals`, { signal }),
  createVariantProposal: (findingId: string, payload: Record<string, unknown>) =>
    request<VariantProposalRecord>(`/api/v1/findings/${findingId}/variant-proposals`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  decideVariantProposal: (proposalId: string, status: "approved" | "rejected", note: string) =>
    request<VariantProposalRecord>(`/api/v1/variant-proposals/${proposalId}/decision`, {
      method: "POST", body: JSON.stringify({ status, note }),
    }),
  runConfigurations: (projectId: string, signal?: AbortSignal) =>
    request<RunConfiguration[]>(`/api/v1/projects/${projectId}/run-configurations`, { signal }),
  createRunConfiguration: (projectId: string, payload: { name: string; runner_kind: string; model_id: string; payload: Record<string, unknown> }) =>
    request<RunConfiguration>(`/api/v1/projects/${projectId}/run-configurations`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  runs: (projectId: string, signal?: AbortSignal) =>
    request<RunRecord[]>(`/api/v1/projects/${projectId}/runs`, { signal }),
  run: (runId: string, signal?: AbortSignal) => request<RunRecord>(`/api/v1/runs/${runId}`, { signal }),
  submitRun: (configurationId: string) => request<RunRecord>(`/api/v1/run-configurations/${configurationId}/submit`, { method: "POST" }),
  runEvents: (runId: string, afterSequence = -1, signal?: AbortSignal) =>
    request<RunEvent[]>(`/api/v1/runs/${runId}/events?${new URLSearchParams({ after_sequence: String(afterSequence) })}`, { signal }),
  checkpoints: (runId: string, signal?: AbortSignal) => request<RunCheckpoint[]>(`/api/v1/runs/${runId}/checkpoints`, { signal }),
  cancelRun: (runId: string) => request<Record<string, unknown>>(`/api/v1/runs/${runId}/cancel`, { method: "POST" }),
  resumeRun: (runId: string, checkpointId: string) => request<RunRecord>(`/api/v1/runs/${runId}/resume`, { method: "POST", body: JSON.stringify({ checkpoint_id: checkpointId }) }),
  resultStates: (runId: string, signal?: AbortSignal) => request<ResultState[]>(`/api/v1/runs/${runId}/states`, { signal }),
  resultField: (stateId: string, field: string, offset = 0, count = 25000, signal?: AbortSignal, stride = 1) => request<GridValues>(`/api/v1/result-states/${stateId}/fields/${field}?${new URLSearchParams({ offset: String(offset), count: String(count), stride: String(stride) })}`, { signal }),
  dashboard: (runId: string, signal?: AbortSignal) => request<ResultDashboard>(`/api/v1/runs/${runId}/dashboard`, { signal }),
  compareRuns: (referenceRunId: string, candidateRunId: string) => request<RunComparison>("/api/v1/run-comparisons", { method: "POST", body: JSON.stringify({ reference_run_id: referenceRunId, candidate_run_id: candidateRunId }) }),
  difference: (referenceRunId: string, candidateRunId: string, reportStep: number, field: string, offset = 0, count = 25000, signal?: AbortSignal) => request<GridValues & { reference_run_id: string; candidate_run_id: string; report_step: number }>(`/api/v1/run-comparisons/difference?${new URLSearchParams({ reference_run_id: referenceRunId, candidate_run_id: candidateRunId, report_step: String(reportStep), field, offset: String(offset), count: String(count) })}`, { signal }),
  animation: (runId: string, field: string, signal?: AbortSignal) => request<{ schema_version: number; run_id: string; field: string; frames: Array<{ state_id: string; report_step: number; simulated_time_seconds: number; field_url: string }> }>(`/api/v1/runs/${runId}/animations/${field}`, { signal }),
  exportField: (stateId: string, field: string, kind: "csv" | "json" = "csv") => request<{ export_id: string; relative_path: string; content_hash: string }>(`/api/v1/result-states/${stateId}/exports`, { method: "POST", body: JSON.stringify({ field, kind }) }),
};
