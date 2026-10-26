# Other PyMRST Improvements

## Agentic Engineering and a Professional Simulation Application

## 1. Purpose

This document describes two complementary extensions for PyMRST:

1. An evidence-aware agentic engineering system that can inspect reservoir models, execute controlled studies, diagnose simulations, compare results, and prepare technical reports.
2. A robust, installable application—provisionally named **PyMRST Studio**—with the model-building, simulation, visualization, and reporting workflow expected from professional reservoir simulators.

These capabilities extend PyMRST without changing its core identity as a Python implementation of MRST. The deterministic numerical implementation remains authoritative. Agents and user-interface services operate around that implementation through stable, typed interfaces.

The intended workflow is:

```text
Project -> Model -> Validate -> Initialize -> Schedule -> Run
        -> Monitor -> Analyze -> Compare -> Qualify -> Report
```

## 2. Design Principles

### 2.1 Preserve the MRST-compatible numerical core

- Keep MRST-equivalent functions and module layouts in their existing owners.
- Do not insert language-model decisions into residual, Jacobian, property, flash, timestep, or linear-solver calculations.
- Make application and agent capabilities optional layers.
- Keep every simulation runnable from Python and the command line without the UI or an AI provider.
- Preserve direct PyMRST-to-MRST validation paths.

### 2.2 Deterministic computation, assisted interpretation

Numerical tools calculate quantities such as material balance, convergence metrics, grid quality, well performance, and spatial statistics. Agents select these tools, interpret their outputs, organize investigations, and explain conclusions. An agent must never fabricate a numerical result or use language-model arithmetic in place of a deterministic calculation.

### 2.3 Evidence before conclusions

Every engineering claim should be traceable to one or more of:

- An input file, keyword, record, and line number.
- A grid object, region, cell, face, NNC, fracture, well, completion, or segment.
- A simulation identifier, report step, timestep, or nonlinear iteration.
- A stored state, summary vector, convergence record, plot, or table.
- An analytical, MRST, OPM, laboratory, or field-history reference.

### 2.4 Immutable inputs and reproducible variants

- Never silently modify an imported model.
- Express changes as typed, reviewable patches.
- Assign stable identifiers and content hashes to models, variants, runs, and reports.
- Store the exact simulator, solver, hardware, thread, unit, and dependency configuration used by every run.

### 2.5 Friendly for new users, complete for experts

The application should offer a guided engineering workflow and an expert workflow over the same underlying project representation. Guided mode must not create a simplified or incompatible model format.

## 3. Agentic PyMRST

### 3.1 Value proposition

An agentic PyMRST can reduce the time spent manually navigating decks, logs, state arrays, plots, and scenario directories. It is particularly valuable for:

- Model-input quality assurance.
- Simulation readiness reviews.
- Convergence and timestep-failure diagnosis.
- Material-balance and physical-invariant checks.
- Reservoir and well-performance analysis.
- Scenario, sensitivity, and uncertainty studies.
- PyMRST/MRST/OPM comparison and disagreement analysis.
- Reproducible research investigations.
- Technical report preparation.

A representative request could be:

> Review this deck, identify input risks, initialize and run the model, diagnose any convergence problems, compare three development scenarios, evaluate material balance, and prepare a traceable technical report.

### 3.2 Proposed layout

```text
pymrst/
  research_studio/
    agents/
      orchestrator.py
      base.py
      context.py
      memory.py
      permissions.py

      specialists/
        deck_review.py
        grid_quality.py
        rock_fluid.py
        initialization.py
        wells_schedule.py
        numerical_diagnostics.py
        reservoir_performance.py
        conservation.py
        uncertainty.py
        history_matching.py
        economics.py
        reporting.py

      tools/
        deck_tools.py
        case_tools.py
        simulation_tools.py
        result_tools.py
        plotting_tools.py
        validation_tools.py
        experiment_tools.py
        report_tools.py

      schemas/
        findings.py
        evidence.py
        recommendations.py
        run_request.py
        run_result.py
        engineering_report.py

      policies/
        execution.py
        mutation.py
        resource_limits.py
        evidence.py
```

This layer should reuse the existing Research Studio subsystems:

- `runtime/` for sessions, controls, cancellation, and compatibility.
- `evidence/` for manifests, events, convergence, conservation, invariants, and failure bundles.
- `storage/` for results, checkpoints, HDF5/Zarr data, and migrations.
- `validation/` for analytical, MATLAB, OPM, property, metamorphic, and benchmark evidence.
- `experiments/` for variants, branching, ensembles, execution, comparison, and publishing.
- `components/` for registered models, solvers, capabilities, surrogates, and fallbacks.
- `studio/` for browser, notebook, and report presentation.

### 3.3 Orchestrator agent

The orchestrator converts an engineering question into a bounded investigation graph. It determines:

- Which specialist agents and deterministic tools are required.
- Dependencies among inspection, simulation, comparison, and reporting tasks.
- What can execute concurrently.
- Resource budgets and stopping conditions.
- Whether a proposed input change or expensive run requires approval.
- Whether the available evidence supports the requested conclusion.

The orchestrator must not directly invent results or modify source models.

### 3.4 Specialist agents

#### Deck-review agent

- Inspect `RUNSPEC`, `GRID`, `EDIT`, `PROPS`, `REGIONS`, `SOLUTION`, `SUMMARY`, and `SCHEDULE`.
- Resolve and audit included files.
- Verify units, dimensions, array lengths, defaulted values, and active-cell mappings.
- Identify unsupported keywords and incompatible keyword combinations.
- Cite file, keyword, record, and source line for every finding.

#### Grid-quality agent

- Detect negative or extremely small volumes.
- Analyze aspect ratio, skewness, pinch-outs, faults, and inactive cells.
- Identify disconnected regions and suspicious transmissibilities.
- Inspect NNC topology and fracture connections.
- Detect completions in inactive, isolated, or unsuitable cells.

#### Rock-and-fluid agent

- Inspect porosity, permeability, net-to-gross, compressibility, and region assignments.
- Validate PVT-table ordering, interpolation, continuity, and extrapolation.
- Validate relative-permeability and capillary-pressure endpoints.
- Check saturation-function normalization and phase consistency.
- Inspect EOS components, interactions, phase envelopes, and flash behavior.

#### Initialization agent

- Validate contacts, datum definitions, equilibration regions, and hydrostatic trends.
- Check saturation closure and phase appearance.
- Verify Rs/Rv and composition consistency.
- Calculate pore volume and hydrocarbons initially in place.
- Compare initialized fields with requested physical constraints.

#### Well-and-schedule agent

- Validate trajectories, completions, well indices, and segment topology.
- Inspect rate/BHP controls, limits, crossflow, and control switching.
- Detect duplicate or contradictory completions.
- Inspect schedule discontinuities and group constraints.
- Explain the active control state at any requested date.

#### Numerical-diagnostics agent

- Analyze Newton, linear-solver, CNV, material-balance, and update histories.
- Find cells, equations, phases, wells, or controls dominating nonlinearity.
- Analyze timestep cuts, restarts, oscillations, and stagnation.
- Separate physical difficulty from initialization, schedule, grid, solver, or implementation problems.
- Propose bounded diagnostic experiments rather than arbitrary tolerance relaxation.

#### Reservoir-performance agent

- Analyze pressure depletion, production, injection, GOR, water cut, and cumulative volumes.
- Evaluate voidage replacement, recovery, sweep, breakthrough, and remaining hydrocarbons.
- Analyze well interference, regional balance, constraints, and facility bottlenecks.
- Produce cell-, region-, well-, group-, and field-level evidence.

#### Experiment-and-uncertainty agent

- Create controlled timestep, grid, solver, PVT, relperm, aquifer, fault, and well-control studies.
- Execute deterministic ensembles with explicit seeds.
- Perform parameter screening and response analysis.
- Preserve failed realizations and their evidence rather than silently discarding them.

#### Reporting agent

- Assemble executive and technical summaries.
- Separate observations, calculations, inferences, uncertainties, and recommendations.
- Include evidence-linked plots, tables, manifests, warnings, and assumptions.
- Generate reproducible Markdown, HTML, PDF, and Word-oriented outputs.

### 3.5 Structured findings

Agents should exchange typed records instead of unstructured prose:

```python
@dataclass(frozen=True)
class Finding:
    finding_id: str
    agent: str
    category: str
    severity: Literal["info", "warning", "error", "critical"]
    statement: str
    evidence: tuple[EvidenceReference, ...]
    confidence: float
    affected_objects: tuple[str, ...]
    recommendation: str | None
    requires_confirmation: bool
```

Reports and the UI must distinguish:

- **Observed:** directly read from an input or result.
- **Calculated:** produced by a deterministic PyMRST tool.
- **Inferred:** an engineering interpretation of observations and calculations.
- **Recommended:** a proposed action or experiment.
- **Uncertain:** a conclusion sensitive to missing data or assumptions.

### 3.6 Constrained agent tools

Agents should use a typed engineering API rather than unrestricted filesystem or shell access:

```python
inspect_deck(case_id)
get_keyword(case_id, keyword)
validate_case(case_id)
summarize_grid(case_id)
inspect_cells(case_id, cell_ids)
inspect_well(case_id, well_name)
create_variant(case_id, patch)
run_case(run_request)
get_convergence(run_id)
get_states(run_id, fields, steps)
compute_material_balance(run_id)
compare_runs(run_ids, metrics)
create_plot(plot_request)
publish_report(report_request)
```

Input changes should use declarative patches:

```python
CasePatch(
    reason="Evaluate uncertainty in fault communication",
    operations=[SetFaultMultiplier(fault="FAULT_3", value=0.05)],
)
```

### 3.7 Permissions and resource governance

Suggested permission levels are:

1. Inspect inputs and existing results.
2. Calculate derived metrics, validations, and plots.
3. Create immutable case variants.
4. Run bounded simulations.
5. Launch expensive ensembles or remote jobs.
6. Publish or transmit reports.

Policies should enforce:

- Maximum cells, timesteps, realizations, workers, memory, and runtime.
- Model, backend, solver, and runner allowlists.
- Cancellation and checkpoint requirements.
- Isolated output directories.
- Explicit approval for expensive or externally visible actions.
- Complete tool-call, decision, and evidence logs.

### 3.8 Provider independence

The agent system should not depend on a single language-model provider:

```python
class ReasoningProvider(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        response_schema: type[T],
    ) -> T:
        ...
```

Adapters may support cloud providers, local models through Ollama/vLLM, and deterministic rule-based execution. All simulation, validation, storage, and reporting calculations must remain usable without an AI provider.

## 4. PyMRST Studio

### 4.1 Product vision

PyMRST Studio should be a polished, installable reservoir-engineering application rather than a thin web page around Python. It should offer familiar commercial-simulator workflows while distinguishing itself through Python extensibility, MRST compatibility, reproducible research, open validation, and evidence-aware agents.

The application should support:

- Native desktop use on macOS, Windows, and Linux.
- Browser access for remote workstations and team deployments.
- Local, remote, container, and HPC execution.
- Guided and expert workflows.
- Large-model visualization and lazy data access.
- Cross-simulator qualification and research experiments.

### 4.2 Recommended technology

#### Desktop shell

Use Tauri 2 for native installers, menus, file dialogs, notifications, updates, and secure local-service integration. Tauri provides a smaller desktop footprint than Electron while permitting a modern TypeScript frontend.

#### Frontend

- React and TypeScript for the application.
- A custom PyMRST design system over Material UI or Fluent UI.
- AG Grid for large property, schedule, well, and summary tables.
- Plotly for interactive engineering charts.
- vtk.js for GPU-accelerated reservoir visualization.
- Monaco Editor for ECLIPSE decks, JSON/YAML configuration, and Python.
- TanStack Query for API/server state.
- Zustand or an equivalent small store for transient workspace state.

#### Backend

- FastAPI for a typed local and remote application API.
- Pydantic for request, event, and response schemas.
- Isolated worker processes for simulations.
- WebSockets or server-sent events for progress and logs.
- SQLite for project metadata and application state.
- HDF5 or Zarr for grids and simulator states.
- Parquet for summaries, convergence histories, and tabular analytics.

### 4.3 Proposed application layout

```text
pymrst/
  app/
    api/
      projects.py
      models.py
      validation.py
      simulation.py
      results.py
      experiments.py
      agents.py
      reports.py

    services/
      project_service.py
      case_service.py
      run_service.py
      result_service.py
      report_service.py

    schemas/
      project.py
      model.py
      schedule.py
      run.py
      result.py

    workers/
      local.py
      subprocess.py
      slurm.py
      remote.py

    database/
      models.py
      migrations/

ui/
  src/
    application/
    project/
    model/
    grid/
    properties/
    fluids/
    initialization/
    wells/
    schedule/
    simulation/
    results/
    experiments/
    agents/
    reports/
```

### 4.4 Main interface

The main engineering workspace should use a persistent project tree, central workspace, contextual inspector, and bottom task/output panel:

```text
+------------------------------------------------------------------+
| Project  Model  Simulation  Results  Experiments  Reports        |
+--------------+------------------------------------+--------------+
| PROJECT TREE | ENGINEERING WORKSPACE              | INSPECTOR    |
|              |                                    |              |
| Model        | Grid / chart / table / editor     | Selection    |
| |- Grid      |                                    | Properties   |
| |- Rock      |                                    | Validation   |
| |- Fluids    |                                    | Evidence     |
| |- Wells     |                                    |              |
| `- Schedule  |                                    |              |
+--------------+------------------------------------+--------------+
| Problems | Output | Convergence | Tasks | Agent Investigations   |
+------------------------------------------------------------------+
```

Selection should remain synchronized across the project tree, tables, charts, 3D views, schedule, results, evidence, and agent findings.

## 5. Core Application Workspaces

### 5.1 Project manager

- Create, import, duplicate, branch, archive, and compare projects and cases.
- Maintain immutable imported files and versioned patches.
- Organize grids, models, runs, experiments, saved views, evidence, and reports.
- Display recent projects, running tasks, qualification state, and recovery status.

### 5.2 Deck and model editor

- Provide synchronized structured-table and source-deck modes.
- Support keyword completion, syntax highlighting, included-file navigation, and documentation.
- Mark errors and warnings at the responsible source location.
- Validate units, dimensions, defaults, and array sizes interactively.
- Offer search by keyword, region, cell, well, date, and source file.
- Show staged changes and side-by-side model differences.
- Support undo/redo without rewriting the original deck.

### 5.3 Grid and property workspace

- Display structured, corner-point, unstructured, fractured, and polyhedral grids.
- Render cells, faces, faults, fractures, NNCs, wells, completions, and segments.
- Support slices, clipping planes, thresholds, layer selection, and region isolation.
- Provide linear and logarithmic color maps, histograms, and cross-plots.
- Animate properties over simulation time.
- Support synchronized cell probing and multiple linked views.

A selected cell should expose indices, coordinates, geometry, rock properties, regions, state variables, fluxes, residual contributions, connectivity, and applicable evidence.

### 5.4 Rock and fluid workspace

- Inspect PVT tables and interpolated functions.
- Plot relative permeability and capillary pressure.
- Compare saturation endpoints and fluid regions.
- Inspect EOS components and interactions.
- Plot phase envelopes and flash results.
- Warn about discontinuities, invalid ordering, and extrapolation.
- Offer interactive pressure-temperature-composition queries.

### 5.5 Initialization workspace

- Configure contacts, datum pressure/depth, equilibration, and compositions.
- Display pressure and saturation against depth.
- Show initialized 3D states and regional phase volumes.
- Calculate pore volume and hydrocarbons initially in place.
- Produce and retain a formal initialization report.

### 5.6 Well and schedule designer

- Provide calendar, timeline, table, and source views.
- Edit trajectories, completions, segments, controls, limits, and groups.
- Show control transitions and active constraints over time.
- Validate repeated completions, inactive cells, conflicting controls, and schedule discontinuities.
- Compare schedules between variants.
- Link schedule events to production and convergence behavior.

### 5.7 Run configuration

Every run configuration should explicitly select:

- Physical model and formulation.
- AD backend.
- Nonlinear and linear solvers.
- Timestep selector.
- Convergence tolerances and iteration limits.
- Hardware and execution backend.
- State, checkpoint, and summary-retention policies.
- Resource constraints.

Named presets are allowed, but every run must store the fully expanded configuration.

### 5.8 Simulation control room

The control room should stream structured simulator events and display:

- Report-step and simulated-date progress.
- Timestep size and nonlinear iteration.
- Residual, update, CNV, and material-balance histories.
- Linear-solver iterations and timings.
- Timestep cuts, restarts, and control switches.
- Current well controls.
- Estimated remaining time.
- CPU, memory, thread, and optional GPU utilization.
- Runtime breakdown for assembly, properties, wells, linear solves, and I/O.

It should support queueing, cancellation, checkpointing, resume, and failure recovery. A worker failure must not terminate the desktop application.

### 5.9 Results and comparison workspace

- Display pressure, saturation, composition, flux, displacement, and stress fields.
- Plot well, group, region, and field summaries.
- Calculate recovery, voidage, breakthrough, sweep, and material balance.
- Compare cases using overlays, statistics, and spatial difference maps.
- Display ensemble distributions and uncertainty envelopes.
- Synchronize time cursors across charts, tables, and 3D views.
- Export CSV, Parquet, HDF5, Zarr, VTK, images, animations, and reports.

### 5.10 Experiment workspace

- Define typed parameter spaces and scenario variants.
- Create deterministic ensembles and branches.
- Select local, remote, or HPC execution.
- Monitor individual realizations and aggregate progress.
- Preserve failed realizations and diagnostic evidence.
- Compare objectives, constraints, uncertainty, and runtime.

### 5.11 Engineering-assistant workspace

The assistant should be a contextual investigation panel rather than an isolated chatbot. It should understand the selected model object, timestep, well, plot, warning, or compared runs.

A finding should provide actions such as:

- Highlight affected cells or wells.
- Open the responsible source record.
- Display supporting convergence history.
- Create a reviewable sensitivity variant.
- Compare before and after.
- Add the finding to a report.
- Mark the finding as accepted, rejected, or unresolved.

## 6. Guided and Expert Modes

### 6.1 Guided mode

A project wizard should guide users through:

1. Importing or creating a grid.
2. Assigning rock properties.
3. Defining fluids.
4. Configuring initialization.
5. Adding wells and completions.
6. Building a schedule.
7. Validating the model.
8. Configuring and running the simulation.
9. Reviewing results.
10. Producing a report.

Each step should explain required inputs, units, assumptions, defaults, and likely errors.

### 6.2 Expert mode

Expert facilities should include:

- Direct deck and Python editing.
- Notebook and console integration.
- Advanced solver and backend configuration.
- Custom models, state functions, and components.
- Experiment and validation-graph definitions.
- Cluster and distributed-run configuration.
- Complete logs, Jacobian diagnostics, profiling, and provenance.

## 7. Professional UI/UX Requirements

Commercial quality depends on predictable behavior and information clarity, not decorative styling. PyMRST Studio should provide:

- Persistent project navigation.
- Unit-aware inputs everywhere.
- Undo/redo and staged changes.
- Autosaved workspace state.
- Clear model and run status.
- Keyboard navigation and command palette.
- Virtualized large tables.
- Cross-view selection synchronization.
- Background task management.
- Nonblocking simulation execution.
- Recoverable checkpoints and crash recovery.
- Useful empty, loading, warning, and failure states.
- Accessible dark and light themes.
- Color-blind-safe engineering palettes.
- Export-quality figures and tables.
- Consistent terminology across deck, Python, API, and interface objects.

The visual design should use a neutral scientific workspace, restrained decoration, dense adjustable tables, clear engineering units, and standardized semantic colors. It should avoid excessive cards, gradients, animations, and rounded decorative elements.

## 8. Project and Run Lifecycle

Use explicit model states:

```text
Imported -> Parsed -> Validated -> Initialized -> Run-ready
         -> Running -> Completed/Failed/Cancelled -> Qualified
```

`Completed` means numerical execution ended successfully. `Qualified` additionally means that requested conservation, convergence, invariant, parity, and acceptance checks passed.

Each run should store:

- PyMRST version and Git commit.
- Dependency versions.
- Platform and hardware information.
- Native thread configuration.
- Input and model hashes.
- Unit system.
- Solver and backend configuration.
- Random seeds.
- Requested outputs.
- Warnings, events, checkpoints, and evidence.

## 9. Storage and Data Management

Recommended storage responsibilities are:

- SQLite for metadata, relationships, task state, notes, and UI state.
- HDF5 or Zarr for grids and multidimensional states.
- Parquet for summaries, convergence records, and analysis tables.
- JSON/JSONL for manifests and structured events.
- Immutable original files for imported decks and data.
- Content hashes for lineage and deduplication.

Example project layout:

```text
project/
  inputs/
  models/
  variants/
  runs/
    run-<uuid>/
      manifest.json
      events.jsonl
      states.h5
      summary.parquet
      convergence.parquet
      evidence/
      logs/
  reports/
```

Large models require lazy state loading, chunked access, table virtualization, on-demand field transfer, level-of-detail geometry, memory quotas, and configurable state-retention policies.

## 10. Local, Remote, and HPC Execution

Define a common runner interface:

```python
submit(run_request) -> RunHandle
status(run_id) -> RunStatus
events(run_id, cursor) -> list[RunEvent]
cancel(run_id)
resume(run_id)
download_results(run_id)
```

Implement runners for:

- Isolated local subprocesses.
- Local multi-process ensembles.
- Containers.
- SSH-accessible workstations.
- SLURM and PBS clusters.
- A shared PyMRST server.

The interface should present the same lifecycle regardless of where a simulation runs.

## 11. Packaging and Deployment

Target distributable packages such as:

- `PyMRST-Studio-macOS.dmg`
- `PyMRST-Studio-Windows.msi`
- `PyMRST-Studio-Linux.AppImage`

The installer may include a managed Python environment, compatible numerical libraries, the local API service, examples, and optional GPU components. First launch should perform an environment check and optionally locate MATLAB/MRST for parity validation.

The same codebase may expose capability profiles for desktop, research, HPC, team-server, and teaching use. These should not become incompatible product forks.

## 12. Robustness and Testing

### 12.1 Process and project safety

- Run every simulation in an isolated process or worker environment.
- Use atomic writes and database transactions.
- Keep originals immutable.
- Version project schemas and provide migrations.
- Recover interrupted tasks and partially downloaded remote results.
- Prevent a simulation crash from terminating the application.

### 12.2 Test layers

- Numerical core and MRST parity tests.
- Project, API, storage, migration, and worker-lifecycle tests.
- UI component, accessibility, and end-to-end workflow tests.
- Visual regressions for stable charts and renderer-safe 3D cases.
- Agent tool-contract, evidence-grounding, permission, and resource-limit tests.
- Cross-platform installer and upgrade tests.

Critical end-to-end acceptance paths should include:

```text
Import -> Validate -> Initialize -> Run -> Checkpoint -> Resume
       -> Analyze -> Compare -> Qualify -> Report
```

## 13. Implementation Roadmap

### Agent M1 — Contracts and evidence

- Typed findings, evidence references, recommendations, and agent protocol.
- Observation/calculation/inference/recommendation classification.
- Provider-independent reasoning interface.
- Permissions and resource-policy foundation.

### Agent M2 — Read-only engineering tools

- Deck, grid, rock/fluid, initialization, schedule, and result inspection.
- Deterministic qualification rules.
- Material-balance, convergence, and invariant tools.

### Agent M3 — Single-case investigation

- Single-agent evidence-backed investigation.
- Context construction and bounded memory.
- Claim-to-evidence validation.
- Reproducible investigation bundle.

### Agent M4 — Specialist orchestration

- Specialist agents and execution graphs.
- Conflict and uncertainty handling.
- Human review and approval workflow.

### Agent M5 — Controlled execution

- Immutable variants.
- Bounded simulation and retry proposals.
- Failure diagnosis and checkpoint-aware investigation.

### Agent M6 — Experiments and reporting

- Scenarios, ensembles, uncertainty, and cross-run analysis.
- Evidence-linked technical reporting.
- MRST/OPM disagreement investigations.

### App M1 — Application foundation

- Tauri/React application shell.
- FastAPI service.
- Project database and storage abstraction.
- Project creation/import and navigation.
- Background task and notification systems.

### App M2 — Model inspection

- Deck editor and structured keyword tables.
- Grid/property viewer.
- Unit system and validation panel.
- Input patches, differences, and lineage.

### App M3 — Simulation execution

- Run configurations and presets.
- Worker-process execution.
- Structured progress events.
- Simulation control room.
- Cancellation, checkpointing, resume, and recovery.

### App M4 — Results and comparison

- State browser and 3D visualization.
- Well, group, region, and field dashboards.
- Case comparison and difference maps.
- Animation and export.

### App M5 — Model construction

- Rock/fluid editing.
- Initialization workspace.
- Well and schedule designer.
- Complete guided workflow.

### App M6 — Research and HPC

- Experiment and ensemble interface.
- Parameter spaces and branching.
- Remote and HPC runners.
- Uncertainty visualization.
- MRST/OPM qualification.

### App M7 — Agentic engineering workspace

- Context-aware assistant.
- Evidence-linked findings.
- Failure investigation.
- User-reviewed variant proposals.
- Report authoring and investigation history.

### App M8 — Production hardening

- Cross-platform installers and updates.
- Large-model performance qualification.
- Authentication and authorization for shared deployments.
- Backup, migration, security, and recovery qualification.

Implementation status (August 2026):

- Workspace backup/recovery is implemented with streaming checksums, transactional
  SQLite snapshots, bounded extraction, background jobs, cancellation, retention,
  verification, and disposable recovery drills. Scheduling remains disabled by
  default.
- Shared-deployment security is implemented as an opt-in boundary. Local desktop
  mode remains the default and retains the frictionless single-engineer workflow.
- Shared mode provides active/disabled user identities, global roles, project
  memberships, project-scoped authorization, expiring/revocable personal access
  tokens, and least-privilege token scopes. Only token hashes are persisted and a
  plaintext token is returned once.
- The complete FastAPI surface is protected in shared mode. Resource identifiers
  are resolved to their owning project before authorization; ambiguous global
  collection operations fail closed to administrators.
- Security decisions and administrative changes enter an append-only SHA-256 hash
  chain with verification and tamper detection. Audit payloads reject
  credential-like fields.
- The Settings workspace includes deployment-mode controls, session credential
  activation, identity and role administration, one-time token presentation,
  project membership assignment, and audit-chain status.
- Database migration 18, service-level security qualification, dependency-light
  ASGI middleware tests, and real FastAPI/HTTPX authorization workflows are
  covered by the application test suite when `pymrst[app]` is installed.
- The `pymrst-studio` launcher fails closed for unsafe non-loopback binds,
  incomplete TLS configuration, wildcard shared-mode hosts/origins, insecure
  production transport, and untrusted forwarding metadata. A production runbook
  records direct-TLS and trusted reverse-proxy deployment procedures.
- First-launch environment qualification is persisted as immutable evidence. It
  separates required Python, NumPy/SciPy, subprocess, workspace, and SQLite checks
  from optional storage, table, visualization, acceleration, HPC, MATLAB, and MRST
  capabilities, and presents remediation in the Settings workspace.
- Release-integrity tooling synchronizes the Python package, Python runtime, npm,
  Cargo, and Tauri versions; creates traversal-safe SHA-256 artifact manifests;
  atomically writes release metadata; and supports detached Ed25519 signing and
  offline signature/artifact verification.
- The Tauri desktop shell now owns an authenticated managed Python sidecar. It uses
  native target-triple binaries, OS-selected loopback ports, readiness/version
  handshakes, in-memory 256-bit credentials, constant-time backend verification,
  bounded startup, private graceful shutdown, and forced-kill fallback. The React
  client learns the endpoint only through minimal Tauri IPC.
- Native-only PyInstaller build planning produces target-named sidecars and
  checksummed receipts without committing generated binaries. The packaging
  runbook defines the six-platform build matrix and acceptance evidence.
- A native Apple-silicon macOS debug acceptance build now produces a complete
  application bundle and checksum-valid DMG containing the managed sidecar. The
  Tauri host, frontend, packaged loopback API, credential boundary, and graceful
  shutdown have passed their local build/runtime gates. Production Developer ID
  signing and Apple notarization still require release credentials.
- Signed-update integration is implemented as an opt-in native trust boundary:
  build-pinned HTTPS endpoint and Ed25519 public key, Tauri v2 updater artifacts,
  native-only candidate retention, default version/downgrade protection, and
  explicit user confirmation before installation. No release endpoint or key is
  embedded in development builds.
- Installation workload qualification now has a durable schema-20 registry,
  bounded isolated-process execution, deterministic seeds, absolute resource
  budgets, previous-pass wall/memory regression checks, cancellation and restart
  recovery, atomic checksummed evidence, tamper re-verification, and a Settings
  dashboard. Numerical-core and state-I/O establish infrastructure. The first
  production owner now executes canonical SPE1 black-oil through the real ADI
  model/solver, reports dataset availability before submission, and retains deck
  and state hashes, nonlinear diagnostics, CNV/MB/well/control evidence, state
  bounds, well outputs, and named correctness invariants for smoke and
  engineering tiers.
- A second production qualification owner exercises fully coupled Mandel
  poromechanics through the VEM mechanics/flow kernel. Smoke and engineering
  tiers retain analytical profile error, Mandel-Cryer overshoot, constant-load
  force balance, thickness uniformity, coupled state hashes and nine physical
  correctness invariants. This demonstrates that installation acceptance is
  not tied to SPE1 or black-oil behavior.
- Workload performance decisions are now statistically controlled: one warmup
  absorbs import/kernel initialization, three deterministic measured runs retain
  raw timing samples plus median/MAD/min/max/P10/P90, excessive relative MAD is
  an acceptance failure, and only multi-run passing evidence can become a
  regression baseline. Repetition output hashes must match, making numerical
  nondeterminism a named correctness failure rather than hidden timing noise.
- Bounded 250 ms process telemetry now spans startup, warmup, every numbered
  measurement, and aggregation. Evidence includes current RSS, cumulative and
  interval CPU, process I/O, thread count, sampled peak locations and a retained
  OS high-water RSS. A dependency-free fallback preserves partial telemetry;
  evidence explicitly declares its provider and unavailable counters.
- Still remaining in M8: platform-signing and native acceptance of production
  macOS, Windows, and Linux installers; publishing a real signed updater channel
  and exercising its negative cases; real large-model workload owners and
  field-scale acceptance evidence; browser-automation security
  workflows; and production recovery evidence on each supported operating system.

## 14. First Complete Product Target

The first professional release should support this complete path:

1. Install and launch PyMRST Studio.
2. Create a project and import an ECLIPSE or PyMRST case.
3. Inspect the grid, properties, fluids, wells, and schedule.
4. Run an evidence-linked input qualification review.
5. Initialize and inspect initial states and volumes.
6. Configure a reproducible simulation.
7. Monitor report steps, timesteps, Newton iterations, convergence, and resource use live.
8. Inspect saved states, 3D properties, wells, and field summaries.
9. Compare a PyMRST run with another case or MRST reference.
10. Ask contextual engineering questions and inspect their supporting evidence.
11. Create controlled variants and compare their outcomes.
12. Export a reproducible engineering report.

SPE1 is a suitable initial acceptance case, but the architecture must also be exercised with a corner-point field case, a compositional case, a fractured/shale case, and a poromechanics case before the application is considered broadly robust.

## 15. Long-Term Differentiation

PyMRST Studio should not be positioned merely as an open clone of a commercial simulator interface. Its distinctive capabilities can be:

- Python-native access to the complete numerical workflow.
- MRST-compatible model and algorithm organization.
- Transparent PyMRST/MRST/OPM qualification.
- Claim-to-cell, claim-to-well, claim-to-timestep, and claim-to-deck evidence.
- Automatic failure localization and reproducible diagnostic experiments.
- Immutable case branching and complete experiment lineage.
- Integrated uncertainty, surrogate, optimization, and differentiable workflows.
- Open, auditable agents that assist without replacing deterministic physics.
- Reproducible reports and publication-quality evidence bundles.
- A single environment spanning teaching, engineering, research, desktop, and HPC use.

The combination of an MRST-compatible simulator, professional application, reproducible research studio, and evidence-aware engineering agents could make PyMRST uniquely valuable to reservoir engineers and researchers.
