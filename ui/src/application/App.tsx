import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { api, type GridSnapshot, type InputPatch, type ModelInspection, type ModelRevision, type NotificationRecord, type PatchPreview, type ProjectRecord, type ResultDashboard, type ResultState, type RunCheckpoint, type RunComparison, type RunConfiguration, type RunEvent, type RunRecord, type SourceSlice } from "../api/client";
import { Icon } from "../components/Icon";
import { ConstructionWorkspace } from "./ConstructionWorkspace";
import { GridCanvas } from "./GridCanvas";
import { ResearchWorkspace } from "./ResearchWorkspace";
import { ExamplesWorkspace } from "./ExamplesWorkspace";
import { AssistantWorkspace } from "./AssistantWorkspace";
import { ProviderSettingsWorkspace } from "./ProviderSettingsWorkspace";

const workspaces = [
  ["Model", "cube"], ["Simulation", "play"], ["Results", "chart"],
  ["Experiments", "flask"], ["Examples", "flask"], ["Assistant", "agent"],
  ["Reports", "report"], ["Settings", "gear"],
] as const;

type ModelTab = "source" | "keywords" | "validation" | "properties" | "changes" | "builder";
type ModelSectionRequest = { tab: ModelTab; step?: number; token: number };

// Sidebar model-tree entries and where each one navigates: the first two
// are ModelWorkspace tabs, the rest are steps inside its "builder" tab.
const modelTreeSections: Array<[string, ModelSectionRequest]> = [
  ["Deck", { tab: "source", token: 0 }],
  ["Grid", { tab: "properties", token: 0 }],
  ["Rock", { tab: "builder", step: 0, token: 0 }],
  ["Fluids", { tab: "builder", step: 1, token: 0 }],
  ["Initialization", { tab: "builder", step: 2, token: 0 }],
  ["Wells", { tab: "builder", step: 3, token: 0 }],
  ["Schedule", { tab: "builder", step: 4, token: 0 }],
];

function ProjectDialog({ onClose, onCreated }: { onClose: () => void; onCreated: (project: ProjectRecord) => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true); setError("");
    try {
      const project = await api.createProject({ name, description, tags: [] });
      onCreated(project);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not create project");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onClose}>
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="project-dialog-title" onMouseDown={(event) => event.stopPropagation()}>
        <div className="eyebrow">New workspace</div>
        <h2 id="project-dialog-title">Create engineering project</h2>
        <p>PyMRST will create an isolated project with immutable original inputs and traceable runs.</p>
        <form onSubmit={submit}>
          <label htmlFor="project-name">Project name<input id="project-name" name="project-name" autoFocus required value={name} onChange={(event) => setName(event.target.value)} placeholder="Norne development study" /></label>
          <label htmlFor="project-description">Description<textarea id="project-description" name="project-description" value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Objectives, simulator basis, and qualification scope" /></label>
          {error && <div className="form-error" role="alert">{error}</div>}
          <div className="dialog-actions"><button type="button" className="button ghost" onClick={onClose}>Cancel</button><button disabled={busy || !name.trim()} className="button primary">{busy ? "Creating…" : "Create project"}</button></div>
        </form>
      </div>
    </div>
  );
}

function ImportDialog({ project, onClose, onImported }: { project: ProjectRecord; onClose: () => void; onImported: (model: ModelInspection) => void }) {
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function submit(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError("");
    try {
      const asset = await api.importAsset(project.project_id, { source_path: path, kind: "deck" });
      onImported(await api.inspectModel(project.project_id, { asset_id: asset.asset_id, name }));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not inspect model");
    } finally { setBusy(false); }
  }
  return <div className="dialog-backdrop" role="presentation" onMouseDown={onClose}>
    <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="import-title" onMouseDown={(event) => event.stopPropagation()}>
      <div className="eyebrow">Immutable import</div><h2 id="import-title">Open reservoir model</h2>
      <p>The source is copied into protected project storage, hashed, and inspected. The original file is never edited.</p>
      <form onSubmit={submit}>
        <label htmlFor="import-deck-path">Deck file path<input id="import-deck-path" name="deck-path" autoFocus required value={path} onChange={(event) => setPath(event.target.value)} placeholder="/projects/NORNE/NORNE.DATA" /></label>
        <label htmlFor="import-model-name">Model name <span className="optional">optional</span><input id="import-model-name" name="model-name" value={name} onChange={(event) => setName(event.target.value)} placeholder="Norne history model" /></label>
        {error && <div className="form-error" role="alert">{error}</div>}
        <div className="dialog-actions"><button type="button" className="button ghost" onClick={onClose}>Cancel</button><button disabled={busy || !path.trim()} className="button primary">{busy ? "Inspecting…" : "Import and inspect"}</button></div>
      </form>
    </div>
  </div>;
}

function GenerateGridDialog({ project, onClose, onGenerated }: { project: ProjectRecord; onClose: () => void; onGenerated: (grid: GridSnapshot) => void }) {
  const [name, setName] = useState("Reservoir grid");
  const [dims, setDims] = useState<[number, number, number]>([10, 10, 5]);
  const [physicalDims, setPhysicalDims] = useState<[number, number, number]>([1000, 1000, 50]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const cellCount = dims[0] * dims[1] * dims[2];
  function setDim(axis: 0 | 1 | 2, value: number) { setDims((current) => { const next = [...current] as [number, number, number]; next[axis] = value; return next; }); }
  function setPhysicalDim(axis: 0 | 1 | 2, value: number) { setPhysicalDims((current) => { const next = [...current] as [number, number, number]; next[axis] = value; return next; }); }
  async function submit(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError("");
    try {
      onGenerated(await api.generateGrid(project.project_id, { name, dims, physical_dims_m: physicalDims }));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not generate grid");
    } finally { setBusy(false); }
  }
  return <div className="dialog-backdrop" role="presentation" onMouseDown={onClose}>
    <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="generate-grid-title" onMouseDown={(event) => event.stopPropagation()}>
      <div className="eyebrow">New grid</div><h2 id="generate-grid-title">Generate a Cartesian grid</h2>
      <p>Start a case without an existing deck. Rock, fluid, initialization, wells and schedule are defined afterward in the guided builder.</p>
      <form onSubmit={submit}>
        <label htmlFor="generate-grid-name">Grid name<input id="generate-grid-name" name="grid-name" required value={name} onChange={(event) => setName(event.target.value)} /></label>
        <div className="vector-input"><span>Cells per axis · i, j, k</span><div>
          {(["I", "J", "K"] as const).map((axis, index) => <label htmlFor={`generate-grid-dim-${axis}`} key={axis}><b>{axis}</b><input id={`generate-grid-dim-${axis}`} name={`grid-dim-${axis}`} type="number" min={1} step={1} value={dims[index]} onChange={(event) => setDim(index as 0 | 1 | 2, Math.max(1, Math.round(Number(event.target.value))))} /></label>)}
        </div></div>
        <div className="vector-input"><span>Physical extent · m</span><div>
          {(["X", "Y", "Z"] as const).map((axis, index) => <label htmlFor={`generate-grid-extent-${axis}`} key={axis}><b>{axis}</b><input id={`generate-grid-extent-${axis}`} name={`grid-extent-${axis}`} type="number" min={0} step="any" value={physicalDims[index]} onChange={(event) => setPhysicalDim(index as 0 | 1 | 2, Number(event.target.value))} /></label>)}
        </div></div>
        <p><strong>{cellCount.toLocaleString()}</strong> cells{cellCount > 2_000_000 ? " — exceeds the synchronous generation limit" : ""}</p>
        {error && <div className="form-error" role="alert">{error}</div>}
        <div className="dialog-actions"><button type="button" className="button ghost" onClick={onClose}>Cancel</button><button disabled={busy || !name.trim() || cellCount < 1 || cellCount > 2_000_000} className="button primary">{busy ? "Generating…" : "Generate grid"}</button></div>
      </form>
    </div>
  </div>;
}

function PatchWorkspace({ model, source }: { model: ModelInspection; source?: SourceSlice }) {
  const [patches, setPatches] = useState<InputPatch[]>([]);
  const [selected, setSelected] = useState<InputPatch>();
  const [preview, setPreview] = useState<PatchPreview>();
  const [title, setTitle] = useState(""); const [start, setStart] = useState(1); const [end, setEnd] = useState(1);
  const [kind, setKind] = useState<InputPatch["operations"][number]["kind"]>("replace_lines");
  const [text, setText] = useState(""); const [error, setError] = useState(""); const [busy, setBusy] = useState(false);
  useEffect(() => { const controller = new AbortController(); api.patches(model.model_id, controller.signal).then(setPatches).catch((caught) => { if (caught.name !== "AbortError") setError(caught.message); }); return () => controller.abort(); }, [model.model_id]);
  async function choose(patch: InputPatch) { setSelected(patch); setError(""); try { setPreview(await api.patchPreview(patch.patch_id)); } catch (caught) { setError(caught instanceof Error ? caught.message : "Preview failed"); } }
  async function create(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError("");
    try {
      const patch = await api.createPatch(model.model_id, { title, description: "Created in PyMRST Studio", operations: [{ kind, start_line: start, end_line: end, text: kind === "delete_lines" ? "" : text }] });
      setPatches((current) => current.some((item) => item.patch_id === patch.patch_id) ? current : [...current, patch]); setTitle(""); await choose(patch);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Patch creation failed"); } finally { setBusy(false); }
  }
  async function materialize() { if (!selected) return; setBusy(true); setError(""); try { const result = await api.materializePatch(selected.patch_id); setSelected(result); setPatches((current) => current.map((item) => item.patch_id === result.patch_id ? result : item)); setPreview(await api.patchPreview(result.patch_id)); } catch (caught) { setError(caught instanceof Error ? caught.message : "Variant creation failed"); } finally { setBusy(false); } }
  return <div className="changes-workspace"><div className="patch-sidebar"><div className="table-heading"><div><span className="eyebrow">Controlled changes</span><h3>Input patches</h3></div></div>{patches.map((patch) => <button className={selected?.patch_id === patch.patch_id ? "active" : ""} onClick={() => choose(patch)} key={patch.patch_id}><strong>{patch.title}</strong><span>{patch.status} · {patch.operations.length} operation</span></button>)}{!patches.length && <p>No input changes proposed.</p>}</div><div className="patch-main">{preview ? <><div className="patch-summary"><div><span className="eyebrow">Difference preview</span><h3>{preview.patch.title}</h3></div><div><span className={preview.original_unchanged ? "lineage-ok" : "lineage-bad"}>{preview.original_unchanged ? "Original verified" : "Source mismatch"}</span><button className="button primary" disabled={busy || selected?.status === "materialized"} onClick={materialize}>{selected?.status === "materialized" ? "Variant materialized" : "Materialize variant"}</button></div></div><pre className="diff-view">{preview.diff || "No textual difference"}</pre></> : <form className="patch-form" onSubmit={create}><span className="eyebrow">Non-destructive editor</span><h3>Propose an input change</h3><p>Changes become a hashed patch first. Materialization creates a separate variant and never rewrites the imported deck.</p><label htmlFor="patch-title">Patch title<input id="patch-title" name="patch-title" required value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Increase injector target" /></label><div className="patch-range"><label htmlFor="patch-operation">Operation<select id="patch-operation" name="patch-operation" value={kind} onChange={(event) => setKind(event.target.value as typeof kind)}><option value="replace_lines">Replace lines</option><option value="insert_before">Insert before</option><option value="insert_after">Insert after</option><option value="delete_lines">Delete lines</option></select></label><label htmlFor="patch-start-line">Start line<input id="patch-start-line" name="patch-start-line" type="number" min="1" max={source?.total_lines} value={start} onChange={(event) => setStart(Number(event.target.value))} /></label><label htmlFor="patch-end-line">End line<input id="patch-end-line" name="patch-end-line" type="number" min={start} max={source?.total_lines} value={end} onChange={(event) => setEnd(Number(event.target.value))} /></label></div>{kind !== "delete_lines" && <label htmlFor="patch-replacement-text">Replacement deck text<textarea id="patch-replacement-text" name="patch-replacement-text" required value={text} onChange={(event) => setText(event.target.value)} spellCheck={false} placeholder="WCONPROD\n  'P1' 'OPEN' 'ORAT' 1500 /" /></label>}<button className="button primary" disabled={busy || !title.trim()}>{busy ? "Creating…" : "Create patch and preview"}</button></form>}{error && <div className="form-error">{error}</div>}</div></div>;
}

function SimulationWorkspace({ project, model }: { project: ProjectRecord; model?: ModelInspection }) {
  const [configurations, setConfigurations] = useState<RunConfiguration[]>([]);
  const [runs, setRuns] = useState<RunRecord[]>([]); const [selectedId, setSelectedId] = useState("");
  const [events, setEvents] = useState<RunEvent[]>([]); const [checkpoints, setCheckpoints] = useState<RunCheckpoint[]>([]);
  const [name, setName] = useState("Execution qualification"); const [steps, setSteps] = useState(12); const [checkpointAt, setCheckpointAt] = useState(6);
  const [runnerKind, setRunnerKind] = useState(model ? "pymrst-spe1" : "protocol-qualification");
  const [revisions, setRevisions] = useState<ModelRevision[]>([]); const [revisionId, setRevisionId] = useState("");
  const [error, setError] = useState(""); const [busy, setBusy] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    api.projectModelRevisions(project.project_id, controller.signal)
      .then((values) => { setRevisions(values); setRevisionId((current) => current || values[0]?.revision_id || ""); })
      .catch((caught) => { if (caught.name !== "AbortError") setError(caught.message); });
    return () => controller.abort();
  }, [project.project_id]);
  const selected = runs.find((run) => run.run_id === selectedId) ?? runs.at(-1);
  const terminal = selected && ["completed", "failed", "cancelled", "interrupted"].includes(selected.status);
  const refreshLists = async () => {
    const [configValues, runValues] = await Promise.all([api.runConfigurations(project.project_id), api.runs(project.project_id)]);
    setConfigurations(configValues); setRuns(runValues); setSelectedId((current) => current || runValues.at(-1)?.run_id || "");
  };
  useEffect(() => { let active = true; refreshLists().catch((caught) => active && setError(caught.message)); return () => { active = false; }; }, [project.project_id]);
  useEffect(() => {
    if (!selected?.run_id) { setEvents([]); setCheckpoints([]); return; }
    let active = true;
    const update = async () => {
      try {
        const [run, eventValues, checkpointValues] = await Promise.all([api.run(selected.run_id), api.runEvents(selected.run_id), api.checkpoints(selected.run_id)]);
        if (!active) return; setRuns((current) => current.map((item) => item.run_id === run.run_id ? run : item)); setEvents(eventValues); setCheckpoints(checkpointValues);
      } catch (caught) { if (active) setError(caught instanceof Error ? caught.message : "Run refresh failed"); }
    };
    update(); const timer = window.setInterval(update, 700); return () => { active = false; window.clearInterval(timer); };
  }, [selected?.run_id]);
  const progress = [...events].reverse().find((event) => event.event_type === "progress")?.payload.progress as number | undefined;
  const reportEvent = [...events].reverse().find((event) => event.event_type === "progress" && event.payload.report_step !== undefined);
  const convergence = events.filter((event) => event.event_type === "convergence");
  async function configure(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError("");
    try {
      const payload = runnerKind === "pymrst-spe1"
        ? { report_steps: steps, checkpoint_interval: checkpointAt, save_states: true, use_line_search: true }
        : runnerKind === "pymrst-construction-run"
        ? { revision_id: revisionId, save_states: true, use_line_search: true }
        : { steps, checkpoint_at: checkpointAt };
      const value = await api.createRunConfiguration(project.project_id, { name, runner_kind: runnerKind, model_id: model?.model_id ?? "", payload });
      setConfigurations((current) => current.some((item) => item.configuration_id === value.configuration_id) ? current : [...current, value]);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Configuration failed"); } finally { setBusy(false); }
  }
  async function submit(configuration: RunConfiguration) { setBusy(true); setError(""); try { const run = await api.submitRun(configuration.configuration_id); setRuns((current) => [...current, run]); setSelectedId(run.run_id); } catch (caught) { setError(caught instanceof Error ? caught.message : "Submission failed"); } finally { setBusy(false); } }
  async function cancel() { if (!selected) return; setBusy(true); try { await api.cancelRun(selected.run_id); } catch (caught) { setError(caught instanceof Error ? caught.message : "Cancellation failed"); } finally { setBusy(false); } }
  async function resume(checkpoint: RunCheckpoint) { if (!selected) return; setBusy(true); try { const run = await api.resumeRun(selected.run_id, checkpoint.checkpoint_id); setRuns((current) => [...current, run]); setSelectedId(run.run_id); } catch (caught) { setError(caught instanceof Error ? caught.message : "Resume failed"); } finally { setBusy(false); } }
  return <section className="control-room">
    <div className="control-sidebar"><div className="table-heading"><div><span className="eyebrow">Reproducible presets</span><h3>Run configurations</h3></div></div><form className="compact-form" onSubmit={configure}><label htmlFor="run-config-name">Name<input id="run-config-name" name="run-config-name" required value={name} onChange={(event) => setName(event.target.value)} /></label><label htmlFor="run-config-runner">Runner<select id="run-config-runner" name="run-config-runner" value={runnerKind} onChange={(event) => setRunnerKind(event.target.value)}><option value="protocol-qualification">Execution protocol qualification</option><option value="pymrst-spe1" disabled={!model}>PyMRST SPE1 black-oil</option><option value="pymrst-construction-run" disabled={!revisions.length}>Constructed model</option></select></label>{runnerKind === "pymrst-construction-run" ? <label htmlFor="run-config-revision">Model revision<select id="run-config-revision" name="run-config-revision" value={revisionId} onChange={(event) => setRevisionId(event.target.value)}>{!revisions.length && <option value="">No materialized revisions yet</option>}{revisions.map((item) => <option value={item.revision_id} key={item.revision_id}>{String(item.summary.name ?? "Model")} · v{item.revision_number} · {String(item.summary.cells ?? "?")} cells</option>)}</select></label> : <div><label htmlFor="run-config-steps">Report steps<input id="run-config-steps" name="run-config-steps" type="number" min="1" max="1000" value={steps} onChange={(event) => setSteps(Number(event.target.value))} /></label><label htmlFor="run-config-checkpoint">{runnerKind === "pymrst-spe1" ? "Checkpoint interval" : "Checkpoint at"}<input id="run-config-checkpoint" name="run-config-checkpoint" type="number" min="1" max={steps} value={checkpointAt} onChange={(event) => setCheckpointAt(Number(event.target.value))} /></label></div>}<button className="button ghost" disabled={busy || (runnerKind === "pymrst-construction-run" && !revisionId)}>Save run preset</button></form><div className="configuration-list">{configurations.map((configuration) => <article key={configuration.configuration_id}><strong>{configuration.name}</strong><span>{configuration.runner_kind}</span><code>{configuration.configuration_hash.slice(0, 10)}</code><button className="button primary" disabled={busy} onClick={() => submit(configuration)}>Run</button></article>)}</div></div>
    <div className="run-console"><div className="run-selector"><div><span className="eyebrow">Isolated attempts</span><h3>{selected ? `Attempt ${selected.attempt_number}` : "No run selected"}</h3></div><select id="run-attempt-selector" name="run-attempt-selector" aria-label="Select run attempt" value={selected?.run_id ?? ""} onChange={(event) => setSelectedId(event.target.value)}><option value="">Select run</option>{runs.map((run) => <option value={run.run_id} key={run.run_id}>#{run.attempt_number} · {run.status} · {run.run_id.slice(-8)}</option>)}</select>{selected && !terminal && <button className="button danger" disabled={busy} onClick={cancel}>Cancel run</button>}</div>{selected ? <><div className="live-status"><div className={`run-state ${selected.status}`}><i />{selected.status}</div><div className="progress-track"><span style={{ width: `${Math.max(0, Math.min(1, progress ?? 0)) * 100}%` }} /></div><strong>{Math.round((progress ?? 0) * 100)}%</strong><small>{reportEvent ? `Report step ${reportEvent.payload.report_step} of ${reportEvent.payload.report_steps}` : "Waiting for worker progress"}</small></div><div className="telemetry-grid"><article><span>Process</span><strong>{selected.process_id ?? "—"}</strong></article><article><span>Events</span><strong>{events.length}</strong></article><article><span>Checkpoints</span><strong>{checkpoints.length}</strong></article><article><span>Elapsed</span><strong>{events.at(-1)?.elapsed_seconds.toFixed(2) ?? "0.00"} s</strong></article></div><div className="convergence-panel"><div className="table-heading"><div><span className="eyebrow">Live numerical evidence</span><h3>Convergence stream</h3></div><span>{convergence.length} accepted reports</span></div><div className="spark-bars">{convergence.map((event) => { const residual = Number(event.payload.residual); const height = Math.max(5, Math.min(100, -Math.log10(residual) * 14)); return <i key={event.sequence} style={{ height: `${height}%` }} title={`Step ${event.payload.report_step}: ${residual}`} />; })}</div></div><div className="event-log">{events.slice(-12).map((event) => <div key={event.sequence}><code>{String(event.sequence).padStart(4, "0")}</code><span>{event.event_type}</span><strong>{String(event.payload.message ?? event.payload.error_type ?? "")}</strong><small>{event.elapsed_seconds.toFixed(3)}s</small></div>)}</div>{checkpoints.length > 0 && <div className="checkpoint-strip"><span>Restart points</span>{checkpoints.map((checkpoint) => <button disabled={busy || !terminal} key={checkpoint.checkpoint_id} onClick={() => resume(checkpoint)}>Resume step {String(checkpoint.metadata.completed_steps ?? checkpoint.sequence)}</button>)}</div>}</> : <div className="empty-data"><h3>Simulation control room</h3><p>Save a qualification preset and start it to verify isolated execution, live structured events, cancellation, and checkpoint restart.</p></div>}{error && <div className="form-error control-error">{error}</div>}</div>
  </section>;
}

function ResultFieldCanvas({ coordinates, values, label }: { coordinates: number[][]; values: number[]; label: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const target = canvas.current; if (!target || !values.length) return;
    const draw = () => {
      const bounds = target.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
      target.width = Math.max(1, bounds.width * ratio); target.height = Math.max(1, bounds.height * ratio);
      const context = target.getContext("2d"); if (!context) return; context.scale(ratio, ratio);
      context.fillStyle = "#07131c"; context.fillRect(0, 0, bounds.width, bounds.height);
      const finite = values.filter(Number.isFinite), min = Math.min(...finite), max = Math.max(...finite);
      if (coordinates.length === values.length && coordinates[0]?.length >= 2) {
        const projected = coordinates.map((point) => point.length >= 3
          ? [point[0] + .52*point[2], point[1] - .34*point[2]] : [point[0], point[1]]);
        const xs = projected.map((point) => point[0]), ys = projected.map((point) => point[1]);
        const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), pad = 18;
        values.forEach((value, index) => {
          const fraction = max > min ? (value-min)/(max-min) : .5;
          const x = pad + (projected[index][0]-x0)/Math.max(x1-x0, 1e-20)*(bounds.width-2*pad);
          const y = bounds.height-pad-(projected[index][1]-y0)/Math.max(y1-y0, 1e-20)*(bounds.height-2*pad);
          context.fillStyle = Number.isFinite(value) ? `hsl(${200-170*fraction} 76% ${43+10*fraction}%)` : "#697781";
          context.fillRect(x-2, y-2, 4, 4);
        });
      } else {
        const bins = 42, counts = Array(bins).fill(0);
        finite.forEach((value) => counts[Math.min(bins-1, Math.floor((value-min)/Math.max(max-min, 1e-20)*bins))]++);
        const peak = Math.max(...counts, 1); counts.forEach((count, index) => {
          const width = bounds.width/bins-1, height = count/peak*(bounds.height-30);
          context.fillStyle = `hsl(${195-145*index/(bins-1)} 70% 48%)`;
          context.fillRect(index*bounds.width/bins, bounds.height-height, width, height);
        });
      }
    };
    draw(); const observer = new ResizeObserver(draw); observer.observe(target); return () => observer.disconnect();
  }, [coordinates, values]);
  return <div className="result-canvas"><canvas ref={canvas} aria-label={`${label} result visualization`} /><span>{label}</span></div>;
}

function HistoryChart({ dashboard, quantity }: { dashboard: ResultDashboard; quantity: string }) {
  const series = dashboard.field[quantity]?.map((item) => item.mean) ?? dashboard.wells[quantity]?.map((row) => row[0]) ?? [];
  if (!series.length) return <div className="empty-chart">No history</div>;
  const min = Math.min(...series), max = Math.max(...series);
  const points = series.map((value, index) => `${10 + index/Math.max(series.length-1, 1)*280},${85-(value-min)/Math.max(max-min, 1e-20)*70}`).join(" ");
  return <svg className="history-chart" viewBox="0 0 300 100" role="img" aria-label={`${quantity} time history`}><line x1="10" y1="85" x2="290" y2="85" /><polyline points={points} /><text x="10" y="97">{(dashboard.times[0]/86400).toFixed(0)} d</text><text x="270" y="97">{(dashboard.times.at(-1)!/86400).toFixed(0)} d</text><text x="12" y="12">{max.toPrecision(4)}</text></svg>;
}

function ResultsWorkspace({ project, grids }: { project: ProjectRecord; grids: GridSnapshot[] }) {
  const [runs, setRuns] = useState<RunRecord[]>([]); const [runId, setRunId] = useState("");
  const [states, setStates] = useState<ResultState[]>([]); const [stateId, setStateId] = useState(""); const [field, setField] = useState("pressure");
  const [values, setValues] = useState<number[]>([]); const [coordinates, setCoordinates] = useState<number[][]>([]);
  const [dashboard, setDashboard] = useState<ResultDashboard>(); const [compareRunId, setCompareRunId] = useState(""); const [comparison, setComparison] = useState<RunComparison>();
  const [differenceValues, setDifferenceValues] = useState<number[]>([]);
  const [frame, setFrame] = useState(0); const [playing, setPlaying] = useState(false); const [error, setError] = useState(""); const [exportMessage, setExportMessage] = useState("");
  const state = states.find((item) => item.state_id === stateId) ?? states.at(-1);
  useEffect(() => { const controller = new AbortController(); api.runs(project.project_id, controller.signal).then((items) => { const withResults = items.filter((item) => ["completed", "cancelled", "failed"].includes(item.status)); setRuns(withResults); setRunId(withResults.at(-1)?.run_id ?? ""); }).catch((caught) => { if (caught.name !== "AbortError") setError(caught.message); }); return () => controller.abort(); }, [project.project_id]);
  useEffect(() => {
    setStates([]); setDashboard(undefined); setComparison(undefined); if (!runId) return;
    const controller = new AbortController(); Promise.all([api.resultStates(runId, controller.signal), api.dashboard(runId, controller.signal)]).then(([stateValues, summary]) => { setStates(stateValues); setDashboard(summary); const selected = stateValues.at(-1); setStateId(selected?.state_id ?? ""); setField(selected?.fields.find((item) => item.name === "pressure")?.name ?? selected?.fields[0]?.name ?? ""); setFrame(Math.max(0, stateValues.length-1)); }).catch((caught) => { if (caught.name !== "AbortError") setError(caught.message); }); return () => controller.abort();
  }, [runId]);
  useEffect(() => {
    if (!state || !field) { setValues([]); return; } const controller = new AbortController();
    const total = state.fields.find((item) => item.name === field)?.shape[0] ?? 0;
    const stride = Math.max(1, Math.ceil(total/25000));
    const requests: Promise<unknown>[] = [api.resultField(state.state_id, field, 0, 25000, controller.signal, stride)];
    if (grids[0]) requests.push(api.gridValues(grids[0].snapshot_id, "CELL_CENTROIDS", 0, 25000, controller.signal, stride));
    Promise.all(requests).then((items) => { const raw = (items[0] as { values: number[] | number[][] }).values; setValues(raw.map((item) => Array.isArray(item) ? Number(item[0]) : Number(item))); if (items[1]) setCoordinates((items[1] as { values: number[][] }).values); else setCoordinates([]); }).catch((caught) => { if (caught.name !== "AbortError") setError(caught.message); }); return () => controller.abort();
  }, [state?.state_id, field, grids[0]?.snapshot_id]);
  useEffect(() => { if (!playing || states.length < 2) return; const timer = window.setInterval(() => setFrame((current) => (current+1)%states.length), 700); return () => window.clearInterval(timer); }, [playing, states.length]);
  useEffect(() => { if (states[frame]) setStateId(states[frame].state_id); }, [frame, states]);
  useEffect(() => {
    setDifferenceValues([]); if (!comparison || !state) return; const controller = new AbortController();
    api.difference(comparison.reference_run_id, comparison.candidate_run_id, state.report_step, field, 0, 25000, controller.signal).then((result) => { const raw = result.values as number[] | number[][]; setDifferenceValues(raw.map((item) => Array.isArray(item) ? Number(item[0]) : Number(item))); }).catch((caught) => { if (caught.name !== "AbortError") setError(caught.message); }); return () => controller.abort();
  }, [comparison?.comparison_id, state?.report_step, field]);
  async function compare() { if (!runId || !compareRunId) return; setError(""); try { setComparison(await api.compareRuns(runId, compareRunId)); } catch (caught) { setError(caught instanceof Error ? caught.message : "Comparison failed"); } }
  async function exportField() { if (!state) return; setExportMessage(""); try { const result = await api.exportField(state.state_id, field); setExportMessage(`Exported ${result.relative_path}`); } catch (caught) { setError(caught instanceof Error ? caught.message : "Export failed"); } }
  return <section className="results-room"><div className="results-toolbar"><label htmlFor="results-run">Run<select id="results-run" name="results-run" value={runId} onChange={(event) => setRunId(event.target.value)}><option value="">Select completed run</option>{runs.map((run) => <option key={run.run_id} value={run.run_id}>Attempt {run.attempt_number} · {run.status} · {run.run_id.slice(-8)}</option>)}</select></label><label htmlFor="results-report-step">Report step<select id="results-report-step" name="results-report-step" value={state?.state_id ?? ""} onChange={(event) => { setStateId(event.target.value); setFrame(Math.max(0, states.findIndex((item) => item.state_id === event.target.value))); }}>{states.map((item) => <option key={item.state_id} value={item.state_id}>Step {item.report_step} · {(item.simulated_time_seconds/86400).toFixed(1)} d</option>)}</select></label><label htmlFor="results-property">Property<select id="results-property" name="results-property" value={field} onChange={(event) => setField(event.target.value)}>{state?.fields.map((item) => <option key={item.name}>{item.name}</option>)}</select></label><button className="button ghost" disabled={!state} onClick={exportField}>Export CSV</button></div>{state ? <div className="results-layout"><div className="state-viewport"><ResultFieldCanvas coordinates={coordinates.length === values.length ? coordinates : []} values={values} label={`${field} · report ${state.report_step}`} /><div className="animation-controls"><button onClick={() => setPlaying((value) => !value)}>{playing ? "Pause" : "Play"}</button><input id="results-frame" name="results-frame" aria-label="Report step frame" type="range" min="0" max={Math.max(states.length-1, 0)} value={frame} onChange={(event) => { setPlaying(false); setFrame(Number(event.target.value)); }} /><span>{frame+1}/{states.length}</span></div></div><div className="result-details"><span className="eyebrow">State statistics</span><h3>{field}</h3>{(() => { const item = state.fields.find((entry) => entry.name === field); return item ? <dl><dt>Minimum</dt><dd>{item.minimum?.toPrecision(7) ?? "—"}</dd><dt>Mean</dt><dd>{item.mean?.toPrecision(7) ?? "—"}</dd><dt>Maximum</dt><dd>{item.maximum?.toPrecision(7) ?? "—"}</dd><dt>Shape</dt><dd>{item.shape.join(" × ")}</dd><dt>Checksum</dt><dd>{state.content_hash.slice(0, 16)}…</dd></dl> : null; })()}</div><div className="dashboard-grid">{dashboard && ["pressure", "sW", "bhp", "qOs"].filter((item) => dashboard.field[item] || dashboard.wells[item]).map((quantity) => <article key={quantity}><div><span className="eyebrow">History</span><strong>{quantity}</strong></div><HistoryChart dashboard={dashboard} quantity={quantity} /></article>)}</div><div className="comparison-panel"><div className="table-heading"><div><span className="eyebrow">Case comparison</span><h3>Aligned run differences</h3></div><div><select id="results-compare-run" name="results-compare-run" aria-label="Candidate run for comparison" value={compareRunId} onChange={(event) => setCompareRunId(event.target.value)}><option value="">Candidate run</option>{runs.filter((run) => run.run_id !== runId).map((run) => <option value={run.run_id} key={run.run_id}>Attempt {run.attempt_number} · {run.run_id.slice(-8)}</option>)}</select><button className="button primary" disabled={!compareRunId} onClick={compare}>Compare</button></div></div>{comparison && <><div className="comparison-table"><div className="table-row header"><span>Field</span><span>Max absolute</span><span>Max relative</span><span>RMSE</span><span>Worst step</span></div>{Object.entries(comparison.summary.fields).map(([name, metrics]) => <div className="table-row" key={name}><strong>{name}</strong><span>{metrics.maximum_absolute_error.toPrecision(5)}</span><span>{metrics.maximum_relative_error.toPrecision(5)}</span><span>{metrics.root_mean_square_error.toPrecision(5)}</span><span>{metrics.worst_report_step}</span></div>)}</div>{differenceValues.length > 0 && <div className="difference-map"><ResultFieldCanvas coordinates={coordinates.length === differenceValues.length ? coordinates : []} values={differenceValues} label={`Δ ${field} · candidate − reference`} /></div>}</>}</div></div> : <div className="empty-data"><h3>No saved states available</h3><p>Complete a run with state persistence enabled, then return here to browse reservoir and well results.</p></div>}{exportMessage && <div className="export-toast">{exportMessage}</div>}{error && <div className="form-error control-error">{error}</div>}</section>;
}

function ModelWorkspace({ project, model, grids, onImport, onGenerateGrid, sectionRequest }: { project: ProjectRecord; model?: ModelInspection; grids: GridSnapshot[]; onImport: () => void; onGenerateGrid: () => void; sectionRequest?: ModelSectionRequest | null }) {
  const [tab, setTab] = useState<ModelTab>("source");
  const [source, setSource] = useState<SourceSlice>();
  const [sourceError, setSourceError] = useState("");
  const grid = grids[0];
  useEffect(() => {
    setSource(undefined); setSourceError("");
    if (!model?.files[0]) return;
    const controller = new AbortController();
    api.source(model.model_id, model.files[0].path, 1, 500, controller.signal)
      .then(setSource).catch((error) => { if (error.name !== "AbortError") setSourceError(error.message); });
    return () => controller.abort();
  }, [model?.model_id]);
  useEffect(() => {
    if (sectionRequest) setTab(sectionRequest.tab);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sectionRequest?.token]);
  const builderStep = sectionRequest?.tab === "builder" ? sectionRequest : undefined;
  if (!model) return grids.length ? <ConstructionWorkspace project={project} grids={grids} stepRequest={builderStep} /> : <div className="canvas-empty"><div className="reservoir-placeholder"><span /><span /><span /><span /></div><div><h3>Bring your reservoir model into a governed workspace</h3><p>Import an ECLIPSE deck to index source inputs, or generate a grid to construct rock, fluids, initialization, wells and schedule in the guided builder.</p><div className="dialog-actions"><button className="button ghost" onClick={onGenerateGrid}>Generate grid</button><button className="button primary" onClick={onImport}>Import deck</button></div></div></div>;
  const errors = model.issues.filter((item) => item.severity === "error").length;
  return <div className="model-workbench">
    <div className="workbench-tabs" role="tablist">
      {([ ["source", "Deck source"], ["keywords", `Keywords ${model.occurrences?.length ?? 0}`], ["validation", `Validation ${model.issues.length}`], ["properties", `Grid & properties ${grid ? 1 : 0}`], ["builder", "Model builder"], ["changes", "Changes"] ] as const).map(([id, label]) => <button role="tab" aria-selected={tab === id} className={tab === id ? "active" : ""} onClick={() => setTab(id)} key={id}>{label}</button>)}
      <span className={`model-health ${errors ? "bad" : "good"}`}><i />{errors ? `${errors} blocking` : "Inspection passed"}</span>
    </div>
    {tab === "source" && <div className="source-view"><div className="source-meta"><strong>{source?.path ?? model.name}</strong><span>{source ? `${source.total_lines.toLocaleString()} lines` : "Loading…"}</span><code>{model.source_hash.slice(0, 12)}</code></div>{sourceError ? <div className="form-error">{sourceError}</div> : <ol className="code-lines" start={source?.start_line}>{source?.lines.map((line, index) => <li key={index}><code>{line || " "}</code></li>)}</ol>}</div>}
    {tab === "keywords" && <div className="data-view"><div className="table-heading"><div><span className="eyebrow">Structured index</span><h3>Deck keyword occurrences</h3></div><span>{model.sections.join(" · ")}</span></div><div className="data-table"><div className="table-row header"><span>Keyword</span><span>Section</span><span>Source</span><span>Values</span><span>Defaults</span></div>{model.occurrences?.map((item) => <div className="table-row" key={item.occurrence_id}><strong>{item.keyword}</strong><span>{item.section || "—"}</span><span>{item.location.path}:{item.location.line}</span><span>{item.value_count}</span><span>{item.defaulted_value_count}</span></div>)}</div></div>}
    {tab === "validation" && <div className="data-view"><div className="table-heading"><div><span className="eyebrow">Engineering checks</span><h3>Model validation</h3></div><span>Units: {model.unit_system}</span></div><div className="issue-list">{model.issues.length === 0 ? <div className="pass-panel"><i>✓</i><div><strong>No structural findings</strong><p>The bounded source inspection completed without errors or warnings.</p></div></div> : model.issues.map((issue, index) => <article className={`issue ${issue.severity}`} key={`${issue.code}-${index}`}><span>{issue.severity}</span><div><strong>{issue.message}</strong><small>{issue.code} · {issue.category}{issue.location ? ` · ${issue.location.path}:${issue.location.line}` : ""}</small></div></article>)}</div></div>}
    {tab === "properties" && <div className="data-view">{!grid ? <div className="empty-data"><h3>No grid snapshot registered</h3><p>Once model construction produces a grid, its centroids, volumes, active index and cell properties appear here through bounded disk-backed slices.</p></div> : <><div className="grid-metrics"><article><span>Cells</span><strong>{grid.cell_count.toLocaleString()}</strong></article><article><span>Faces</span><strong>{grid.face_count.toLocaleString()}</strong></article><article><span>Nodes</span><strong>{grid.node_count.toLocaleString()}</strong></article><article><span>Dimensions</span><strong>{grid.dimension}D</strong></article></div><GridCanvas grid={grid} /><div className="data-table properties"><div className="table-row header"><span>Array</span><span>Shape</span><span>Minimum</span><span>Mean</span><span>Maximum</span></div>{grid.arrays.map((item) => <div className="table-row" key={item.name}><strong>{item.name}</strong><span>{item.shape.join(" × ")}</span><span>{item.minimum?.toPrecision(5) ?? "—"}</span><span>{item.mean?.toPrecision(5) ?? "—"}</span><span>{item.maximum?.toPrecision(5) ?? "—"}</span></div>)}</div></>}</div>}
    {tab === "changes" && <PatchWorkspace model={model} source={source} />}
    {tab === "builder" && <ConstructionWorkspace project={project} grids={grids} stepRequest={builderStep} />}
  </div>;
}

export function App() {
  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [notifications, setNotifications] = useState<NotificationRecord[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [workspace, setWorkspace] = useState("Model");
  const [modelSectionRequest, setModelSectionRequest] = useState<ModelSectionRequest | null>(null);
  const [online, setOnline] = useState(false);
  const [loading, setLoading] = useState(true);
  const [problem, setProblem] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [showImport, setShowImport] = useState(false);
  const [showGenerateGrid, setShowGenerateGrid] = useState(false);
  const [model, setModel] = useState<ModelInspection>();
  const [grids, setGrids] = useState<GridSnapshot[]>([]);
  const [modelLoading, setModelLoading] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([api.health(controller.signal), api.projects(controller.signal), api.notifications(controller.signal)])
      .then(([, projectValues, notificationValues]) => {
        setOnline(true); setProjects(projectValues); setNotifications(notificationValues);
        setSelectedId((current) => current || projectValues[0]?.project_id || "");
      })
      .catch((error) => { if (error.name !== "AbortError") setProblem(error.message); })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  const selected = useMemo(() => projects.find((project) => project.project_id === selectedId), [projects, selectedId]);

  useEffect(() => {
    setModel(undefined); setGrids([]);
    if (!selectedId) return;
    const controller = new AbortController(); setModelLoading(true);
    Promise.all([api.models(selectedId, controller.signal), api.grids(selectedId, controller.signal)])
      .then(async ([models, gridValues]) => {
        setGrids(gridValues);
        if (models[0]) setModel(await api.model(models[0].model_id, controller.signal));
      })
      .catch((error) => { if (error.name !== "AbortError") setProblem(error.message); })
      .finally(() => setModelLoading(false));
    return () => controller.abort();
  }, [selectedId]);

  function projectCreated(project: ProjectRecord) {
    setProjects((current) => [...current, project]);
    setSelectedId(project.project_id); setShowCreate(false); setOnline(true);
  }

  function modelImported(value: ModelInspection) { setModel(value); setShowImport(false); }
  function gridGenerated(grid: GridSnapshot) { setGrids((current) => [...current, grid]); setShowGenerateGrid(false); }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace">Skip to engineering workspace</a>
      <header className="topbar">
        <div className="brand"><span className="brand-mark">P</span><span>PyMRST <b>Studio</b></span></div>
        <nav className="global-nav" aria-label="Application sections">
          {workspaces.map(([label]) => <button key={label} className={workspace === label ? "active" : ""} onClick={() => setWorkspace(label)}>{label}</button>)}
        </nav>
        <div className="top-actions">
          <span className={`service-state ${online ? "online" : "offline"}`}><i />{online ? "Local service" : "Offline"}</span>
          <button className="icon-button" aria-label={`${notifications.length} unread notifications`}><Icon name="bell" /><span>{notifications.length}</span></button>
          <div className="avatar" aria-label="User profile">CE</div>
        </div>
      </header>

      <aside className="project-panel" aria-label="Project explorer">
        <div className="panel-heading"><div><span className="eyebrow">Workspace</span><h1>Projects</h1></div><button className="icon-button add" aria-label="Create project" onClick={() => setShowCreate(true)}><Icon name="plus" /></button></div>
        <div className="project-list">
          {projects.map((project) => <button key={project.project_id} className={`project-row ${selectedId === project.project_id ? "selected" : ""}`} onClick={() => setSelectedId(project.project_id)}><span className="project-symbol">{project.name.slice(0, 2).toUpperCase()}</span><span><strong>{project.name}</strong><small>{project.tags.join(" · ") || "Reservoir study"}</small></span></button>)}
          {!loading && projects.length === 0 && <div className="empty-mini">No projects yet.<button onClick={() => setShowCreate(true)}>Create the first project</button></div>}
        </div>
        {selected && <div className="model-tree"><div className="tree-title">MODEL</div>{modelTreeSections.map(([item, target], index) => <button key={item} className={workspace === "Model" && (modelSectionRequest ? modelSectionRequest.tab === target.tab && modelSectionRequest.step === target.step : index === 0) ? "active" : ""} onClick={() => { setWorkspace("Model"); setModelSectionRequest({ ...target, token: Date.now() }); }}><span className="tree-line" />{item}<small>{index === 0 ? (modelLoading ? "Indexing" : model ? model.status : "—") : index === 1 && grids[0] ? "Ready" : "—"}</small></button>)}</div>}
        <div className="storage-indicator"><span><i />Project storage</span><strong>Local</strong></div>
      </aside>

      <main id="workspace" className="engineering-workspace" tabIndex={-1}>
        {problem && <div className="connection-banner" role="alert"><strong>Application service unavailable.</strong> {problem}<button onClick={() => location.reload()}>Retry</button></div>}
        <div className="workspace-header"><div><div className="breadcrumb">Projects / {selected?.name ?? "No project"} / {workspace}</div><h2>{selected ? `${selected.name} — ${workspace}` : "Welcome to PyMRST Studio"}</h2><p>{selected?.description || "Create a project to begin a traceable reservoir-engineering workflow."}</p></div><div className="header-actions"><button className="button ghost" disabled={!selected} onClick={() => setShowImport(true)}>{model ? "Import another model" : "Import model"}</button><button className="button primary" disabled={!selected || !model || model.status === "invalid"} onClick={() => setWorkspace("Simulation")}><Icon name="play" />Configure run</button></div></div>
        <section className="status-grid" aria-label="Project status">
          {[ ["MODEL STATUS", model ? (model.status === "ready" ? "Ready for review" : "Blocked by validation") : "Not configured", model?.status === "ready" ? "cyan" : "amber"], ["ACTIVE CELLS", grids[0]?.active_cell_count.toLocaleString() ?? "—", ""], ["DECK KEYWORDS", model?.occurrences?.length.toLocaleString() ?? "—", ""], ["VALIDATION", model ? `${model.issues.length} finding${model.issues.length === 1 ? "" : "s"}` : "Not started", model?.issues.some((item) => item.severity === "error") ? "amber" : "cyan"] ].map(([label, value, tone]) => <article key={label}><span>{label}</span><strong className={tone}>{value}</strong><small>{selected ? "Evidence retained" : "Create or import a model"}</small></article>)}
        </section>
        {workspace === "Settings" ? <ProviderSettingsWorkspace /> : workspace === "Simulation" && selected ? <SimulationWorkspace project={selected} model={model} /> : workspace === "Results" && selected ? <ResultsWorkspace project={selected} grids={grids} /> : workspace === "Experiments" && selected ? <ResearchWorkspace project={selected} /> : workspace === "Examples" && selected ? <ExamplesWorkspace project={selected} /> : workspace === "Assistant" && selected ? <AssistantWorkspace project={selected} /> : workspace === "Reports" && selected ? <AssistantWorkspace project={selected} reportsOnly /> : <section className="canvas-panel model-panel">{selected ? <ModelWorkspace project={selected} model={model} grids={grids} onImport={() => setShowImport(true)} onGenerateGrid={() => setShowGenerateGrid(true)} sectionRequest={modelSectionRequest} /> : <div className="canvas-empty"><div className="reservoir-placeholder"><span /><span /><span /><span /></div><div><h3>Start with a governed reservoir project</h3><p>Create a project, preserve original model inputs, and build evidence-backed simulation variants.</p><button className="button primary" onClick={() => setShowCreate(true)}>Create project</button></div></div>}</section>}
      </main>

      <aside className="inspector" aria-label="Context inspector"><div className="inspector-tabs"><button className="active">Selection</button><button>Evidence</button></div><div className="inspector-content"><span className="eyebrow">Current context</span><h3>{model?.name ?? selected?.name ?? "Nothing selected"}</h3>{selected ? <dl><dt>Project ID</dt><dd>{selected.project_id.slice(0, 18)}…</dd><dt>Model status</dt><dd>{model?.status ?? "No model"}</dd><dt>Unit system</dt><dd>{model?.unit_system ?? "—"}</dd><dt>Source checksum</dt><dd>{model?.source_hash ?? "—"}</dd><dt>Inspection checksum</dt><dd>{model?.inspection_hash ?? "—"}</dd><dt>Input policy</dt><dd>Immutable originals</dd><dt>Last updated</dt><dd>{new Date(model?.updated_at ?? selected.updated_at).toLocaleString()}</dd></dl> : <p>Select or create a project to inspect its model, lineage, and evidence.</p>}</div></aside>

      <footer className="task-panel"><div className="task-tabs"><button className="active">Problems <span>0</span></button><button>Output</button><button>Convergence</button><button>Tasks</button><button>Agent investigations</button></div><div className="task-summary"><span className="ok-dot" /> No active tasks <button>Expand panel</button></div></footer>
      {showCreate && <ProjectDialog onClose={() => setShowCreate(false)} onCreated={projectCreated} />}
      {showImport && selected && <ImportDialog project={selected} onClose={() => setShowImport(false)} onImported={modelImported} />}
      {showGenerateGrid && selected && <GenerateGridDialog project={selected} onClose={() => setShowGenerateGrid(false)} onGenerated={gridGenerated} />}
    </div>
  );
}
