import { useEffect, useMemo, useState } from "react";
import {
  api, type ConstructionWell, type FieldSource, type GridSnapshot,
  type ModelConstructionDocument, type ModelDraft, type ModelRevision,
  type ProjectRecord,
} from "../api/client";
import { GridCanvas } from "./GridCanvas";

const steps = ["Rock", "Fluid", "Initialization", "Wells", "Schedule", "Review"] as const;
const DAY = 86400;

function slug(label: string) { return label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, ""); }

function source(kind: FieldSource["kind"], value: number | null, array = "", scale = 1): FieldSource {
  return { kind, value, array, scale };
}

function initialDocument(grid: GridSnapshot): ModelConstructionDocument {
  const arrays = new Set(grid.arrays.map((item) => item.name));
  return {
    schema_version: 1,
    rock: {
      porosity: arrays.has("PORO") ? source("grid_array", null, "PORO") : source("uniform", 0.2),
      permeability_x: arrays.has("PERMX") ? source("grid_array", null, "PERMX") : source("uniform", 1e-13),
      permeability_y: grid.dimension > 1 ? (arrays.has("PERMY") ? source("grid_array", null, "PERMY") : source("uniform", 1e-13)) : null,
      permeability_z: grid.dimension > 2 ? (arrays.has("PERMZ") ? source("grid_array", null, "PERMZ") : source("uniform", 1e-14)) : null,
      net_to_gross: arrays.has("NTG") ? source("grid_array", null, "NTG") : null,
    },
    fluid: {
      phases: ["W", "O"], viscosity_pa_s: [1e-3, 5e-3], density_kg_m3: [1000, 800],
      relperm_exponent: [2, 2], compressibility_pa_inverse: [4e-10, 1e-9],
      reference_pressure_pa: 20e6,
    },
    initial_state: {
      pressure: source("uniform", 200, "", 1e5),
      saturations: { W: source("uniform", 0.2), O: source("uniform", 0.8) },
      dissolved_gas_ratio: null,
    },
    wells: [],
    schedule: {
      timestep_seconds: [30*DAY], step_control: [0],
      controls: [{ name: "Base controls", well_overrides: {} }],
    },
  };
}

function FieldEditor({ label, value, grid, unit, onChange }: {
  label: string; value: FieldSource; grid: GridSnapshot; unit: string;
  onChange: (value: FieldSource) => void;
}) {
  const arrays = grid.arrays.filter((item) => item.shape.length === 1 && item.shape[0] === grid.cell_count);
  const base = slug(label);
  return <div className="source-editor">
    <label htmlFor={`${base}-kind`}><span>{label}</span><select id={`${base}-kind`} name={`${base}-kind`} value={value.kind} onChange={(event) => onChange(event.target.value === "uniform" ? source("uniform", 0) : source("grid_array", null, arrays[0]?.name ?? ""))}><option value="uniform">Uniform</option><option value="grid_array">Grid property</option></select></label>
    {value.kind === "uniform" ? <label htmlFor={`${base}-value`}><span>Value · {unit}</span><input id={`${base}-value`} name={`${base}-value`} type="number" step="any" value={value.value ?? 0} onChange={(event) => onChange({ ...value, value: Number(event.target.value) })} /></label> : <label htmlFor={`${base}-array`}><span>Array</span><select id={`${base}-array`} name={`${base}-array`} value={value.array} onChange={(event) => onChange({ ...value, array: event.target.value })}>{arrays.map((item) => <option key={item.name}>{item.name}</option>)}</select></label>}
    <label htmlFor={`${base}-scale`}><span>SI multiplier</span><input id={`${base}-scale`} name={`${base}-scale`} type="number" step="any" value={value.scale} onChange={(event) => onChange({ ...value, scale: Number(event.target.value) })} /></label>
  </div>;
}

function VectorInput({ label, values, phases, scale = 1, unit, onChange }: {
  label: string; values: number[]; phases: string[]; scale?: number; unit: string;
  onChange: (values: number[]) => void;
}) {
  const base = slug(label);
  return <div className="vector-input"><span>{label} · {unit}</span><div>{phases.map((phase, index) => <label htmlFor={`${base}-${phase}`} key={phase}><b>{phase}</b><input id={`${base}-${phase}`} name={`${base}-${phase}`} type="number" step="any" value={(values[index] ?? 0)*scale} onChange={(event) => { const changed = [...values]; changed[index] = Number(event.target.value)/scale; onChange(changed); }} /></label>)}</div></div>;
}

function targetScale(type: string) { return type === "bhp" ? 1e5 : 1/DAY; }
function targetUnit(type: string) { return type === "bhp" ? "bar" : "m³/day"; }

function WellCard({ well, index, phases, cells, onChange, onRemove }: {
  well: ConstructionWell; index: number; phases: string[]; cells: number;
  onChange: (well: ConstructionWell) => void; onRemove: () => void;
}) {
  const scale = targetScale(well.control_type);
  const base = `well-${index}`;
  return <article className="well-card"><div className="well-card-head"><label htmlFor={`${base}-name`}><span>Well name</span><input id={`${base}-name`} name={`${base}-name`} value={well.name} onChange={(event) => onChange({ ...well, name: event.target.value })} /></label><button className="button danger" onClick={onRemove}>Remove</button></div><div className="well-grid">
    <label htmlFor={`${base}-cells`}><span>Completion cells · 0–{cells-1}</span><input id={`${base}-cells`} name={`${base}-cells`} value={well.cells.join(", ")} onChange={(event) => onChange({ ...well, cells: event.target.value.split(",").map((item) => Number(item.trim())).filter(Number.isInteger) })} /></label>
    <label htmlFor={`${base}-control`}><span>Control</span><select id={`${base}-control`} name={`${base}-control`} value={well.control_type} onChange={(event) => { const type = event.target.value; onChange({ ...well, control_type: type, target_si: type === "bhp" ? 150e5 : 100/DAY }); }}><option value="bhp">Bottom-hole pressure</option><option value="rate">Total rate</option><option value="orat">Oil rate</option><option value="wrat">Water rate</option><option value="grat">Gas rate</option><option value="lrat">Liquid rate</option></select></label>
    <label htmlFor={`${base}-target`}><span>Target · {targetUnit(well.control_type)}</span><input id={`${base}-target`} name={`${base}-target`} type="number" step="any" value={well.target_si/scale} onChange={(event) => onChange({ ...well, target_si: Number(event.target.value)*scale })} /></label>
    <label htmlFor={`${base}-direction`}><span>Direction</span><select id={`${base}-direction`} name={`${base}-direction`} value={well.direction} onChange={(event) => onChange({ ...well, direction: event.target.value as "x"|"y"|"z" })}><option>x</option><option>y</option><option>z</option></select></label>
    <label htmlFor={`${base}-radius`}><span>Radius · m</span><input id={`${base}-radius`} name={`${base}-radius`} type="number" step="any" value={well.radius_m} onChange={(event) => onChange({ ...well, radius_m: Number(event.target.value) })} /></label>
    <label htmlFor={`${base}-sign`}><span>Flow sign</span><select id={`${base}-sign`} name={`${base}-sign`} value={well.sign} onChange={(event) => onChange({ ...well, sign: Number(event.target.value) })}><option value="1">Injector (+)</option><option value="-1">Producer (−)</option><option value="0">Infer from target</option></select></label>
  </div><VectorInput label={`${base} injection composition`} unit="fraction" phases={phases} values={well.composition} onChange={(composition) => onChange({ ...well, composition })} /></article>;
}

export function ConstructionWorkspace({ project, grids, stepRequest }: { project: ProjectRecord; grids: GridSnapshot[]; stepRequest?: { step?: number; token: number } }) {
  const [gridId, setGridId] = useState(grids[0]?.snapshot_id ?? "");
  const grid = grids.find((item) => item.snapshot_id === gridId) ?? grids[0];
  const [document, setDocument] = useState<ModelConstructionDocument | null>(() => grid ? initialDocument(grid) : null);
  const [drafts, setDrafts] = useState<ModelDraft[]>([]); const [draft, setDraft] = useState<ModelDraft>();
  const [revisions, setRevisions] = useState<ModelRevision[]>([]);
  const [step, setStep] = useState(stepRequest?.step ?? 0);
  const [name, setName] = useState("Reservoir model"); const [dirty, setDirty] = useState(true);
  const [busy, setBusy] = useState(false); const [error, setError] = useState(""); const [notice, setNotice] = useState("");
  const edit = (change: (value: ModelConstructionDocument) => void) => {
    setDocument((current) => { if (!current) return current; const next = structuredClone(current); change(next); return next; });
    setDirty(true); setNotice("");
  };
  useEffect(() => { const controller = new AbortController(); api.modelDrafts(project.project_id, controller.signal).then(setDrafts).catch((caught) => { if (caught.name !== "AbortError") setError(caught.message); }); return () => controller.abort(); }, [project.project_id]);
  useEffect(() => { if (!gridId && grids[0]) setGridId(grids[0].snapshot_id); }, [grids, gridId]);
  useEffect(() => {
    if (stepRequest?.step !== undefined) setStep(stepRequest.step);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stepRequest?.token]);
  function choose(value: ModelDraft) { setDraft(value); setDocument(structuredClone(value.document)); setGridId(value.grid_snapshot_id); setName(value.name); setDirty(false); setError(""); api.modelRevisions(value.draft_id).then(setRevisions).catch((caught) => setError(caught.message)); }
  function newDraft() { if (!grid) return; setDraft(undefined); setDocument(initialDocument(grid)); setName("Reservoir model"); setRevisions([]); setDirty(true); setStep(0); }
  async function save() { if (!grid || !document) return; setBusy(true); setError(""); setNotice(""); try { const value = draft ? await api.updateModelDraft(draft.draft_id, { expected_version: draft.version, reason: "guided model-builder edit", document }) : await api.createModelDraft(project.project_id, { name, grid_snapshot_id: grid.snapshot_id, document }); setDraft(value); setDrafts((current) => [...current.filter((item) => item.draft_id !== value.draft_id), value]); setDirty(false); setNotice(value.status === "ready" ? `Draft v${value.version} passed engineering validation.` : `Draft v${value.version} has ${value.issues.length} blocking finding(s).`); } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not save draft"); } finally { setBusy(false); } }
  async function materialize() { if (!draft || dirty) return; setBusy(true); setError(""); try { const revision = await api.materializeModelDraft(draft.draft_id); setRevisions((current) => current.some((item) => item.revision_id === revision.revision_id) ? current : [...current, revision]); setNotice(`Immutable revision ${revision.revision_number} created.`); } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not materialize revision"); } finally { setBusy(false); } }
  const totalDays = useMemo(() => (document?.schedule.timestep_seconds.reduce((sum, item) => sum+item, 0) ?? 0)/DAY, [document]);
  if (!grid || !document) return <div className="empty-data construction-empty"><h3>Register a grid before constructing model physics</h3><p>The builder binds every cell property, completion and initial state to an immutable grid snapshot. Import or generate a grid first.</p></div>;
  const setPhases = (phases: string[]) => edit((value) => { const defaults: Record<string, [number, number, number, number, number]> = { W: [1e-3, 1000, 2, 4e-10, .2], O: [5e-3, 800, 2, 1e-9, .8], G: [2e-5, 100, 2, 1e-8, 0] }; value.fluid.phases = phases; value.fluid.viscosity_pa_s = phases.map((phase) => defaults[phase][0]); value.fluid.density_kg_m3 = phases.map((phase) => defaults[phase][1]); value.fluid.relperm_exponent = phases.map((phase) => defaults[phase][2]); value.fluid.compressibility_pa_inverse = phases.map((phase) => defaults[phase][3]); value.initial_state.saturations = Object.fromEntries(phases.map((phase) => [phase, source("uniform", defaults[phase][4])])); value.wells.forEach((well) => { well.composition = phases.map((phase) => phase === "W" ? 1 : 0); }); });
  return <div className="construction-room"><aside className="draft-rail"><div><span className="eyebrow">Versioned construction</span><h3>Model drafts</h3></div><button className="button primary" onClick={newDraft}>New draft</button>{drafts.map((item) => <button className={`draft-row ${draft?.draft_id === item.draft_id ? "active" : ""}`} key={item.draft_id} onClick={() => choose(item)}><strong>{item.name}</strong><span>v{item.version} · {item.status}</span><code>{item.document_hash.slice(0, 9)}</code></button>)}{!drafts.length && <p>No governed drafts yet.</p>}<div className="revision-list"><span className="eyebrow">Immutable revisions</span>{revisions.map((item) => <article key={item.revision_id}><strong>Revision {item.revision_number}</strong><span>{item.relative_path}</span></article>)}</div></aside><section className="builder"><header className="builder-header"><div><span className="eyebrow">Grid-bound model definition</span><input id="construction-name" name="construction-name" aria-label="Model name" className="builder-title" value={name} disabled={Boolean(draft)} onChange={(event) => { setName(event.target.value); setDirty(true); }} /></div><label htmlFor="construction-grid-snapshot">Grid snapshot<select id="construction-grid-snapshot" name="construction-grid-snapshot" value={grid.snapshot_id} disabled={Boolean(draft)} onChange={(event) => { const chosen = grids.find((item) => item.snapshot_id === event.target.value); if (chosen) { setGridId(chosen.snapshot_id); setDocument(initialDocument(chosen)); setDirty(true); } }}>{grids.map((item) => <option key={item.snapshot_id} value={item.snapshot_id}>{item.name} · {item.cell_count.toLocaleString()} cells</option>)}</select></label><div className={`draft-status ${draft?.status ?? "unsaved"}`}><i />{draft ? `v${draft.version} ${draft.status}` : "Unsaved"}{dirty && " · edited"}</div></header><GridCanvas grid={grid} compact /><nav className="builder-steps">{steps.map((label, index) => <button className={step === index ? "active" : index < step ? "complete" : ""} onClick={() => setStep(index)} key={label}><b>{index+1}</b><span>{label}</span></button>)}</nav><div className="builder-content">
    {step === 0 && <>
      <div className="section-lead"><h3>Rock properties</h3><p>Bind homogeneous values or cell arrays to SI-valued PyMRST rock fields.</p></div>
      <div className="source-grid">
        <FieldEditor label="Porosity" unit="fraction" value={document.rock.porosity} grid={grid} onChange={(item) => edit((value) => { value.rock.porosity = item; })} />
        <FieldEditor label="Permeability X" unit="m²" value={document.rock.permeability_x} grid={grid} onChange={(item) => edit((value) => { value.rock.permeability_x = item; })} />
        {document.rock.permeability_y && <FieldEditor label="Permeability Y" unit="m²" value={document.rock.permeability_y} grid={grid} onChange={(item) => edit((value) => { value.rock.permeability_y = item; })} />}
        {document.rock.permeability_z && <FieldEditor label="Permeability Z" unit="m²" value={document.rock.permeability_z} grid={grid} onChange={(item) => edit((value) => { value.rock.permeability_z = item; })} />}
      </div>
    </>}
    {step === 1 && <><div className="section-lead"><h3>Fluid system</h3><p>Define phase basis and simple ADI PVT/relative-permeability properties.</p></div><div className="phase-choice"><button className={document.fluid.phases.join("") === "WO" ? "active" : ""} onClick={() => setPhases(["W", "O"])}>Water–oil</button><button className={document.fluid.phases.join("") === "WOG" ? "active" : ""} onClick={() => setPhases(["W", "O", "G"])}>Water–oil–gas</button></div><VectorInput label="Viscosity" unit="cP" scale={1000} phases={document.fluid.phases} values={document.fluid.viscosity_pa_s} onChange={(items) => edit((value) => { value.fluid.viscosity_pa_s = items; })} /><VectorInput label="Surface density" unit="kg/m³" phases={document.fluid.phases} values={document.fluid.density_kg_m3} onChange={(items) => edit((value) => { value.fluid.density_kg_m3 = items; })} /><VectorInput label="Corey exponent" unit="—" phases={document.fluid.phases} values={document.fluid.relperm_exponent} onChange={(items) => edit((value) => { value.fluid.relperm_exponent = items; })} /><VectorInput label="Compressibility" unit="1/bar" scale={1e5} phases={document.fluid.phases} values={document.fluid.compressibility_pa_inverse} onChange={(items) => edit((value) => { value.fluid.compressibility_pa_inverse = items; })} /><label htmlFor="construction-reference-pressure" className="single-input"><span>Reference pressure · bar</span><input id="construction-reference-pressure" name="construction-reference-pressure" type="number" value={document.fluid.reference_pressure_pa/1e5} onChange={(event) => edit((value) => { value.fluid.reference_pressure_pa = Number(event.target.value)*1e5; })} /></label></>}
    {step === 2 && <><div className="section-lead"><h3>Initial reservoir state</h3><p>Initialize pressure and phase saturation on the same active-cell basis.</p></div><FieldEditor label="Reservoir pressure" unit="bar" value={document.initial_state.pressure} grid={grid} onChange={(item) => edit((value) => { value.initial_state.pressure = item; })} /><div className="source-grid">{document.fluid.phases.map((phase) => <FieldEditor key={phase} label={`${phase} saturation`} unit="fraction" value={document.initial_state.saturations[phase]} grid={grid} onChange={(item) => edit((value) => { value.initial_state.saturations[phase] = item; })} />)}</div></>}
    {step === 3 && <><div className="section-lead"><h3>Wells and completions</h3><p>Define completion cells, Peaceman geometry, phase composition and operating controls.</p><button className="button primary" onClick={() => edit((value) => { const index = value.wells.length+1; value.wells.push({ name: `W${index}`, cells: [Math.min(index-1, grid.cell_count-1)], control_type: "bhp", target_si: 150e5, radius_m: .1, direction: "z", composition: value.fluid.phases.map((phase) => phase === "W" ? 1 : 0), sign: 0, status: true, limits: {} }); })}>Add well</button></div><div className="well-list">{document.wells.map((well, index) => <WellCard key={index} well={well} index={index} phases={document.fluid.phases} cells={grid.cell_count} onChange={(item) => edit((value) => { value.wells[index] = item; })} onRemove={() => edit((value) => { value.wells.splice(index, 1); value.schedule.controls.forEach((control) => { delete control.well_overrides[well.name]; }); })} />)}{!document.wells.length && <div className="empty-builder">No wells defined. A closed reservoir model is valid; add wells for production or injection.</div>}</div></>}
    {step === 4 && <><div className="section-lead"><h3>Control schedule</h3><p>Specify report timesteps and map each step to a named control period.</p></div><div className="schedule-grid"><label htmlFor="schedule-timesteps"><span>Timestep lengths · days</span><textarea id="schedule-timesteps" name="schedule-timesteps" value={document.schedule.timestep_seconds.map((item) => item/DAY).join(", ")} onChange={(event) => edit((value) => { value.schedule.timestep_seconds = event.target.value.split(",").map((item) => Number(item.trim())*DAY).filter(Number.isFinite); })} /></label><label htmlFor="schedule-step-control"><span>Control index per step · zero-based</span><textarea id="schedule-step-control" name="schedule-step-control" value={document.schedule.step_control.join(", ")} onChange={(event) => edit((value) => { value.schedule.step_control = event.target.value.split(",").map((item) => Number(item.trim())).filter(Number.isInteger); })} /></label></div><div className="control-periods"><div className="table-heading"><div><span className="eyebrow">Operating periods</span><h3>{document.schedule.controls.length} control definition(s)</h3></div><button className="button ghost" onClick={() => edit((value) => { value.schedule.controls.push({ name: `Control ${value.schedule.controls.length}`, well_overrides: {} }); })}>Add period</button></div>{document.schedule.controls.map((control, controlIndex) => <article key={controlIndex}><div className="control-head"><b>{controlIndex}</b><input id={`control-${controlIndex}-name`} name={`control-${controlIndex}-name`} aria-label="Control period name" value={control.name} onChange={(event) => edit((value) => { value.schedule.controls[controlIndex].name = event.target.value; })} /><button disabled={document.schedule.controls.length === 1} onClick={() => edit((value) => { value.schedule.controls.splice(controlIndex, 1); value.schedule.step_control = value.schedule.step_control.map((item) => item === controlIndex ? 0 : item > controlIndex ? item-1 : item); })}>Remove</button></div>{document.wells.map((well) => { const override = control.well_overrides[well.name] ?? {}; const scale = targetScale(String(override.control_type ?? well.control_type)); const overrideBase = `control-${controlIndex}-${slug(well.name)}`; return <div className="override-row" key={well.name}><strong>{well.name}</strong><label htmlFor={`${overrideBase}-target`}><span>Override target · {targetUnit(String(override.control_type ?? well.control_type))}</span><input id={`${overrideBase}-target`} name={`${overrideBase}-target`} placeholder="Use base" value={override.target_si === undefined ? "" : Number(override.target_si)/scale} onChange={(event) => edit((value) => { const owner = value.schedule.controls[controlIndex]; owner.well_overrides[well.name] ??= {}; if (event.target.value === "") delete owner.well_overrides[well.name].target_si; else owner.well_overrides[well.name].target_si = Number(event.target.value)*scale; if (!Object.keys(owner.well_overrides[well.name]).length) delete owner.well_overrides[well.name]; })} /></label><label htmlFor={`${overrideBase}-status`}><span>Status</span><select id={`${overrideBase}-status`} name={`${overrideBase}-status`} value={override.status === undefined ? "base" : String(override.status)} onChange={(event) => edit((value) => { const owner = value.schedule.controls[controlIndex]; owner.well_overrides[well.name] ??= {}; if (event.target.value === "base") delete owner.well_overrides[well.name].status; else owner.well_overrides[well.name].status = event.target.value === "true"; if (!Object.keys(owner.well_overrides[well.name]).length) delete owner.well_overrides[well.name]; })}><option value="base">Use base</option><option value="true">Open</option><option value="false">Closed</option></select></label></div>; })}</article>)}</div></>}
    {step === 5 && <><div className="section-lead"><h3>Engineering review</h3><p>Save to run server-side validation. Materialization seals a checksum-bound, read-only revision.</p></div><div className="review-grid"><article><span>Grid</span><strong>{grid.cell_count.toLocaleString()} active cells</strong><small>{grid.dimension}D · {grid.content_hash.slice(0, 12)}</small></article><article><span>Physics</span><strong>{document.fluid.phases.join("–")} · {document.wells.length} wells</strong><small>{document.rock.permeability_z ? "Directional permeability" : "Isotropic permeability"}</small></article><article><span>Schedule</span><strong>{document.schedule.timestep_seconds.length} steps · {totalDays.toLocaleString()} days</strong><small>{document.schedule.controls.length} control periods</small></article><article><span>Lineage</span><strong>{draft ? `Draft v${draft.version}` : "Not yet saved"}</strong><small>{draft?.document_hash.slice(0, 16) ?? "Pending checksum"}</small></article></div>{draft?.issues.length ? <div className="issue-list">{draft.issues.map((issue, index) => <article className={`issue ${issue.severity}`} key={`${issue.code}-${index}`}><span>{issue.severity}</span><div><strong>{issue.message}</strong><small>{issue.section} · {issue.code}</small></div></article>)}</div> : draft && <div className="pass-panel"><i>✓</i><div><strong>Construction checks passed</strong><p>Rock, fluid, state, wells and schedule share a consistent active-cell and phase basis.</p></div></div>}</>}
  </div><footer className="builder-actions"><button className="button ghost" disabled={step === 0} onClick={() => setStep((value) => Math.max(0, value-1))}>Previous</button><span>{notice || (dirty ? "Unsaved engineering changes" : "Draft is synchronized")}</span><button className="button ghost" disabled={busy} onClick={save}>{busy ? "Working…" : draft ? "Save new draft version" : "Save and validate"}</button>{step < steps.length-1 ? <button className="button primary" onClick={() => setStep((value) => Math.min(steps.length-1, value+1))}>Next: {steps[step+1]}</button> : <button className="button primary" disabled={!draft || dirty || draft.status !== "ready" || busy} onClick={materialize}>Materialize immutable revision</button>}</footer>{error && <div className="form-error builder-error">{error}</div>}</section></div>;
}
