"""M5.3 local browser workspace backed only by M5.1 service methods."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hmac
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def studio_page():
    return r'''<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyMRST Research Studio</title><style>
:root{--bg:#07111f;--panel:#0d1b2d;--line:#20344d;--text:#dbeafe;--muted:#8da4bf;--cyan:#22d3ee;--green:#34d399;--amber:#fbbf24;--red:#fb7185}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#123154 0,#07111f 38%);color:var(--text);font:14px Inter,ui-sans-serif,system-ui}
header{height:70px;display:flex;align-items:center;justify-content:space-between;padding:0 28px;border-bottom:1px solid var(--line);background:#07111fdd;backdrop-filter:blur(14px);position:sticky;top:0;z-index:2}
.brand{font-size:19px;font-weight:750;letter-spacing:.02em}.brand i{color:var(--cyan);font-style:normal}.live{padding:7px 11px;border:1px solid #166534;border-radius:20px;color:var(--green);background:#052e1a}
main{display:grid;grid-template-columns:250px 1fr;min-height:calc(100vh - 70px)}nav{padding:22px 16px;border-right:1px solid var(--line);background:#081423}.nav{padding:11px 13px;border-radius:9px;color:var(--muted);margin:4px}.nav.active{background:#123050;color:#fff;border:1px solid #1f527e}
.workspace{padding:26px;max-width:1500px}.heading{display:flex;justify-content:space-between;align-items:end;margin-bottom:20px}h1{font-size:25px;margin:0 0 5px}p{color:var(--muted);margin:0}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:linear-gradient(150deg,#10233a,#0b192a);border:1px solid var(--line);border-radius:13px;padding:17px;box-shadow:0 16px 30px #0003}.metric{font-size:26px;font-weight:750;margin-top:7px}.cyan{color:var(--cyan)}
.lower{display:grid;grid-template-columns:1.45fr 1fr;gap:14px;margin-top:14px}.title{font-weight:700;font-size:15px;margin-bottom:14px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:11px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.pill{padding:4px 8px;border-radius:12px;font-size:11px;background:#17304c}.completed{color:var(--green)}.failed{color:var(--red)}.running{color:var(--cyan)}.queued{color:var(--amber)}
.progress{height:7px;background:#192c42;border-radius:8px;overflow:hidden}.progress b{display:block;height:100%;background:linear-gradient(90deg,#0891b2,#22d3ee)}.empty{color:var(--muted);padding:28px;text-align:center;border:1px dashed var(--line);border-radius:10px}.limits{display:grid;grid-template-columns:1fr 1fr;gap:8px}.limit{background:#071525;padding:10px;border-radius:8px}.limit strong{display:block;color:#fff}.refresh{border:1px solid #256088;background:#102c48;color:#dff8ff;border-radius:8px;padding:8px 12px;cursor:pointer}
.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:8px;margin-bottom:12px}input,select{background:#071525;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:9px}.viewer{width:100%;height:580px;border:1px solid var(--line);border-radius:10px;background:#071525}.action{background:#0e7490;color:white;border:0;border-radius:8px;padding:9px 14px;cursor:pointer}
@media(max-width:900px){main{grid-template-columns:1fr}nav{display:none}.grid{grid-template-columns:1fr 1fr}.lower{grid-template-columns:1fr}}
</style></head><body><header><div class="brand"><i>PyMRST</i> Research Studio</div><div id="health" class="live">● Local service</div></header>
<main><nav><div class="nav active">Overview</div><div class="nav">Projects</div><div class="nav">Experiments</div><div class="nav">Runs & checkpoints</div><div class="nav">Scenario comparison</div><div class="nav">Evidence graph</div><div class="nav">Benchmark portal</div><div class="nav">Research packages</div></nav>
<section class="workspace"><div class="heading"><div><h1>Simulation operations</h1><p>Runs, evidence and engineering decisions from one service boundary.</p></div><button class="refresh" onclick="load()">Refresh</button></div>
<div class="grid"><div class="card"><span>Projects</span><div class="metric cyan" id="projects">0</div></div><div class="card"><span>Runs</span><div class="metric" id="runs">0</div></div><div class="card"><span>Completed</span><div class="metric completed" id="completed">0</div></div><div class="card"><span>Evidence events</span><div class="metric" id="events">0</div></div></div>
<div class="lower"><div class="card"><div class="title">Run monitor</div><table><thead><tr><th>Workflow</th><th>Status</th><th>Progress</th><th>Elapsed</th></tr></thead><tbody id="runrows"></tbody></table><div id="noruns" class="empty">No runs yet. Submit through the service or notebook API.</div></div>
<div class="card"><div class="title">Teaching runtime guardrails</div><p style="margin-bottom:14px">Small, bounded instructional cases. This is not a field-scale browser solver.</p><div class="limits" id="limits"></div></div></div>
<div class="lower"><div class="card"><div class="title">Evidence timeline</div><div id="timeline" class="empty">Lifecycle evidence will appear here.</div></div><div class="card"><div class="title">Publication safety</div><p>Benchmark evidence remains internal unless explicitly promoted. Public endpoints filter all internal and candidate records.</p></div></div>
<div class="card" style="margin-top:14px"><div class="title">Run a registered workflow</div><div class="controls"><select id="projectSelect"></select><select id="workflowSelect"></select><button class="action" onclick="submitRun()">Start run</button></div><div id="actionMessage" style="color:var(--muted)">Only host-registered workflows can execute.</div></div>
<div class="card" style="margin-top:14px"><div class="title">Reservoir result workspace</div><div class="controls" style="grid-template-columns:1fr auto"><select id="resultSelect" onchange="showResult()"></select><button class="refresh" onclick="showResult()">Open viewer</button></div><iframe id="resultViewer" class="viewer" title="Interactive reservoir result"></iframe><div id="noresult" class="empty">Register a completed member store and browser scene to inspect fields, slices, wells and timesteps.</div></div>
</section></main><script>
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function load(){try{const [s,t,results]=await Promise.all([fetch('/api/snapshot').then(r=>r.json()),fetch('/api/teaching').then(r=>r.json()),fetch('/api/results').then(r=>r.json())]);
document.querySelector('#projects').textContent=s.projects.length;document.querySelector('#runs').textContent=s.runs.length;document.querySelector('#completed').textContent=s.runs.filter(r=>r.status==='completed').length;document.querySelector('#events').textContent=s.runs.reduce((n,r)=>n+r.events.length,0);
const rows=document.querySelector('#runrows');rows.innerHTML=s.runs.slice().reverse().map(r=>`<tr><td>${esc(r.workflow_id)}</td><td class="${esc(r.status)}">${esc(r.status)}</td><td><div class="progress"><b style="width:${100*r.progress}%"></b></div></td><td>${r.elapsed_seconds.toFixed(2)}s</td></tr>`).join('');document.querySelector('#noruns').style.display=s.runs.length?'none':'block';
document.querySelector('#limits').innerHTML=Object.entries(t.limits).map(([k,v])=>`<div class="limit"><span>${esc(k.replaceAll('_',' '))}</span><strong>${esc(v)}</strong></div>`).join('');
document.querySelector('#projectSelect').innerHTML=s.projects.map(p=>`<option value="${esc(p.project_id)}">${esc(p.name)}</option>`).join('');document.querySelector('#workflowSelect').innerHTML=s.workflows.map(w=>`<option value="${esc(w.workflow_id)}">${esc(w.title)}</option>`).join('');document.querySelector('#resultSelect').innerHTML=results.map(r=>`<option value="${esc(r.result_id)}" data-viewer="${r.viewer_available}">${esc(r.title)}</option>`).join('');document.querySelector('#noresult').style.display=results.length?'none':'block';
const events=s.runs.flatMap(r=>r.events.map(e=>({...e,run:r.run_id}))).slice(-6).reverse();document.querySelector('#timeline').innerHTML=events.length?events.map(e=>`<div style="padding:9px;border-bottom:1px solid var(--line)"><b>${esc(e.kind)}</b><br><small>${esc(e.message)} · ${Math.round(e.progress*100)}%</small></div>`).join(''):'Lifecycle evidence will appear here.';
document.querySelector('#health').textContent='● Local service';}catch(e){document.querySelector('#health').textContent='● Service unavailable';document.querySelector('#health').style.color='var(--red)'}}load();setInterval(load,2000);
async function submitRun(){const p=document.querySelector('#projectSelect').value,w=document.querySelector('#workflowSelect').value,m=document.querySelector('#actionMessage');if(!p||!w){m.textContent='Create a project and register a workflow first.';return}const r=await fetch('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:p,workflow_id:w,parameters:{}})});const v=await r.json();m.textContent=r.ok?`Submitted ${v.run_id}`:`Error: ${v.error}`;load()}
function showResult(){const id=document.querySelector('#resultSelect').value,frame=document.querySelector('#resultViewer');if(id){frame.src=`/api/results/${encodeURIComponent(id)}/viewer`}}
</script></body></html>'''


class _Handler(BaseHTTPRequestHandler):
    server_version = "PyMRSTStudio/1"

    def _json(self, value, status=200):
        payload = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(payload)

    def _authorized(self):
        token = self.server.auth_token
        if not token:
            return True
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {token}"
        if hmac.compare_digest(supplied, expected):
            return True
        self._json({"error": "authentication required"}, 401)
        return False

    def do_GET(self):
        if not self._authorized():
            return
        path = urlparse(self.path).path
        if path == "/":
            payload = studio_page().encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload))); self.end_headers()
            self.wfile.write(payload); return
        if path == "/api/snapshot":
            self._json(self.server.service.snapshot()); return
        if path == "/api/teaching":
            self._json(self.server.teaching.disclosure()); return
        if path == "/api/benchmarks":
            self._json([value.to_dict(include_path=False)
                        for value in self.server.portal.entries(public_only=True)]); return
        if path == "/api/results":
            self._json([self.server.results.metadata(value.result_id)
                        for value in self.server.results.entries()]); return
        if path.startswith("/api/results/"):
            parts = path.strip("/").split("/")
            try:
                if len(parts) == 3:
                    self._json(self.server.results.metadata(parts[2]))
                elif len(parts) == 4 and parts[3] == "array":
                    query = parse_qs(urlparse(self.path).query)
                    required = {name: query[name][0] for name in ("category", "name")}
                    self._json(self.server.results.array(
                        parts[2], required["category"], required["name"],
                        time_index=(int(query["time"][0]) if "time" in query else None),
                        start=int(query.get("start", [0])[0]),
                        stop=(int(query["stop"][0]) if "stop" in query else None)))
                elif len(parts) == 4 and parts[3] == "viewer":
                    payload = self.server.results.viewer(parts[2]).read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Content-Security-Policy",
                                     "default-src 'self' https: data: 'unsafe-inline' 'unsafe-eval'")
                    self.end_headers(); self.wfile.write(payload)
                else:
                    self._json({"error": "not found"}, 404)
            except (KeyError, ValueError) as error:
                self._json({"error": str(error)}, 400)
            return
        if path.startswith("/api/resources/"):
            parts = path.strip("/").split("/")
            try:
                if len(parts) == 3:
                    self._json(list(self.server.service.resources(parts[2])))
                elif len(parts) == 4:
                    self._json(self.server.service.resource(parts[2], parts[3]))
                else:
                    self._json({"error": "not found"}, 404)
            except (KeyError, ValueError) as error:
                self._json({"error": str(error)}, 404)
            return
        if path.startswith("/api/runs/"):
            run_id = path.rsplit("/", 1)[-1]
            try: self._json(self.server.service.run(run_id).to_dict())
            except KeyError as error: self._json({"error": str(error)}, 404)
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._authorized():
            return
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024*1024:
            self._json({"error": "request too large"}, 413); return
        try: body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError: self._json({"error": "invalid JSON"}, 400); return
        try:
            if path == "/api/projects":
                value = self.server.service.create_project(
                    body["name"], description=body.get("description", ""),
                    tags=body.get("tags", ())).to_dict()
            elif path == "/api/runs":
                value = self.server.service.submit(
                    body["project_id"], body["workflow_id"],
                    body.get("parameters", {})).to_dict()
            elif path.endswith("/cancel") and path.startswith("/api/runs/"):
                run_id = path.split("/")[3]; self.server.service.cancel(run_id)
                value = {"run_id": run_id, "cancellation_requested": True}
            elif path.endswith("/resume") and path.startswith("/api/runs/"):
                run_id = path.split("/")[3]
                value = self.server.service.resume(
                    run_id, parameters=body.get("parameters", {})).to_dict()
            else:
                self._json({"error": "not found"}, 404); return
            self._json(value, 201)
        except (KeyError, ValueError) as error:
            self._json({"error": str(error)}, 400)

    def log_message(self, format, *args):
        pass


class ResearchStudioHTTPServer(ThreadingHTTPServer):
    def __init__(self, address, service, *, teaching, portal, results=None,
                 auth_token="", allow_remote=False):
        host = address[0]
        is_loopback = host in {"127.0.0.1", "localhost", "::1"}
        if not is_loopback and not allow_remote:
            raise ValueError("loopback only unless allow_remote=True is explicit")
        if not is_loopback and not auth_token:
            raise ValueError("non-loopback binding requires bearer authentication")
        from .results import ResultCatalog
        self.service, self.teaching, self.portal = service, teaching, portal
        self.results = results or ResultCatalog((Path.cwd(),))
        self.auth_token = str(auth_token)
        super().__init__(address, _Handler)


__all__ = ["ResearchStudioHTTPServer", "studio_page"]
