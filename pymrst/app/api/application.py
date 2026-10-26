"""FastAPI composition kept optional so the numerical package stays lightweight."""

from __future__ import annotations

import zipfile
import re
import hmac

from ..application import ApplicationContext
from ..schemas.project import ProjectCreate
from ..schemas.workspace import WorkspaceState
from ..schemas.backup import BackupPolicy
from ..schemas.security import Capability, DeploymentMode
from ..schemas.deployment import ServerDeploymentSettings
from ..schemas.workload import WorkloadRequest
from ..services.security_service import AuthenticationError, AuthorizationError
from .middleware import (
    AuthenticationFailureLimiter, RequestSizeLimitMiddleware,
    SecureTransportMiddleware, SecurityHeadersMiddleware,
)
from ..schemas.model import PatchOperation
from ..schemas.construction import ModelConstructionDocument
from ..schemas.experiment import BranchRequest, ExecutionTarget, ExperimentRequest
from ..schemas.agent import InvestigationRequest
from pymrst.research_studio.agents.providers import ProviderConfiguration


def create_fastapi_app(
    context: ApplicationContext, deployment: ServerDeploymentSettings | None = None,
    *, desktop_bootstrap_token: str = "",
):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        from starlette.middleware.trustedhost import TrustedHostMiddleware
    except ImportError as error:
        raise ImportError(
            "PyMRST application API requires `pip install pymrst[app]`"
        ) from error

    # Endpoint annotations are postponed by ``from __future__ import
    # annotations``. Publish the optional type only after FastAPI is available
    # so its dependency resolver recognizes Request rather than inventing a
    # query parameter named ``request``.
    globals()["Request"] = Request

    deployment = deployment or ServerDeploymentSettings.from_environment()
    if context.security.mode is DeploymentMode.SHARED:
        deployment.validate_shared_mode()
    app = FastAPI(
        title="PyMRST Studio API", version="1.0.0",
        description="Local and remote project, task, and notification services.",
        docs_url="/docs" if deployment.expose_api_documentation else None,
        redoc_url="/redoc" if deployment.expose_api_documentation else None,
        openapi_url="/openapi.json" if deployment.expose_api_documentation else None,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=list(deployment.allowed_origins),
        allow_credentials=False, allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Accept", "X-PyMRST-Desktop-Token"],
        max_age=600,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(deployment.allowed_hosts))
    app.add_middleware(SecureTransportMiddleware, settings=deployment)
    app.add_middleware(
        RequestSizeLimitMiddleware, maximum_bytes=deployment.maximum_request_bytes,
    )
    authentication_limiter = AuthenticationFailureLimiter(
        deployment.authentication_failures_per_minute,
    )
    desktop_bootstrap_token = str(desktop_bootstrap_token).strip()
    if desktop_bootstrap_token and len(desktop_bootstrap_token) < 32:
        raise ValueError("desktop bootstrap token must contain at least 32 characters")

    def _client_address(request: Request) -> str:
        peer = request.client.host if request.client else "unknown"
        return deployment.client_address(
            peer_address=peer,
            forwarded_for=request.headers.get("x-forwarded-for", ""),
        )

    _resource_pattern = re.compile(
        r"/api/v1/(projects|models|patches|grids|model-drafts|model-revisions|"
        r"experiments|experiment-executions|experiment-branches|investigations|"
        r"investigation-executions|findings|variant-proposals|"
        r"investigation-narratives|run-configurations|runs|result-states|tasks|"
        r"run-comparisons)/([^/]+)"
    )

    def _required_capability(method: str, path: str) -> Capability:
        if method in {"GET", "HEAD", "OPTIONS"}:
            return Capability.VIEW
        if any(marker in path for marker in (
            "/submit", "/cancel", "/resume", "/executions", "/analyze",
            "/qualifications", "/branches",
        )):
            return Capability.EXECUTE
        if any(marker in path for marker in (
            "/reports", "/review", "/synthesize", "/conflicts/",
            "/variant-proposals", "/decision", "/exports",
        )):
            return Capability.PUBLISH
        return Capability.EDIT

    @app.middleware("http")
    async def security_boundary(request: Request, call_next):
        # CORSMiddleware sits inside this middleware in Starlette's stack (it
        # was registered first, so it ends up innermost), so a preflight must
        # be let through here or the browser's OPTIONS probe is rejected by
        # this auth check before CORSMiddleware ever answers it — the real
        # request never gets sent, and the desktop app sees a bare "Load
        # failed" with no indication it was a CORS preflight problem.
        if request.method == "OPTIONS":
            return await call_next(request)
        path = request.url.path
        if path == "/api/v1/health" and not desktop_bootstrap_token:
            return await call_next(request)
        try:
            if desktop_bootstrap_token:
                supplied = request.headers.get("x-pymrst-desktop-token", "")
                if not hmac.compare_digest(supplied, desktop_bootstrap_token):
                    raise AuthenticationError("invalid desktop application credential")
            if context.security.mode is DeploymentMode.LOCAL:
                actor = context.security.local_actor()
            else:
                authorization = request.headers.get("authorization", "")
                peer = _client_address(request)
                if not authentication_limiter.allow(peer):
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "too many failed authentication attempts"},
                        headers={"Retry-After": "60"},
                    )
                scheme, _, credential = authorization.partition(" ")
                if scheme.casefold() != "bearer" or not credential.strip():
                    raise AuthenticationError("shared deployment requires Authorization: Bearer <token>")
                actor = context.security.authenticate(credential)
                authentication_limiter.reset(peer)
            request.state.security_actor = actor
            if context.security.mode is DeploymentMode.SHARED and not path.startswith("/api/v1/security"):
                if path.startswith("/api/v1/system/") or path.startswith("/api/v1/reasoning-provider"):
                    capability, project_id = Capability.SYSTEM_ADMIN, ""
                else:
                    match = _resource_pattern.search(path)
                    project_id = (
                        context.security.project_for_resource(match.group(1), match.group(2))
                        if match else ""
                    )
                    capability = _required_capability(request.method, path)
                    # Collection/global state APIs have no safe project boundary.
                    if not project_id:
                        capability = Capability.SYSTEM_ADMIN
                context.security.require(
                    actor, capability, project_id=project_id, action="api.request",
                    resource_type="http", resource_id=f"{request.method} {path}",
                )
        except AuthenticationError as error:
            peer = _client_address(request)
            authentication_limiter.record_failure(peer)
            context.security.audit(
                "anonymous", "api.authenticate", "http", f"{request.method} {path}",
                outcome="denied", details={"reason": str(error)},
            )
            return JSONResponse(status_code=401, content={"detail": str(error)},
                                headers={"WWW-Authenticate": "Bearer"})
        except AuthorizationError as error:
            return JSONResponse(status_code=403, content={"detail": str(error)})
        return await call_next(request)

    def _actor(request: Request):
        return request.state.security_actor

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "database_schema": context.database.schema_version}

    @app.get("/api/v1/security/mode")
    def security_mode():
        return {"mode": context.security.mode.value}

    @app.get("/api/v1/security/deployment")
    def security_deployment(request: Request):
        try:
            context.security.require(_actor(request), Capability.SYSTEM_ADMIN,
                                     action="security.deployment.read")
            return deployment.to_dict()
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error

    @app.put("/api/v1/security/mode")
    def set_security_mode(payload: dict, request: Request):
        try:
            requested = DeploymentMode(payload["mode"])
            if requested is DeploymentMode.SHARED:
                deployment.validate_shared_mode()
            return {"mode": context.security.set_mode(
                requested, actor=_actor(request),
            ).value}
        except KeyError as error:
            raise HTTPException(status_code=400, detail="mode is required") from error
        except (AuthenticationError, AuthorizationError) as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/security/me")
    def security_identity(request: Request):
        actor = _actor(request)
        return {**actor.to_dict(), "capabilities": [
            item.value for item in context.security.capabilities(actor)
        ]}

    @app.get("/api/v1/security/me/projects")
    def security_visible_projects(request: Request):
        actor = _actor(request)
        return {"project_ids": list(context.security.visible_project_ids(actor))}

    @app.get("/api/v1/security/users")
    def security_users(request: Request):
        try:
            context.security.require(_actor(request), Capability.SYSTEM_ADMIN,
                                     action="security.users.list")
            return [item.to_dict() for item in context.security.users()]
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error

    @app.post("/api/v1/security/users", status_code=201)
    def create_security_user(payload: dict, request: Request):
        try:
            return context.security.create_user(
                payload["display_name"], payload["email"], payload["system_role"],
                actor=_actor(request),
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=400, detail=f"missing field: {error.args[0]}") from error
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/v1/security/users/{user_id}/active")
    def set_security_user_active(user_id: str, payload: dict, request: Request):
        try:
            if not isinstance(payload.get("active"), bool):
                raise ValueError("active must be a boolean")
            return context.security.set_user_active(
                user_id, payload["active"], actor=_actor(request),
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.put("/api/v1/security/projects/{project_id}/members/{user_id}")
    def set_security_membership(project_id: str, user_id: str, payload: dict, request: Request):
        try:
            return context.security.set_membership(
                project_id, user_id, payload["project_role"], actor=_actor(request),
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/security/memberships")
    def security_memberships(request: Request, user_id: str | None = None):
        try:
            actor = _actor(request)
            if user_id and user_id != actor.user_id:
                context.security.require(actor, Capability.SYSTEM_ADMIN,
                                         action="security.memberships.list")
            return [item.to_dict() for item in context.security.memberships(user_id or actor.user_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error

    @app.post("/api/v1/security/users/{user_id}/tokens", status_code=201)
    def issue_security_token(user_id: str, payload: dict, request: Request):
        try:
            return context.security.issue_token(
                user_id, payload["label"], payload["scopes"],
                expires_at=payload.get("expires_at", ""), actor=_actor(request),
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=400, detail=f"missing field: {error.args[0]}") from error
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/security/tokens")
    def security_tokens(request: Request, user_id: str | None = None):
        try:
            actor = _actor(request)
            owner = user_id or actor.user_id
            if owner != actor.user_id:
                context.security.require(actor, Capability.SYSTEM_ADMIN,
                                         action="security.tokens.list")
            return [item.to_dict() for item in context.security.tokens(owner)]
        except (KeyError, AuthorizationError) as error:
            raise HTTPException(status_code=403 if isinstance(error, AuthorizationError) else 404,
                                detail=str(error)) from error

    @app.post("/api/v1/security/tokens/{token_id}/revoke")
    def revoke_security_token(token_id: str, request: Request):
        try:
            return context.security.revoke_token(token_id, actor=_actor(request)).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/security/audit")
    def security_audit(request: Request, project_id: str = "", actor_id: str = "", limit: int = 500):
        try:
            actor = _actor(request)
            context.security.require(actor, Capability.AUDIT, project_id=project_id,
                                     action="security.audit.list")
            return [item.to_dict() for item in context.security.audit_events(
                project_id=project_id, actor_id=actor_id, limit=limit,
            )]
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/security/audit/verify")
    def verify_security_audit(request: Request):
        try:
            context.security.require(_actor(request), Capability.AUDIT,
                                     action="security.audit.verify")
            return context.security.verify_audit_chain()
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error

    @app.get("/api/v1/system/backups/plan")
    def workspace_backup_plan():
        try:
            return context.backups.plan().to_dict()
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/system/environment-qualification")
    def latest_environment_qualification():
        item = context.environment.latest()
        return None if item is None else item.to_dict()

    @app.get("/api/v1/system/environment-qualifications")
    def environment_qualification_history(limit: int = 20):
        try:
            return [item.to_dict() for item in context.environment.history(limit)]
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/v1/system/environment-qualifications", status_code=201)
    def run_environment_qualification():
        try:
            return context.environment.run().to_dict()
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/system/workload-catalog")
    def workload_catalog():
        return [item.to_dict() for item in context.workloads.catalog()]

    @app.get("/api/v1/examples")
    def example_catalog():
        return [item.to_dict() for item in context.examples.list()]

    @app.get("/api/v1/system/workload-qualifications")
    def workload_qualifications(limit: int = 50, workload_id: str = ""):
        try:
            return [item.to_dict() for item in context.workloads.list(
                limit=limit, workload_id=workload_id,
            )]
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/v1/system/workload-qualifications", status_code=202)
    def submit_workload_qualification(payload: dict):
        try:
            return context.workloads.submit(WorkloadRequest.from_dict(payload)).to_dict()
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/system/workload-qualifications/{qualification_id}")
    def workload_qualification(qualification_id: str):
        try:
            return context.workloads.get(qualification_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/system/workload-qualifications/{qualification_id}/cancel", status_code=202)
    def cancel_workload_qualification(qualification_id: str):
        try:
            return context.workloads.cancel(qualification_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/system/workload-qualifications/{qualification_id}/verify")
    def verify_workload_qualification(qualification_id: str):
        try:
            return context.workloads.verify_evidence(qualification_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/system/backups")
    def workspace_backups():
        return [item.to_dict() for item in context.backups.list()]

    @app.post("/api/v1/system/backups", status_code=201)
    def create_workspace_backup():
        try:
            return context.backups.create().to_dict()
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/system/backup-jobs")
    def workspace_backup_jobs():
        return [item.to_dict() for item in context.backups.jobs()]

    @app.post("/api/v1/system/backup-jobs", status_code=202)
    def submit_workspace_backup_job():
        try:
            return context.backups.submit().to_dict()
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/system/backup-jobs/{job_id}")
    def workspace_backup_job(job_id: str):
        try:
            return context.backups.job(job_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/system/backup-jobs/{job_id}/cancel", status_code=202)
    def cancel_workspace_backup_job(job_id: str):
        try:
            return context.backups.cancel(job_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/system/backups/policy")
    def workspace_backup_policy():
        return context.backups.policy().to_dict()

    @app.put("/api/v1/system/backups/policy")
    def update_workspace_backup_policy(payload: dict):
        try:
            return context.backups.set_policy(BackupPolicy.from_dict(payload)).to_dict()
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/system/backup-qualifications")
    def workspace_backup_qualifications(backup_id: str | None = None):
        try:
            return [item.to_dict() for item in context.backups.qualifications(backup_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/system/backups/{backup_id}/qualify", status_code=201)
    def qualify_workspace_backup(backup_id: str):
        try:
            return context.backups.qualify(backup_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/system/backups/retention")
    def apply_workspace_backup_retention(payload: dict):
        try:
            if payload.get("confirmation") != "prune according to enabled policy":
                raise ValueError("retention requires explicit policy confirmation")
            return [item.to_dict() for item in context.backups.apply_retention()]
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/system/backups/{backup_id}/verify")
    def verify_workspace_backup(backup_id: str):
        try:
            return context.backups.verify(backup_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/system/backups/{backup_id}/restore", status_code=201)
    def restore_workspace_backup(backup_id: str, payload: dict):
        try:
            record = context.backups.get(backup_id)
            if payload.get("confirm_archive_hash") != record.content_hash:
                raise ValueError("restore requires the exact archive hash confirmation")
            return context.backups.restore(backup_id, payload["target_root"]).to_dict()
        except KeyError as error:
            message = str(error)
            raise HTTPException(
                status_code=404 if "unknown workspace backup" in message else 400,
                detail=message,
            ) from error
        except (OSError, RuntimeError, TypeError, ValueError, zipfile.BadZipFile) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/reasoning-providers")
    def reasoning_providers():
        return [item.to_dict() for item in context.providers.list()]

    @app.post("/api/v1/reasoning-providers", status_code=201)
    def register_reasoning_provider(payload: dict):
        try:
            return context.providers.register(ProviderConfiguration(**payload)).to_dict()
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/v1/reasoning-providers/{provider_id}/enabled")
    def set_reasoning_provider_enabled(provider_id: str, payload: dict):
        try:
            if "enabled" not in payload or not isinstance(payload["enabled"], bool):
                raise ValueError("enabled must be a boolean")
            return context.providers.set_enabled(provider_id, payload["enabled"]).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/reasoning-providers/{provider_id}/checks")
    def reasoning_provider_checks(provider_id: str):
        try:
            return [item.to_dict() for item in context.providers.checks(provider_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/reasoning-providers/{provider_id}/check")
    def check_reasoning_provider(provider_id: str):
        try:
            return context.providers.check(provider_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/reasoning-provider-evaluations")
    def reasoning_provider_evaluations(provider_id: str | None = None):
        return [item.to_dict() for item in context.providers.evaluations(provider_id)]

    @app.post("/api/v1/reasoning-providers/{provider_id}/evaluate")
    def evaluate_reasoning_provider(provider_id: str):
        try:
            return context.providers.evaluate(provider_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/projects/{project_id}/investigations")
    def investigations(project_id: str):
        try:
            return [item.to_dict() for item in context.investigations.list(project_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/projects/{project_id}/investigations", status_code=201)
    def create_investigation(project_id: str, payload: dict):
        try:
            return context.investigations.create(
                project_id, InvestigationRequest.from_dict(payload),
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/investigations/{investigation_id}")
    def investigation(investigation_id: str):
        try:
            return context.investigations.get(investigation_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/investigations/{investigation_id}/findings")
    def investigation_findings(investigation_id: str):
        try:
            return [item.to_dict() for item in context.investigations.findings(investigation_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/investigations/{investigation_id}/plans")
    def investigation_plans(investigation_id: str):
        try:
            return [item.to_dict() for item in context.investigations.plans(investigation_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/investigations/{investigation_id}/executions")
    def investigation_executions(investigation_id: str):
        try:
            return [item.to_dict() for item in context.investigations.executions(investigation_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/investigation-executions/{execution_id}")
    def investigation_execution(execution_id: str):
        try:
            return context.investigations.execution(execution_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/investigations/{investigation_id}/syntheses")
    def investigation_syntheses(investigation_id: str):
        try:
            return [item.to_dict() for item in context.investigations.syntheses(investigation_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/investigations/{investigation_id}/synthesize")
    def synthesize_investigation(investigation_id: str):
        try:
            return context.investigations.synthesize(investigation_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/investigations/{investigation_id}/narratives")
    def investigation_narratives(investigation_id: str):
        try:
            return [item.to_dict() for item in context.investigations.narratives(
                investigation_id,
            )]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/investigations/{investigation_id}/narratives", status_code=201)
    def draft_investigation_narrative(investigation_id: str, payload: dict | None = None):
        try:
            provider_id = (payload or {}).get("provider", "deterministic-v1")
            return context.investigations.draft_narrative(
                investigation_id, provider_id=provider_id,
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/investigation-narratives/{narrative_id}/review")
    def review_investigation_narrative(narrative_id: str, payload: dict):
        try:
            return context.investigations.review_narrative(
                narrative_id, status=payload["status"], reviewer=payload["reviewer"],
                note=payload["note"],
            ).to_dict()
        except KeyError as error:
            message = str(error)
            status_code = 404 if "unknown investigation narrative" in message else 400
            raise HTTPException(status_code=status_code, detail=message) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/investigations/{investigation_id}/conflicts/{conflict_id}/resolve")
    def resolve_investigation_conflict(investigation_id: str, conflict_id: str, payload: dict):
        try:
            return context.investigations.resolve_conflict(
                investigation_id, conflict_id, action=payload["action"],
                reviewer=payload["reviewer"], rationale=payload["rationale"],
                selected_finding_ids=payload.get("selected_finding_ids", ()),
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/v1/investigations/{investigation_id}/plan", status_code=201)
    def plan_investigation(investigation_id: str, payload: dict | None = None):
        try:
            payload = payload or {}
            provider = payload.get("provider", "deterministic-v1")
            provider_record = None if provider == "deterministic-v1" else context.providers.get(provider)
            adapter = None if provider_record is None else context.providers.build(provider)
            metadata = None if provider_record is None else {
                "kind": provider_record.kind, "model": provider_record.model,
                "endpoint": provider_record.endpoint,
                "configuration_hash": provider_record.content_hash,
            }
            return context.investigations.plan(
                investigation_id, provider=adapter, provider_name=provider,
                maximum_calls=payload.get("maximum_calls", 32),
                provider_metadata=metadata,
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (PermissionError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/v1/investigations/{investigation_id}/findings", status_code=201)
    def add_investigation_finding(investigation_id: str, payload: dict):
        try:
            return context.investigations.add_finding(investigation_id, payload).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/v1/investigations/{investigation_id}/analyze")
    def analyze_investigation(investigation_id: str, payload: dict | None = None):
        try:
            return [item.to_dict() for item in context.investigations.analyze(
                investigation_id, plan_id=(payload or {}).get("plan_id"),
            )]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/findings/{finding_id}/review")
    def review_finding(finding_id: str, payload: dict):
        try:
            return context.investigations.review(
                finding_id, payload["status"], payload.get("note", ""),
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/v1/findings/{finding_id}/variant-proposals", status_code=201)
    def propose_variant(finding_id: str, payload: dict):
        try:
            return context.investigations.propose_variant(
                finding_id, name=payload["name"],
                proposal_kind=payload["proposal_kind"],
                proposal=payload["proposal"],
            ).to_dict()
        except KeyError as error:
            message = str(error)
            status = 404 if "unknown finding" in message else 400
            raise HTTPException(status_code=status, detail=message) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/investigations/{investigation_id}/variant-proposals")
    def variant_proposals(investigation_id: str):
        try:
            return [item.to_dict() for item in context.investigations.proposals(investigation_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/variant-proposals/{proposal_id}/decision")
    def decide_variant_proposal(proposal_id: str, payload: dict):
        try:
            return context.investigations.decide_proposal(
                proposal_id, payload["status"], payload["note"],
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/investigations/{investigation_id}/reports", status_code=201)
    def build_engineering_report(investigation_id: str, payload: dict):
        try:
            return context.investigations.build_report(
                investigation_id, title=payload["title"],
                qualified=payload.get("qualified", False),
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/projects/{project_id}/engineering-reports")
    def engineering_reports(project_id: str):
        try:
            return [item.to_dict() for item in context.investigations.reports(project_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/projects")
    def projects():
        return [item.to_dict() for item in context.projects.list()]

    @app.post("/api/v1/projects", status_code=201)
    def create_project(payload: dict):
        try:
            request = ProjectCreate(
                payload["name"], payload.get("description", ""),
                tuple(payload.get("tags", ())),
            )
            return context.projects.create(request).to_dict()
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/projects/{project_id}")
    def project(project_id: str):
        try:
            return context.projects.get(project_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/projects/{project_id}/navigation")
    def navigation(project_id: str):
        try:
            return context.projects.navigation(project_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/projects/{project_id}/assets")
    def project_assets(project_id: str):
        try:
            return [item.to_dict() for item in context.projects.assets(project_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/projects/{project_id}/assets", status_code=201)
    def import_project_asset(project_id: str, payload: dict):
        try:
            return context.projects.import_input(
                project_id, payload["source_path"], kind=payload.get("kind", "deck"),
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/projects/{project_id}/models")
    def models(project_id: str):
        try:
            return [item.to_dict(include_occurrences=False)
                    for item in context.models.list(project_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/projects/{project_id}/models", status_code=201)
    def inspect_model(project_id: str, payload: dict):
        try:
            return context.models.inspect(
                project_id, payload["asset_id"], name=payload.get("name", ""),
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/models/{model_id}")
    def model(model_id: str):
        try:
            return context.models.get(model_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/models/{model_id}/source")
    def model_source(model_id: str, path: str, start_line: int = 1, line_count: int = 200):
        try:
            return context.models.source_lines(
                model_id, path, start_line=start_line, line_count=line_count,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/models/{model_id}/patches")
    def model_patches(model_id: str):
        try:
            return [item.to_dict() for item in context.patches.list(model_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/models/{model_id}/patches", status_code=201)
    def create_patch(model_id: str, payload: dict):
        try:
            operations = tuple(PatchOperation(
                item["kind"], item["start_line"], item["end_line"], item.get("text", ""),
            ) for item in payload["operations"])
            return context.patches.create(
                model_id, payload["title"], operations,
                description=payload.get("description", ""),
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/patches/{patch_id}/preview")
    def patch_preview(patch_id: str, context_lines: int = 3):
        try:
            return context.patches.preview(patch_id, context_lines=context_lines)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/v1/patches/{patch_id}/materialize")
    def materialize_patch(patch_id: str):
        try:
            return context.patches.materialize(patch_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/projects/{project_id}/grids")
    def grids(project_id: str):
        try:
            return [item.to_dict() for item in context.grids.list(project_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/projects/{project_id}/grids/generate")
    def generate_grid(project_id: str, payload: dict):
        if "dims" not in payload or "physical_dims_m" not in payload:
            raise HTTPException(status_code=400, detail="dims and physical_dims_m are required")
        try:
            return context.grids.generate_cartesian(
                project_id, str(payload.get("name", "Grid")),
                payload["dims"], payload["physical_dims_m"],
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/grids/{snapshot_id}")
    def grid(snapshot_id: str):
        try:
            return context.grids.get(snapshot_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/grids/{snapshot_id}/arrays/{array_name}")
    def grid_values(snapshot_id: str, array_name: str, offset: int = 0, count: int = 1000, stride: int = 1):
        try:
            return context.grids.values(snapshot_id, array_name, offset=offset, count=count, stride=stride)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/projects/{project_id}/model-drafts")
    def model_drafts(project_id: str):
        try:
            return [item.to_dict() for item in context.construction.list(project_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/projects/{project_id}/model-revisions")
    def project_model_revisions(project_id: str):
        try:
            return [item.to_dict() for item in context.construction.project_revisions(project_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/projects/{project_id}/model-drafts", status_code=201)
    def create_model_draft(project_id: str, payload: dict):
        try:
            document = ModelConstructionDocument.from_dict(payload["document"])
            return context.construction.create(
                project_id, payload["grid_snapshot_id"], payload.get("name", "Model draft"),
                document, actor=payload.get("actor", "user"),
            ).to_dict()
        except KeyError as error:
            message = str(error)
            status = 404 if "unknown" in message else 400
            raise HTTPException(status_code=status, detail=message) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/model-drafts/{draft_id}")
    def model_draft(draft_id: str):
        try:
            return context.construction.get(draft_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.put("/api/v1/model-drafts/{draft_id}")
    def update_model_draft(draft_id: str, payload: dict):
        try:
            document = ModelConstructionDocument.from_dict(payload["document"])
            return context.construction.update(
                draft_id, document, expected_version=payload["expected_version"],
                actor=payload.get("actor", "user"),
                reason=payload.get("reason", "model edited"),
            ).to_dict()
        except KeyError as error:
            message = str(error)
            status = 404 if "unknown model draft" in message else 400
            raise HTTPException(status_code=status, detail=message) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/v1/model-drafts/{draft_id}/validate")
    def validate_model_draft(draft_id: str, payload: dict | None = None):
        try:
            draft = context.construction.get(draft_id)
            document = (draft.document if not payload or "document" not in payload else
                        ModelConstructionDocument.from_dict(payload["document"]))
            issues = context.construction.validate(draft.grid_snapshot_id, document)
            return {"draft_id": draft_id, "valid": not any(
                issue["severity"] == "error" for issue in issues
            ), "issues": list(issues)}
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/model-drafts/{draft_id}/history")
    def model_draft_history(draft_id: str):
        try:
            return list(context.construction.history(draft_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/model-drafts/{draft_id}/revisions")
    def model_draft_revisions(draft_id: str):
        try:
            return [item.to_dict() for item in context.construction.revisions(draft_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/model-drafts/{draft_id}/materialize", status_code=201)
    def materialize_model_draft(draft_id: str):
        try:
            return context.construction.materialize(draft_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/model-revisions/{revision_id}")
    def model_revision(revision_id: str):
        try:
            return context.construction.revision(revision_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/projects/{project_id}/experiments")
    def experiments(project_id: str):
        try:
            return [item.to_dict() for item in context.experiments.list(project_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/projects/{project_id}/experiments", status_code=201)
    def create_experiment(project_id: str, payload: dict):
        try:
            return context.experiments.create(
                project_id, ExperimentRequest.from_dict(payload),
            ).to_dict()
        except KeyError as error:
            message = str(error)
            raise HTTPException(
                status_code=404 if "unknown" in message else 400, detail=message,
            ) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/experiments/{experiment_id}")
    def experiment(experiment_id: str):
        try:
            return context.experiments.get(experiment_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/experiments/{experiment_id}/executions")
    def experiment_executions(experiment_id: str):
        try:
            return [item.to_dict() for item in context.experiments.executions(experiment_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/experiments/{experiment_id}/executions", status_code=202)
    def create_experiment_execution(experiment_id: str, payload: dict):
        try:
            target = ExecutionTarget.from_dict(payload)
            execution = context.experiments.prepare_execution(experiment_id, target)
            if target.kind.value == "local":
                run = context.isolated.submit(execution.configuration_id)
                execution = context.experiments.bind_run(execution.execution_id, run.run_id)
            return execution.to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/experiment-executions/{execution_id}")
    def experiment_execution(execution_id: str):
        try:
            return context.experiments.execution(execution_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/experiment-executions/{execution_id}/cancel", status_code=202)
    def cancel_experiment_execution(execution_id: str):
        try:
            execution = context.experiments.execution(execution_id)
            if not execution.run_id:
                raise ValueError("staged remote executions are not controlled locally")
            context.isolated.cancel(execution.run_id)
            return context.experiments.execution(execution_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/experiments/{experiment_id}/qualifications")
    def experiment_qualifications(experiment_id: str):
        try:
            return [item.to_dict() for item in context.experiments.qualifications(experiment_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/experiment-executions/{execution_id}/qualifications", status_code=201)
    def create_experiment_qualification(execution_id: str, payload: dict):
        try:
            return context.experiments.record_qualification(
                execution_id, reference_kind=payload["reference_kind"],
                reference_id=payload["reference_id"],
                tolerances=payload["tolerances"], metrics=payload["metrics"],
                evidence=payload.get("evidence", {}),
            ).to_dict()
        except KeyError as error:
            message = str(error)
            raise HTTPException(
                status_code=404 if "unknown" in message else 400, detail=message,
            ) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/experiments/{experiment_id}/branches")
    def experiment_branches(experiment_id: str):
        try:
            return [item.to_dict() for item in context.experiments.branches(experiment_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/experiment-executions/{execution_id}/branches", status_code=202)
    def create_experiment_branch(execution_id: str, payload: dict):
        try:
            branch = context.experiments.prepare_branch(
                execution_id, BranchRequest.from_dict(payload),
            )
            run = context.isolated.submit(branch.configuration_id)
            return context.experiments.bind_branch_run(
                branch.branch_id, run.run_id,
            ).to_dict()
        except KeyError as error:
            message = str(error)
            raise HTTPException(
                status_code=404 if "unknown" in message else 400, detail=message,
            ) from error
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/experiment-branches/{branch_id}")
    def experiment_branch(branch_id: str):
        try:
            return context.experiments.branch(branch_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/tasks")
    def tasks(project_id: str | None = None):
        return [item.to_dict() for item in context.tasks.list(project_id=project_id)]

    @app.get("/api/v1/projects/{project_id}/run-configurations")
    def run_configurations(project_id: str):
        try:
            return [item.to_dict() for item in context.runs.configurations(project_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/projects/{project_id}/run-configurations", status_code=201)
    def create_run_configuration(project_id: str, payload: dict):
        try:
            return context.runs.create_configuration(
                project_id, payload["name"], payload["runner_kind"],
                payload.get("payload", {}), model_id=payload.get("model_id", ""),
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/projects/{project_id}/runs")
    def project_runs(project_id: str):
        try:
            return [item.to_dict() for item in context.runs.list(project_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/run-configurations/{configuration_id}/submit", status_code=202)
    def submit_run(configuration_id: str):
        try:
            return context.isolated.submit(configuration_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/runs/{run_id}")
    def run(run_id: str):
        try:
            return context.runs.get(run_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/runs/{run_id}/events")
    def run_events(run_id: str, after_sequence: int = -1, limit: int = 1000):
        try:
            return [item.to_dict() for item in context.runs.events(
                run_id, after_sequence=after_sequence, limit=limit,
            )]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/runs/{run_id}/checkpoints")
    def run_checkpoints(run_id: str):
        try:
            return [item.to_dict() for item in context.runs.checkpoints(run_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/runs/{run_id}/cancel", status_code=202)
    def cancel_run(run_id: str):
        try:
            return context.isolated.cancel(run_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/runs/{run_id}/resume", status_code=202)
    def resume_run(run_id: str, payload: dict):
        try:
            return context.isolated.resume(run_id, payload["checkpoint_id"]).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/runs/{run_id}/results/index")
    def index_run_results(run_id: str):
        try:
            return [item.to_dict() for item in context.results.index_run(run_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/runs/{run_id}/states")
    def result_states(run_id: str):
        try:
            return [item.to_dict() for item in context.results.states(run_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/result-states/{state_id}/fields/{field}")
    def result_field(state_id: str, field: str, offset: int = 0, count: int = 25_000, stride: int = 1):
        try:
            return context.results.field_values(state_id, field, offset=offset, count=count, stride=stride)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/runs/{run_id}/dashboard")
    def result_dashboard(run_id: str):
        try:
            return context.results.dashboard(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/runs/{run_id}/animations/{field}")
    def animation_manifest(run_id: str, field: str):
        try:
            return context.results.animation_manifest(run_id, field)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/result-states/{state_id}/exports", status_code=201)
    def export_result_field(state_id: str, payload: dict):
        try:
            return context.results.export_field(
                state_id, payload["field"], kind=payload.get("kind", "csv"),
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/v1/run-comparisons", status_code=201)
    def compare_runs(payload: dict):
        try:
            return context.results.compare(
                payload["reference_run_id"], payload["candidate_run_id"],
            ).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/run-comparisons/difference")
    def comparison_difference(
        reference_run_id: str, candidate_run_id: str, report_step: int,
        field: str, offset: int = 0, count: int = 25_000,
    ):
        try:
            return context.results.difference(
                reference_run_id, candidate_run_id, report_step, field,
                offset=offset, count=count,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/workspace-state")
    def workspace_state():
        return context.state.workspace().to_dict()

    @app.put("/api/v1/workspace-state")
    def save_workspace_state(payload: dict):
        try:
            return context.state.save_workspace(WorkspaceState.from_dict(payload)).to_dict()
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/tasks/{task_id}")
    def task(task_id: str):
        try:
            return context.tasks.get(task_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/v1/tasks/{task_id}/cancel", status_code=202)
    def cancel_task(task_id: str):
        try:
            return context.tasks.request_cancellation(task_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/notifications")
    def notifications(project_id: str | None = None, unread_only: bool = False):
        return [item.to_dict() for item in context.notifications.list(
            project_id=project_id, unread_only=unread_only,
        )]

    # Added last so security headers also cover authentication, size-limit,
    # transport, trusted-host, CORS, and endpoint error responses.
    app.add_middleware(
        SecurityHeadersMiddleware, hsts_seconds=deployment.hsts_seconds,
        documentation_enabled=deployment.expose_api_documentation,
    )
    return app


__all__ = ["create_fastapi_app"]
