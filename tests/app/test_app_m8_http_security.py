from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from pymrst.app.api.application import create_fastapi_app
from pymrst.app.application import create_application
from pymrst.app.schemas.deployment import ServerDeploymentSettings


@pytest.fixture
def http_application(tmp_path):
    context = create_application(tmp_path / "workspace")
    application = create_fastapi_app(context, ServerDeploymentSettings())
    with TestClient(application) as client:
        yield context, client
    context.close()


def _bootstrap_shared(client: TestClient):
    project = client.post("/api/v1/projects", json={
        "name": "Secured reservoir", "description": "HTTP qualification", "tags": [],
    })
    assert project.status_code == 201
    user = client.post("/api/v1/security/users", json={
        "display_name": "Platform Administrator", "email": "admin@example.test",
        "system_role": "admin",
    })
    assert user.status_code == 201
    user_id = user.json()["user_id"]
    issued = client.post(f"/api/v1/security/users/{user_id}/tokens", json={
        "label": "HTTP qualification", "scopes": [
            "system_admin", "view", "edit", "execute", "publish", "audit",
        ], "expires_at": "2099-01-01T00:00:00+00:00",
    })
    assert issued.status_code == 201
    token = issued.json()["token"]
    switched = client.put("/api/v1/security/mode", json={"mode": "shared"})
    assert switched.status_code == 200
    return project.json(), user_id, token


def test_http_boundary_enforces_shared_authentication_and_project_roles(http_application):
    _, client = http_application
    project, _, admin_token = _bootstrap_shared(client)
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    assert client.get("/api/v1/health").status_code == 200
    unauthenticated = client.get(f"/api/v1/projects/{project['project_id']}")
    assert unauthenticated.status_code == 401
    assert unauthenticated.headers["www-authenticate"] == "Bearer"
    assert unauthenticated.headers["x-content-type-options"] == "nosniff"
    assert client.get("/api/v1/projects", headers=admin_headers).status_code == 200

    engineer = client.post("/api/v1/security/users", headers=admin_headers, json={
        "display_name": "Reservoir Engineer", "email": "engineer@example.test",
        "system_role": "engineer",
    })
    assert engineer.status_code == 201
    engineer_id = engineer.json()["user_id"]
    membership = client.put(
        f"/api/v1/security/projects/{project['project_id']}/members/{engineer_id}",
        headers=admin_headers, json={"project_role": "editor"},
    )
    assert membership.status_code == 200
    issued = client.post(
        f"/api/v1/security/users/{engineer_id}/tokens", headers=admin_headers,
        json={"label": "engineering", "scopes": ["view", "edit", "execute"],
              "expires_at": "2099-01-01T00:00:00+00:00"},
    )
    engineer_headers = {"Authorization": f"Bearer {issued.json()['token']}"}
    assert client.get(
        f"/api/v1/projects/{project['project_id']}", headers=engineer_headers,
    ).status_code == 200
    # A global collection could reveal other projects, so it fails closed.
    assert client.get("/api/v1/projects", headers=engineer_headers).status_code == 403
    assert client.get("/api/v1/system/backups", headers=engineer_headers).status_code == 403
    verification = client.get("/api/v1/security/audit/verify", headers=admin_headers)
    assert verification.status_code == 200
    assert verification.json()["valid"] is True


def test_http_tokens_are_not_retrievable_and_revocation_is_immediate(http_application):
    _, client = http_application
    _, _, admin_token = _bootstrap_shared(client)
    headers = {"Authorization": f"Bearer {admin_token}"}
    identity = client.get("/api/v1/security/me", headers=headers)
    token_id = identity.json()["token_id"]
    listed = client.get("/api/v1/security/tokens", headers=headers)
    assert listed.status_code == 200
    assert all("token" not in record for record in listed.json())
    # Last-admin protection rejects revocation until a replacement exists.
    refusal = client.post(f"/api/v1/security/tokens/{token_id}/revoke", headers=headers)
    assert refusal.status_code == 409


def test_runtime_shared_transition_revalidates_deployment_policy(tmp_path):
    context = create_application(tmp_path / "workspace")
    unsafe = ServerDeploymentSettings(allowed_hosts=("*",))
    with TestClient(create_fastapi_app(context, unsafe)) as client:
        user = client.post("/api/v1/security/users", json={
            "display_name": "Admin", "email": "admin@example.test", "system_role": "admin",
        }).json()
        client.post(f"/api/v1/security/users/{user['user_id']}/tokens", json={
            "label": "bootstrap", "scopes": ["system_admin"],
            "expires_at": "2099-01-01T00:00:00+00:00",
        })
        response = client.put("/api/v1/security/mode", json={"mode": "shared"})
        assert response.status_code == 409
        assert "wildcard allowed hosts" in response.json()["detail"]
    context.close()


def test_real_http_stack_rejects_oversized_payload_before_endpoint(tmp_path):
    context = create_application(tmp_path / "workspace")
    settings = ServerDeploymentSettings(maximum_request_bytes=1024)
    with TestClient(create_fastapi_app(context, settings)) as client:
        response = client.post(
            "/api/v1/projects", content=b"x" * 1025,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413
        assert response.headers["x-frame-options"] == "DENY"
    context.close()


def test_real_http_stack_enforces_host_and_origin_allowlists(tmp_path):
    context = create_application(tmp_path / "workspace")
    settings = ServerDeploymentSettings(
        allowed_hosts=("studio.example.test", "testserver"),
        allowed_origins=("https://studio.example.test",),
    )
    with TestClient(create_fastapi_app(context, settings)) as client:
        hostile_host = client.get("/api/v1/health", headers={"Host": "attacker.example"})
        assert hostile_host.status_code == 400
        accepted = client.options("/api/v1/health", headers={
            "Origin": "https://studio.example.test",
            "Access-Control-Request-Method": "GET",
        })
        assert accepted.status_code == 200
        assert accepted.headers["access-control-allow-origin"] == "https://studio.example.test"
        rejected = client.options("/api/v1/health", headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "GET",
        })
        assert rejected.status_code == 400
        assert "access-control-allow-origin" not in rejected.headers
    context.close()


def test_http_authentication_failure_limit_precedes_token_lookup(tmp_path):
    context = create_application(tmp_path / "workspace")
    settings = ServerDeploymentSettings(authentication_failures_per_minute=2)
    with TestClient(create_fastapi_app(context, settings)) as client:
        _, _, _ = _bootstrap_shared(client)
        headers = {"Authorization": "Bearer pymrst_pat_missing.invalid"}
        assert client.get("/api/v1/security/me", headers=headers).status_code == 401
        assert client.get("/api/v1/security/me", headers=headers).status_code == 401
        limited = client.get("/api/v1/security/me", headers=headers)
        assert limited.status_code == 429
        assert limited.headers["retry-after"] == "60"
    context.close()


def test_desktop_bootstrap_credential_protects_even_local_health(tmp_path):
    context = create_application(tmp_path / "workspace")
    token = "d" * 64
    with TestClient(create_fastapi_app(
        context, ServerDeploymentSettings(), desktop_bootstrap_token=token,
    )) as client:
        assert client.get("/api/v1/health").status_code == 401
        wrong = client.get("/api/v1/health", headers={"X-PyMRST-Desktop-Token": "x" * 64})
        assert wrong.status_code == 401
        accepted = client.get(
            "/api/v1/health", headers={"X-PyMRST-Desktop-Token": token},
        )
        assert accepted.status_code == 200
        assert accepted.json()["status"] == "ok"
    context.close()


def test_desktop_preflight_succeeds_without_credential_but_real_call_still_requires_it(
    tmp_path,
):
    # Regression test: the desktop shell's own CORS preflight (an OPTIONS
    # probe browsers send before any request carrying a custom header) was
    # being rejected by the desktop-token check before CORSMiddleware ever
    # saw it, and X-PyMRST-Desktop-Token was missing from the CORS
    # allow-list even when it was reached. Both made the packaged desktop
    # app permanently unable to reach its own bundled backend, while direct
    # (non-browser) HTTP calls worked fine and hid the bug.
    context = create_application(tmp_path / "workspace")
    token = "e" * 64
    settings = ServerDeploymentSettings(
        allowed_hosts=("127.0.0.1", "localhost", "testserver"),
        allowed_origins=("tauri://localhost", "http://tauri.localhost"),
    )
    with TestClient(create_fastapi_app(
        context, settings, desktop_bootstrap_token=token,
    )) as client:
        preflight = client.options("/api/v1/projects", headers={
            "Origin": "tauri://localhost",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-pymrst-desktop-token,content-type",
        })
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "tauri://localhost"
        assert "x-pymrst-desktop-token" in preflight.headers["access-control-allow-headers"].lower()

        unauthenticated = client.get(
            "/api/v1/projects", headers={"Origin": "tauri://localhost"},
        )
        assert unauthenticated.status_code == 401

        authenticated = client.get("/api/v1/projects", headers={
            "Origin": "tauri://localhost", "X-PyMRST-Desktop-Token": token,
        })
        assert authenticated.status_code == 200
        assert authenticated.json() == []
    context.close()
