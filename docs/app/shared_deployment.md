# PyMRST Studio shared deployment

PyMRST Studio starts in local desktop mode. Local mode is intentionally suitable
only for a loopback listener used by one engineer. A network listener is refused
until the workspace has been switched to shared mode and a viable administrator
credential exists.

## Bootstrap without creating a lockout

1. Launch on loopback in local mode.
2. Open **Settings → Identity and access**.
3. Create an administrator identity.
4. Issue an expiring token containing `system_admin`, `view`, `edit`, `execute`,
   `publish`, and `audit` scopes.
5. Copy the token and select **Use in this session**. Its plaintext is shown once;
   only its SHA-256 digest is stored.
6. Enable shared mode. PyMRST refuses the transition if no viable administrator
   token exists or the active network policy is unsafe.
7. Store the token in the deployment's secret manager. Do not place it in a deck,
   project, report, command history, environment committed to source control, or
   reverse-proxy configuration.

Create a second administrator credential before revoking or rotating the first.
Last-administrator safeguards intentionally reject a disabling or revocation action
that would make the workspace inaccessible.

## Production environment

The API reads these variables at process start:

```shell
export PYMRST_APP_ENVIRONMENT=production
export PYMRST_APP_ALLOWED_HOSTS=studio.example.org
export PYMRST_APP_ALLOWED_ORIGINS=https://studio.example.org
export PYMRST_APP_REQUIRE_HTTPS=true
export PYMRST_APP_EXPOSE_DOCS=false
export PYMRST_APP_MAXIMUM_REQUEST_BYTES=16777216
export PYMRST_APP_AUTH_FAILURES_PER_MINUTE=30
export PYMRST_APP_HSTS_SECONDS=31536000
```

For TLS terminated by a reverse proxy, list only the proxy networks that can make a
direct connection to the application:

```shell
export PYMRST_APP_TRUSTED_PROXIES=127.0.0.1/32,::1/128
```

`X-Forwarded-Proto` and `X-Forwarded-For` are ignored from every other peer. Never
set the trusted proxy list to all networks. Configure the firewall so the Uvicorn
port is reachable only by the selected proxy.

Start one application process for one workspace:

```shell
pymrst-studio /srv/pymrst/workspace --host 127.0.0.1 --port 8765
```

PyMRST uses subprocess isolation for simulations, while the workspace application
itself deliberately uses one HTTP process. Starting multiple application workers
against one workspace would duplicate schedulers and recovery ownership and is not
currently supported.

## Caddy TLS termination example

```caddyfile
studio.example.org {
    encode zstd gzip
    reverse_proxy 127.0.0.1:8765 {
        header_up X-Forwarded-Proto {scheme}
        header_up X-Forwarded-For {remote_host}
    }
    header {
        -Server
        X-Content-Type-Options nosniff
        X-Frame-Options DENY
        Referrer-Policy no-referrer
    }
}
```

The public UI origin must match `PYMRST_APP_ALLOWED_ORIGINS` exactly. Wildcard host
and origin rules are rejected in shared mode. Bearer authentication does not use
cookies, and CORS credentials are disabled.

## Direct TLS

Direct TLS is supported when no reverse proxy is used:

```shell
pymrst-studio /srv/pymrst/workspace \
  --host 0.0.0.0 --port 8765 \
  --ssl-certfile /etc/pymrst/tls/fullchain.pem \
  --ssl-keyfile /etc/pymrst/tls/private-key.pem
```

Restrict the private key to the operating-system account running PyMRST. Certificate
renewal and service restart are the responsibility of the deployment supervisor.

## Authorization model

System administrators manage deployment-wide configuration and backup operations.
Auditors can inspect governed evidence. Engineers and viewers receive no project
access merely from their global title; they need a project membership:

| Project role | View | Edit | Execute | Publish | Audit |
|---|---:|---:|---:|---:|---:|
| Owner | Yes | Yes | Yes | Yes | Yes |
| Editor | Yes | Yes | Yes | Yes | No |
| Runner | Yes | No | Yes | No | No |
| Analyst | Yes | No | No | Yes | No |
| Viewer | Yes | No | No | No | No |

The effective permission is the intersection of the user's roles and the token's
scopes. A broadly scoped token cannot elevate a narrowly assigned user.

## Operational qualification

Before making a deployment reachable:

1. Run `python3 -m pytest tests/app` with the `pymrst[app]` dependencies installed.
2. Verify **Settings → Identity and access → Security audit** reports a valid chain.
3. Create and verify a workspace backup, then run a disposable recovery drill.
4. Confirm an unauthenticated project request returns `401` and an unrelated
   project member receives `403`.
5. Confirm an unapproved `Host` and browser origin are rejected.
6. Confirm HTTP is rejected when HTTPS enforcement is enabled.
7. Confirm the application port is inaccessible outside the proxy/firewall path.
8. Retain audit and recovery evidence under the site's operational policy.

The hash chain detects modification; it is not a substitute for exporting audit
records to append-only external storage. That export and multi-node rate limiting
remain required before a horizontally scaled deployment is supported.
