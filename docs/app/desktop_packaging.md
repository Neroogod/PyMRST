# PyMRST Studio desktop packaging

The desktop application is a Tauri shell plus one managed Python sidecar. It is not
an Electron application and it does not require an engineer to start a separate API
server.

## Trust and lifecycle model

At startup, the Tauri process:

1. Resolves the platform-specific `pymrst-sidecar` through Tauri's supported
   shell-sidecar resolver. The webview is not granted shell spawn/execute access.
2. Creates or opens the operating-system application-data workspace.
3. Generates a fresh 256-bit random bootstrap credential in memory.
4. Starts the sidecar with an OS-selected loopback port. The credential is supplied
   only through the child environment.
5. Waits at most 30 seconds for one JSON readiness record on stdout.
6. Rejects non-loopback endpoints, invalid records, absent schema information, and
   early process exits.
7. Exposes endpoint, credential, managed status, and application version only to the
   main Tauri window through the `desktop_backend` command.
8. Sends `shutdown` through the private child stdin pipe during desktop exit, waits
   up to five seconds, and uses forced termination only as a fallback.

Every desktop API request must contain `X-PyMRST-Desktop-Token`. Constant-time
comparison occurs before local/shared authorization. The token is not written to
SQLite, browser session storage, project files, reports, logs, backups, or release
metadata.

## Native build matrix

Sidecars must be built natively. Cross-building Python scientific packages is not
accepted because it can silently bundle incompatible BLAS, OpenMP, or binary wheels.

| Operating system | Architectures | Tauri sidecar target |
|---|---|---|
| macOS | Apple Silicon, Intel | `aarch64-apple-darwin`, `x86_64-apple-darwin` |
| Windows | x86-64, ARM64 | `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc` |
| Linux | x86-64, ARM64 | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` |

## Build sequence

Use a clean, locked Python environment containing PyMRST's required numerical and
application dependencies plus PyInstaller. Then run:

```shell
python3 scripts/check_studio_release_versions.py
python3 scripts/build_studio_sidecar.py
npm --prefix ui ci
npm --prefix ui run build
npm --prefix ui run tauri -- build
```

The sidecar builder writes the native executable under
`ui/src-tauri/binaries/pymrst-sidecar-<target-triple>` and records its SHA-256,
size, Python version, target, and build time in a sibling receipt. Generated
binaries are ignored by Git.

Tauri's `externalBin` declaration selects that target-triple binary and bundles it
as `pymrst-sidecar`. A development build may set `PYMRST_STUDIO_SIDECAR` to an
explicit executable; an invalid or missing override fails startup instead of
falling back to an arbitrary system Python.

## Release evidence

After platform signing/notarization, build the release manifest over the final
installer and Python artifacts—not intermediate unsigned files:

```shell
python3 scripts/build_studio_release_manifest.py release/macos-arm64 \
  --artifact PyMRST-Studio_0.1.0_aarch64.dmg \
  --version 0.1.0 --platform macos --architecture arm64 \
  --output release/macos-arm64/release-manifest.json \
  --signing-key /secure/offline/pymrst-release-ed25519.pem
```

The signing key must live outside the repository and CI logs. The detached
signature and public verification key may be distributed with the manifest.

## Signed updater channel

The updater is disabled unless both `PYMRST_UPDATER_ENDPOINT` and
`PYMRST_UPDATER_PUBLIC_KEY` are present while compiling the Tauri host. The
endpoint is immutable for that build and must use HTTPS. The public value is the
Tauri updater public-key content, not a filename. Keep the matching private key
outside the repository and inject it only into the native release runner through
`TAURI_SIGNING_PRIVATE_KEY` (and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` when used).

Release builds create Tauri v2 updater artifacts. Publish the signed bundle,
its `.sig` content, version, and URL using Tauri's static or dynamic metadata
schema. The desktop UI performs an explicit check, retains the verified candidate
inside the native process, and requires a second human confirmation before
download and installation. It never permits HTTP endpoints, downgrades,
frontend-supplied endpoints or keys, or silent installation.

Use the release overlay only after the signing environment is present:

```shell
npm --prefix ui run tauri -- build \
  --config src-tauri/tauri.release.conf.json
```

Ordinary development/debug builds deliberately omit updater artifacts so a
developer does not need access to the release private key.

An update release is acceptable only when a native clean-install runner proves:

1. no-update response handling;
2. valid signed update discovery and installation;
3. invalid signature, malformed metadata, wrong platform, and downgrade refusal;
4. application and sidecar version agreement after restart;
5. workspace migration, backup, and recovery drill after the update; and
6. platform code-signing/notarization independently of updater signing.

## Acceptance requirements

Each native runner must retain evidence for:

- version synchronization;
- clean sidecar construction;
- installer construction and platform signature/notarization verification;
- first launch with a new application-data directory;
- readiness, authenticated health, and denied unauthenticated health;
- environment qualification and SQLite migration;
- a small isolated simulation and clean application exit;
- reopen with the same workspace;
- backup, recovery drill, uninstall, and reinstall without project loss;
- release-manifest checksum and Ed25519 verification.

The current implementation supplies the lifecycle, security, build planner,
receipts, release verification, and opt-in signed updater boundary. An unsigned
Apple-silicon development app and checksum-valid DMG have passed local build
acceptance. Actual production-signed installer and updater evidence must still be
produced on native macOS, Windows, and Linux build runners.
