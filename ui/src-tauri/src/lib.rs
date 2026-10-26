use rand::RngCore;
use serde::Serialize;
use std::{
    path::Path,
    sync::{mpsc, Mutex},
    time::Duration,
};
use tauri::{AppHandle, Manager, State};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};
use tauri_plugin_updater::{Update, UpdaterExt};

const PRODUCT_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Serialize)]
struct DesktopBackend {
    endpoint: String,
    token: String,
    managed: bool,
    version: String,
}

struct BackendState {
    connection: DesktopBackend,
    child: Mutex<Option<CommandChild>>,
    terminated: Mutex<mpsc::Receiver<()>>,
}

struct PendingUpdate(Mutex<Option<Update>>);

struct UpdateConfiguration {
    endpoint: String,
    enabled: bool,
}

#[derive(Clone, Serialize)]
struct UpdateStatus {
    enabled: bool,
    current_version: String,
    trust_model: &'static str,
}

#[derive(Clone, Serialize)]
struct UpdateMetadata {
    version: String,
    current_version: String,
    notes: Option<String>,
    publication_date: Option<String>,
}

#[derive(serde::Deserialize)]
struct ReadinessEvent {
    event: String,
    endpoint: String,
    database_schema: u32,
}

#[tauri::command]
fn desktop_backend(state: tauri::State<'_, BackendState>) -> DesktopBackend {
    state.connection.clone()
}

#[tauri::command]
fn desktop_update_status(configuration: State<'_, UpdateConfiguration>) -> UpdateStatus {
    UpdateStatus {
        enabled: configuration.enabled,
        current_version: PRODUCT_VERSION.into(),
        trust_model: "pinned-ed25519-signature",
    }
}

#[tauri::command]
async fn check_for_update(
    app: AppHandle,
    configuration: State<'_, UpdateConfiguration>,
    pending: State<'_, PendingUpdate>,
) -> Result<Option<UpdateMetadata>, String> {
    if !configuration.enabled {
        return Err("updates are not configured in this distribution".into());
    }
    let endpoint = url::Url::parse(&configuration.endpoint)
        .map_err(|error| format!("invalid pinned update endpoint: {error}"))?;
    if endpoint.scheme() != "https" {
        return Err("update endpoint must use HTTPS".into());
    }
    let update = app
        .updater_builder()
        .endpoints(vec![endpoint])
        .map_err(|error| error.to_string())?
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|error| error.to_string())?
        .check()
        .await
        .map_err(|error| error.to_string())?;
    let metadata = update.as_ref().map(|candidate| UpdateMetadata {
        version: candidate.version.clone(),
        current_version: candidate.current_version.clone(),
        notes: candidate.body.clone(),
        publication_date: candidate.date.map(|value| value.to_string()),
    });
    *pending
        .0
        .lock()
        .map_err(|_| "pending update lock is poisoned".to_string())? = update;
    Ok(metadata)
}

#[tauri::command]
async fn install_pending_update(
    pending: State<'_, PendingUpdate>,
    user_confirmed: bool,
) -> Result<(), String> {
    if !user_confirmed {
        return Err("update installation requires explicit user confirmation".into());
    }
    let update = pending
        .0
        .lock()
        .map_err(|_| "pending update lock is poisoned".to_string())?
        .take()
        .ok_or_else(|| "there is no verified pending update".to_string())?;
    update
        .download_and_install(|_, _| {}, || {})
        .await
        .map_err(|error| error.to_string())
}

fn bootstrap_token() -> String {
    let mut bytes = [0_u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|value| format!("{value:02x}")).collect()
}

fn start_backend(
    app: &tauri::AppHandle,
    workspace: &Path,
) -> Result<(DesktopBackend, CommandChild, mpsc::Receiver<()>), String> {
    std::fs::create_dir_all(workspace)
        .map_err(|error| format!("cannot create Studio workspace: {error}"))?;
    let token = bootstrap_token();
    let workspace_argument = workspace.to_string_lossy().to_string();
    let command = app
        .shell()
        .sidecar("pymrst-sidecar")
        .map_err(|error| format!("cannot resolve managed PyMRST backend: {error}"))?
        .args(["--workspace", workspace_argument.as_str(), "--port", "0"])
        .env("PYMRST_DESKTOP_BOOTSTRAP_TOKEN", &token);
    let (mut events, child) = command
        .spawn()
        .map_err(|error| format!("cannot start managed PyMRST backend: {error}"))?;
    let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
    let (exit_sender, exit_receiver) = mpsc::sync_channel(1);
    tauri::async_runtime::spawn(async move {
        let mut readiness_sent = false;
        while let Some(event) = events.recv().await {
            match event {
                CommandEvent::Stdout(bytes) if !readiness_sent => {
                    readiness_sent = true;
                    let parsed = serde_json::from_slice::<ReadinessEvent>(&bytes)
                        .map_err(|error| format!("invalid backend readiness record: {error}"));
                    let _ = ready_sender.send(parsed);
                }
                CommandEvent::Error(bytes) if !readiness_sent => {
                    readiness_sent = true;
                    let _ = ready_sender.send(Err(format!("managed backend error: {bytes}")));
                }
                CommandEvent::Terminated(_) => {
                    if !readiness_sent {
                        let _ =
                            ready_sender.send(Err("backend exited before readiness".to_string()));
                    }
                    let _ = exit_sender.send(());
                    break;
                }
                _ => {}
            }
        }
    });
    // The managed backend is a PyInstaller onefile sidecar: every launch pays a
    // cold unpack-and-import cost for its full NumPy/SciPy/Numba stack before it
    // can report readiness. Measured on-machine this ranges from ~20s to over
    // 40s under ordinary background load, so a 30s budget intermittently aborts
    // an otherwise-healthy startup.
    let readiness = match ready_receiver.recv_timeout(Duration::from_secs(90)) {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => {
            let _ = child.kill();
            return Err(error);
        }
        Err(_) => {
            let _ = child.kill();
            return Err("managed backend did not become ready within 90 seconds".into());
        }
    };
    if readiness.event != "ready"
        || !readiness.endpoint.starts_with("http://127.0.0.1:")
        || readiness.database_schema == 0
    {
        let _ = child.kill();
        return Err("managed backend returned an unsafe readiness record".into());
    }
    Ok((
        DesktopBackend {
            endpoint: readiness.endpoint,
            token,
            managed: true,
            version: PRODUCT_VERSION.into(),
        },
        child,
        exit_receiver,
    ))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let application = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let endpoint = option_env!("PYMRST_UPDATER_ENDPOINT")
                .unwrap_or("")
                .trim()
                .to_string();
            let public_key = option_env!("PYMRST_UPDATER_PUBLIC_KEY")
                .unwrap_or("")
                .trim()
                .to_string();
            if endpoint.is_empty() != public_key.is_empty() {
                return Err(std::io::Error::other(
                    "updater endpoint and public key must be configured together",
                )
                .into());
            }
            if !endpoint.is_empty() {
                let parsed = url::Url::parse(&endpoint).map_err(|error| {
                    std::io::Error::other(format!("invalid updater endpoint: {error}"))
                })?;
                if parsed.scheme() != "https" {
                    return Err(std::io::Error::other(
                        "production updater endpoint must use HTTPS",
                    )
                    .into());
                }
                app.handle()
                    .plugin(
                        tauri_plugin_updater::Builder::new()
                            .pubkey(public_key)
                            .build(),
                    )
                    .map_err(std::io::Error::other)?;
            }
            app.manage(UpdateConfiguration {
                enabled: !endpoint.is_empty(),
                endpoint,
            });
            app.manage(PendingUpdate(Mutex::new(None)));
            let root = app.path().app_data_dir().map_err(|error| {
                std::io::Error::other(format!(
                    "cannot resolve application-data directory: {error}",
                ))
            })?;
            let (connection, child, terminated) =
                start_backend(app.handle(), &root.join("workspace"))
                    .map_err(std::io::Error::other)?;
            app.manage(BackendState {
                connection,
                child: Mutex::new(Some(child)),
                terminated: Mutex::new(terminated),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            desktop_backend,
            desktop_update_status,
            check_for_update,
            install_pending_update
        ])
        .build(tauri::generate_context!())
        .expect("error while building PyMRST Studio");
    application.run(|handle, event| {
        if let tauri::RunEvent::Exit = event {
            if let Some(state) = handle.try_state::<BackendState>() {
                if let Ok(mut guard) = state.child.lock() {
                    if let Some(mut child) = guard.take() {
                        let _ = child.write(b"shutdown\n");
                        let stopped = state
                            .terminated
                            .lock()
                            .ok()
                            .and_then(|receiver| receiver.recv_timeout(Duration::from_secs(5)).ok())
                            .is_some();
                        if !stopped {
                            let _ = child.kill();
                        }
                    }
                }
            }
        }
    });
}
