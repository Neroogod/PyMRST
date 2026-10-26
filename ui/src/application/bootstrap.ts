import { configureDesktopBackend } from "../api/client";

interface DesktopBackend {
  endpoint: string;
  token: string;
  managed: boolean;
  version: string;
}

export function isDesktopApplication(): boolean {
  return "__TAURI_INTERNALS__" in window;
}

export async function invokeDesktop<T>(command: string, arguments_: Record<string, unknown> = {}): Promise<T> {
  if (!isDesktopApplication()) throw new Error("This operation is available only in the installed desktop application");
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, arguments_);
}

export async function bootstrapApplication(): Promise<void> {
  if (!isDesktopApplication()) return;
  const backend = await invokeDesktop<DesktopBackend>("desktop_backend");
  if (!backend.managed) throw new Error("Native shell did not start a managed backend");
  configureDesktopBackend(backend.endpoint, backend.token);
}
