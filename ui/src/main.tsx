import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./application/App";
import "./styles/tokens.css";
import "./styles/shell.css";
import { bootstrapApplication } from "./application/bootstrap";

const root = createRoot(document.getElementById("root")!);
bootstrapApplication().then(() => root.render(
  <StrictMode><App /></StrictMode>,
)).catch((error) => root.render(
  <main className="startup-failure"><span>PyMRST Studio</span><h1>Backend startup failed</h1><p>{error instanceof Error ? error.message : "The managed backend did not become ready."}</p><small>No simulation was started. Restart the application or run the installation qualification.</small></main>,
));
