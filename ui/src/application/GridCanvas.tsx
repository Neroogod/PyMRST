import { useEffect, useRef, useState } from "react";
import { api, type GridSnapshot } from "../api/client";

export function GridCanvas({ grid, compact }: { grid: GridSnapshot; compact?: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const properties = grid.arrays.filter((item) => !["CELL_CENTROIDS", "ACTIVE_INDEX"].includes(item.name));
  const [field, setField] = useState(properties[0]?.name ?? "ACTIVE_INDEX");
  const [points, setPoints] = useState<number[][]>([]);
  const [values, setValues] = useState<number[]>([]);
  const [error, setError] = useState("");
  const limit = Math.min(grid.cell_count, 25000);
  const stride = Math.max(1, Math.ceil(grid.cell_count/limit));
  useEffect(() => {
    const controller = new AbortController(); setError("");
    Promise.all([
      api.gridValues(grid.snapshot_id, "CELL_CENTROIDS", 0, limit, controller.signal, stride),
      api.gridValues(grid.snapshot_id, field, 0, limit, controller.signal, stride),
    ]).then(([coordinates, property]) => {
      setPoints(coordinates.values as number[][]);
      const raw = property.values as number[] | number[][];
      setValues(raw.map((item) => Array.isArray(item) ? Number(item[0]) : Number(item)));
    }).catch((caught) => { if (caught.name !== "AbortError") setError(caught.message); });
    return () => controller.abort();
  }, [grid.snapshot_id, field, limit, stride]);
  useEffect(() => {
    const target = canvas.current; if (!target || !points.length) return;
    const draw = () => {
      const ratio = window.devicePixelRatio || 1;
      const bounds = target.getBoundingClientRect();
      target.width = Math.max(1, Math.floor(bounds.width * ratio)); target.height = Math.max(1, Math.floor(bounds.height * ratio));
      const context = target.getContext("2d"); if (!context) return;
      context.scale(ratio, ratio); context.fillStyle = "#07131c"; context.fillRect(0, 0, bounds.width, bounds.height);
      const xs = points.map((point) => point[0]); const ys = points.map((point) => point[1] ?? 0);
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const finite = values.filter(Number.isFinite); const minV = Math.min(...finite), maxV = Math.max(...finite);
      const pad = 25, width = Math.max(maxX - minX, 1e-20), height = Math.max(maxY - minY, 1e-20);
      points.forEach((point, index) => {
        const value = values[index]; const fraction = Number.isFinite(value) && maxV > minV ? (value - minV) / (maxV - minV) : .5;
        const x = pad + ((point[0] - minX) / width) * (bounds.width - 2 * pad);
        const y = bounds.height - pad - (((point[1] ?? 0) - minY) / height) * (bounds.height - 2 * pad);
        context.fillStyle = Number.isFinite(value) ? `hsl(${190 - 150 * fraction} 72% ${42 + 12 * fraction}%)` : "#6f7880";
        context.fillRect(x - 1.6, y - 1.6, 3.2, 3.2);
      });
    };
    draw(); const observer = new ResizeObserver(draw); observer.observe(target); return () => observer.disconnect();
  }, [points, values]);
  return <div className={compact ? "grid-viewer compact" : "grid-viewer"}><div className="grid-view-toolbar"><label htmlFor="grid-color-by">Color by<select id="grid-color-by" name="grid-color-by" value={field} onChange={(event) => setField(event.target.value)}>{grid.arrays.filter((item) => item.shape.length === 1).map((item) => <option key={item.name}>{item.name}</option>)}</select></label><span>{limit.toLocaleString()} / {grid.cell_count.toLocaleString()} cells {limit < grid.cell_count ? "· display LOD" : ""}</span></div>{error ? <div className="form-error">{error}</div> : <canvas ref={canvas} aria-label={`Plan-view cell centroids colored by ${field}`} />}<div className="grid-axis">X–Y centroid projection · {field}</div></div>;
}
