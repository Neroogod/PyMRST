"""Protocol qualification jobs; these exercise orchestration without simulation."""

from __future__ import annotations

import json
from pathlib import Path
import time


def qualification_probe(context, request):
    steps = int(request.get("steps", 3))
    delay = float(request.get("delay_seconds", 0.0))
    if not 1 <= steps <= 1000 or not 0 <= delay <= 2:
        raise ValueError("probe bounds exceeded")
    resume = request.get("_resume") or {}
    start = int(resume.get("metadata", {}).get("completed_steps", 0))
    context.emit("report", report_step=start, status="starting")
    for index in range(start, steps):
        context.raise_if_cancelled()
        if delay:
            time.sleep(delay)
        context.progress((index + 1) / steps, f"Report step {index + 1}/{steps}",
                         report_step=index + 1, report_steps=steps,
                         newton_iterations=2 + index)
        context.emit("convergence", report_step=index + 1, residual=10.0 ** (-(index + 2)))
        if request.get("checkpoint_at") == index + 1:
            relative = f"checkpoints/report_{index + 1:04d}.json"
            target = context.run_directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"completed_steps": index + 1}) + "\n")
            context.checkpoint(relative, completed_steps=index + 1, report_steps=steps)
    if request.get("fail"):
        raise RuntimeError("requested protocol qualification failure")
    return {"report_steps": steps, "resumed_from_step": start}


__all__ = ["qualification_probe"]
