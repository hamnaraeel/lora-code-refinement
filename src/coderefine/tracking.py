"""Experiment tracking with a graceful degradation ladder.

W&B is the primary backend, MLflow the alternative, and a local JSONL writer
the floor. The floor matters: a training run that dies because ``wandb login``
was never run is a bad trade, and more importantly the *record* of a run should
survive in the run directory itself, not only on someone else's server. So the
local backend is always active, and W&B/MLflow are layered on top when
available.

Everything the brief asks to be logged goes through here: hyperparameters,
loss curves, validation metrics at each checkpoint, GPU memory and wall-clock.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Protocol


class Tracker(Protocol):
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...
    def log_summary(self, summary: dict[str, Any]) -> None: ...
    def log_artifact(self, path: str | Path, name: str = "", kind: str = "model") -> None: ...
    def finish(self) -> None: ...


class LocalTracker:
    """Append-only JSONL log inside the run directory. Always on."""

    def __init__(self, run_dir: Path, config: dict):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self._start = time.time()
        (self.run_dir / "config_resolved.json").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8"
        )
        self._summary: dict[str, Any] = {}

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        row = {"_step": step, "_elapsed_s": round(time.time() - self._start, 2), **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def log_summary(self, summary: dict[str, Any]) -> None:
        self._summary.update(summary)
        self.summary_path.write_text(
            json.dumps(self._summary, indent=2, default=str), encoding="utf-8"
        )

    def log_artifact(self, path: str | Path, name: str = "", kind: str = "model") -> None:
        # Local runs keep artifacts in place; record the pointer only.
        self.log_summary({f"artifact_{name or Path(path).name}": str(path)})

    def finish(self) -> None:
        self.log_summary({"wall_clock_s": round(time.time() - self._start, 2)})


class WandbTracker:
    def __init__(self, project: str, name: str, config: dict, run_dir: Path):
        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            project=project, name=name, config=config, dir=str(run_dir), reinit=True
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.run.log(metrics, step=step)

    def log_summary(self, summary: dict[str, Any]) -> None:
        for key, value in summary.items():
            self.run.summary[key] = value

    def log_artifact(self, path: str | Path, name: str = "", kind: str = "model") -> None:
        artifact = self._wandb.Artifact(name or Path(path).name, type=kind)
        p = Path(path)
        if p.is_dir():
            artifact.add_dir(str(p))
        else:
            artifact.add_file(str(p))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        self.run.finish()


class MlflowTracker:
    def __init__(self, project: str, name: str, config: dict):
        import mlflow

        self._mlflow = mlflow
        mlflow.set_experiment(project)
        self.run = mlflow.start_run(run_name=name)
        mlflow.log_params(_flatten(config))

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric:
            self._mlflow.log_metrics(numeric, step=step)

    def log_summary(self, summary: dict[str, Any]) -> None:
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                self._mlflow.log_metric(key, value)
            else:
                self._mlflow.set_tag(key, str(value)[:500])

    def log_artifact(self, path: str | Path, name: str = "", kind: str = "model") -> None:
        p = Path(path)
        if p.is_dir():
            self._mlflow.log_artifacts(str(p), artifact_path=name or p.name)
        else:
            self._mlflow.log_artifact(str(p), artifact_path=name or None)

    def finish(self) -> None:
        self._mlflow.end_run()


class MultiTracker:
    """Fans every call out to the local log plus whatever remote backend loaded."""

    def __init__(self, trackers: list[Tracker]):
        self.trackers = trackers

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        for t in self.trackers:
            try:
                t.log(metrics, step=step)
            except Exception:  # noqa: BLE001 - tracking must never kill a run
                pass

    def log_summary(self, summary: dict[str, Any]) -> None:
        for t in self.trackers:
            try:
                t.log_summary(summary)
            except Exception:  # noqa: BLE001
                pass

    def log_artifact(self, path: str | Path, name: str = "", kind: str = "model") -> None:
        for t in self.trackers:
            try:
                t.log_artifact(path, name=name, kind=kind)
            except Exception:  # noqa: BLE001
                pass

    def finish(self) -> None:
        for t in self.trackers:
            try:
                t.finish()
            except Exception:  # noqa: BLE001
                pass


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{full}."))
        else:
            out[full] = value
    return out


def build_tracker(kind: str, project: str, name: str, config: dict, run_dir: Path) -> MultiTracker:
    """Assemble the tracker stack.

    ``auto`` uses W&B when an API key is present, MLflow when a tracking URI
    is set, and otherwise stays local-only. That ordering means the pipeline
    works on a fresh machine with no credentials and light up automatically on
    a machine that has them.
    """
    trackers: list[Tracker] = [LocalTracker(run_dir, config)]
    flat = _flatten(config)

    resolved = kind
    if kind == "auto":
        if os.environ.get("WANDB_API_KEY"):
            resolved = "wandb"
        elif os.environ.get("MLFLOW_TRACKING_URI"):
            resolved = "mlflow"
        else:
            resolved = "local"

    if resolved == "wandb":
        try:
            trackers.append(WandbTracker(project, name, flat, run_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"[tracking] W&B unavailable ({exc}); continuing with local logging.")
    elif resolved == "mlflow":
        try:
            trackers.append(MlflowTracker(project, name, flat))
        except Exception as exc:  # noqa: BLE001
            print(f"[tracking] MLflow unavailable ({exc}); continuing with local logging.")

    return MultiTracker(trackers)
