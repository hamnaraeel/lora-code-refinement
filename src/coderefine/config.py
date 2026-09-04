"""Typed run configuration.

Reproducibility rule for this project: **a run must be reproducible from its
config file alone.** So the config carries everything that changes the outcome
— model, LoRA shape, optimiser, seed, data paths — and nothing that does not.
The resolved config is written into every run directory and logged to the
experiment tracker as-is, which is what makes the sweep table meaningful.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoraConfigSpec:
    """LoRA hyperparameters.

    Defaults are the ones justified in the report: rank 16 with alpha 32
    (the conventional alpha = 2r, which keeps the effective update scale
    alpha/r fixed at 2 as rank varies, so a rank sweep measures capacity rather
    than confounding capacity with step size), dropout 0.05.

    ``target_modules`` defaults to attention projections only (q_proj, v_proj),
    the original LoRA paper's configuration and the cheapest thing that works.
    The sweep also tries the "all-linear" variant that adds k/o and the MLP
    projections, which costs roughly 3x the adapter parameters.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    task_type: str = "CAUSAL_LM"


@dataclass
class QuantConfig:
    """4-bit NF4 quantisation (QLoRA). CUDA only — bitsandbytes has no CPU build."""

    load_in_4bit: bool = False
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    double_quant: bool = True


@dataclass
class TrainConfig:
    learning_rate: float = 2e-4
    num_epochs: float = 3.0
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = True
    #: Evaluate and checkpoint this many times per epoch. Frequent evaluation is
    #: what makes early stopping meaningful on a run this short.
    evals_per_epoch: int = 4
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 1e-4
    max_grad_norm: float = 0.3
    seed: int = 42
    #: Cap on training examples, for smoke runs.
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


@dataclass
class RunConfig:
    name: str = "default"
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    output_root: str = "artifacts/runs"
    data_dir: str = "data/processed"
    #: "auto" resolves to cuda > mps > cpu at runtime.
    device: str = "auto"
    dtype: str = "auto"
    attn_implementation: str = "auto"
    tracker: str = "auto"  # auto | wandb | mlflow | local | none
    project: str = "lora-code-refinement"
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    quant: QuantConfig = field(default_factory=QuantConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""

    # -- construction -----------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "RunConfig":
        """Load a config, resolving the full ``extends:`` chain.

        Sweep configs are thin — they set ``extends: ../qlora_mistral7b.yaml``
        and change two fields — which keeps the difference between arms visible
        at a glance instead of buried in eight near-identical files. Chains are
        resolved to arbitrary depth (arm -> qlora -> base), because stopping at
        one level would silently drop the grandparent's settings and every arm
        would quietly train with different defaults than the config it claims
        to extend.
        """
        raw = _load_with_extends(Path(path), seen=[])
        if overrides:
            raw = _deep_merge(raw, overrides)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        raw = dict(raw)
        nested = {"lora": LoraConfigSpec, "quant": QuantConfig, "train": TrainConfig}
        kwargs: dict[str, Any] = {}
        known = {f.name for f in fields(cls)}
        for key, value in raw.items():
            if key not in known:
                raise ValueError(f"Unknown config key: {key!r}")
            if key in nested and isinstance(value, dict):
                sub = nested[key]
                sub_known = {f.name for f in fields(sub)}
                unknown = set(value) - sub_known
                if unknown:
                    raise ValueError(f"Unknown keys in {key}: {sorted(unknown)}")
                kwargs[key] = sub(**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    # -- identity ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable hash of everything that affects the result.

        Two runs with the same fingerprint should produce the same adapter.
        ``name`` and ``notes`` are excluded so that renaming a run does not
        make it look like a different experiment.
        """
        payload = self.to_dict()
        payload.pop("name", None)
        payload.pop("notes", None)
        payload.pop("output_root", None)
        payload.pop("tracker", None)
        payload.pop("project", None)
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.name

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["_fingerprint"] = self.fingerprint()
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_with_extends(path: Path, seen: list[Path]) -> dict:
    """Recursively resolve ``extends`` into a single flat mapping."""
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in seen + [path])
        raise ValueError(f"Circular config inheritance: {chain}")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if not parent:
        return raw

    parent_path = Path(parent)
    if not parent_path.is_absolute():
        # Relative to the child config, so a config directory can be relocated
        # wholesale without rewriting every `extends` line.
        candidate = path.parent / parent_path
        parent_path = candidate if candidate.exists() else parent_path
    base = _load_with_extends(parent_path, seen + [path])
    return _deep_merge(base, raw)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn ``--set train.learning_rate=1e-4`` style pairs into a nested dict."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return out


def _coerce(value: str):
    """Parse an override value, repairing YAML 1.1's scientific-notation gap.

    ``yaml.safe_load("1e-4")`` returns the *string* ``"1e-4"`` — the YAML 1.1
    float pattern requires a decimal point and a signed exponent, so only
    ``1.0e-4`` parses as a number. Left alone, ``--set train.learning_rate=1e-4``
    would hand a string to the optimiser and either crash deep in the scheduler
    or, worse, be silently coerced somewhere downstream. Anything that survives
    as a string but parses as a float is converted here.
    """
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):
        try:
            return float(parsed)
        except ValueError:
            return parsed
    return parsed
