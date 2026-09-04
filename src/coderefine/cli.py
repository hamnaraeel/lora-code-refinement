"""Command line entry point: ``python -m coderefine <command>``.

Commands are thin wrappers. All the logic lives in the modules they call, so
each stage is equally usable from a notebook (the Colab training path) as from
a shell.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help="LoRA code-refinement pipeline.")
console = Console()

DEFAULT_RAW = Path("Code_Refinement")
DEFAULT_PROCESSED = Path("data/processed")


@app.command("build-data")
def build_data(
    raw_dir: Path = typer.Option(DEFAULT_RAW, help="Directory holding ref-*.jsonl."),
    out_dir: Path = typer.Option(DEFAULT_PROCESSED, help="Where to write splits."),
    n_train: int = typer.Option(2000),
    n_valid: int = typer.Option(250),
    n_test: int = typer.Option(250),
    scan_limit: Optional[int] = typer.Option(
        200_000, help="Raw records to read per file (None = whole file)."
    ),
    seed: int = typer.Option(13),
) -> None:
    """Curate raw diffs into leak-free train/valid/test splits."""
    from .data.build import build_splits
    from .data.curate import CurationConfig

    card = build_splits(
        raw_dir=raw_dir,
        out_dir=out_dir,
        cfg=CurationConfig(),
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        scan_limit=scan_limit,
        seed=seed,
    )

    table = Table(title="Curated splits", show_lines=False)
    table.add_column("split"); table.add_column("n", justify="right")
    table.add_column("repos", justify="right"); table.add_column("mean grounding", justify="right")
    table.add_column("langs", justify="right"); table.add_column("categories", justify="right")
    for split, info in card["splits"].items():
        table.add_row(
            split, str(info["n"]), str(info.get("n_repos", 0)),
            f"{info.get('grounding', {}).get('mean', 0):.3f}",
            str(len(info.get("languages", {}))), str(len(info.get("categories", {}))),
        )
    console.print(table)

    checks = card["leakage_checks"]
    leaks = (
        checks["repo_overlap_train_valid"]
        + checks["repo_overlap_train_test"]
        + checks["repo_overlap_valid_test"]
        + checks["id_overlap_any"]
    )
    if leaks:
        console.print(f"[bold red]LEAK DETECTED:[/] {leaks[:10]}")
        raise typer.Exit(1)
    console.print("[bold green]Leakage checks passed[/] (no shared repos or ids across splits)")
    console.print(f"Dataset card -> {out_dir / 'dataset_card.json'}")


@app.command("build-benchmark")
def build_benchmark(
    test_path: Path = typer.Option(DEFAULT_PROCESSED / "test.jsonl"),
    handwritten: Path = typer.Option(Path("data/benchmark/handwritten.jsonl")),
    out_path: Path = typer.Option(Path("data/benchmark/benchmark.jsonl")),
    n_curated: int = typer.Option(25, help="Stratified picks from the held-out test pool."),
    seed: int = typer.Option(7),
) -> None:
    """Assemble the hard evaluation benchmark (curated + hand-written)."""
    from .data.benchmark import build_benchmark as _build

    card = _build(test_path, handwritten, out_path, n_curated=n_curated, seed=seed)
    console.print(json.dumps(card, indent=2))


@app.command("train")
def train(
    config: Path = typer.Argument(..., help="YAML run config."),
    set_: list[str] = typer.Option([], "--set", help="Override, e.g. train.num_epochs=1"),
    resume: bool = typer.Option(
        False, "--resume",
        help="Resume from this run's latest checkpoint (same output_root/name) instead of starting fresh. "
        "Combine with --set train.num_epochs=N to raise the target — e.g. run once with "
        "num_epochs=1, then again with --resume --set train.num_epochs=2 to train one more epoch.",
    ),
) -> None:
    """Fine-tune a LoRA adapter."""
    from .config import RunConfig, parse_overrides
    from .train import run_training

    cfg = RunConfig.from_yaml(config, parse_overrides(set_))
    result = run_training(cfg, resume=resume)
    console.print(json.dumps(result, indent=2, default=str))


@app.command("evaluate")
def evaluate(
    split: str = typer.Option("valid", help="valid | test | benchmark"),
    base_model: str = typer.Option("mistralai/Mistral-7B-Instruct-v0.3"),
    adapter: Optional[Path] = typer.Option(None, help="LoRA adapter dir; omit for the base model."),
    tag: str = typer.Option("", help="Label for the output file; defaults to base/adapter name."),
    limit: Optional[int] = typer.Option(None),
    max_new_tokens: int = typer.Option(512),
    batch_size: int = typer.Option(4),
    device: str = typer.Option("auto"),
    load_in_4bit: bool = typer.Option(False),
    final: bool = typer.Option(False, help="Required to score the sacred test split."),
    out_dir: Path = typer.Option(Path("artifacts/eval")),
) -> None:
    """Generate predictions and score them against gold."""
    from .evaluate import run_evaluation

    summary = run_evaluation(
        split=split, base_model=base_model, adapter=adapter, tag=tag, limit=limit,
        max_new_tokens=max_new_tokens, batch_size=batch_size, device=device,
        load_in_4bit=load_in_4bit, final=final, out_dir=out_dir,
    )
    console.print(json.dumps(summary["metrics"], indent=2))


@app.command("compare")
def compare(
    base: Path = typer.Argument(..., help="Predictions JSONL from the base model."),
    tuned: Path = typer.Argument(..., help="Predictions JSONL from the fine-tuned model."),
    out_path: Path = typer.Option(Path("artifacts/eval/comparison.json")),
) -> None:
    """Head-to-head comparison, including per-category wins and regressions."""
    from .compare import compare_runs

    report = compare_runs(base, tuned, out_path)
    console.print(json.dumps(report["headline"], indent=2))


@app.command("judge")
def judge(
    base: Path = typer.Argument(...),
    tuned: Path = typer.Argument(...),
    provider: str = typer.Option("anthropic", help="anthropic | openai"),
    model: str = typer.Option(""),
    limit: Optional[int] = typer.Option(None),
    out_path: Path = typer.Option(Path("artifacts/eval/judge.json")),
) -> None:
    """Blind pairwise LLM-as-judge scoring of base vs fine-tuned outputs."""
    from .judge import run_judge

    report = run_judge(base, tuned, provider=provider, model=model, limit=limit, out_path=out_path)
    console.print(json.dumps(report["summary"], indent=2))


@app.command("forgetting")
def forgetting(
    base_model: str = typer.Option("mistralai/Mistral-7B-Instruct-v0.3"),
    adapter: Optional[Path] = typer.Option(None),
    device: str = typer.Option("auto"),
    load_in_4bit: bool = typer.Option(False),
    limit: Optional[int] = typer.Option(None),
    out_dir: Path = typer.Option(Path("artifacts/forgetting")),
) -> None:
    """Probe general capabilities to quantify catastrophic forgetting."""
    from .forgetting import run_probe

    summary = run_probe(
        base_model=base_model, adapter=adapter, device=device,
        load_in_4bit=load_in_4bit, limit=limit, out_dir=out_dir,
    )
    console.print(json.dumps(summary, indent=2))


@app.command("export")
def export(
    adapter: Path = typer.Argument(...),
    out_dir: Path = typer.Option(Path("artifacts/release")),
    merge: bool = typer.Option(False, help="Also write a merged full-weight model (large)."),
    base_model: str = typer.Option(""),
) -> None:
    """Package the LoRA adapter for deployment."""
    from .export import export_adapter

    manifest = export_adapter(adapter, out_dir, merge=merge, base_model=base_model or None)
    console.print(json.dumps(manifest, indent=2, default=str))


@app.command("serve")
def serve(
    base_model: str = typer.Option("mistralai/Mistral-7B-Instruct-v0.3"),
    adapter: Optional[Path] = typer.Option(None),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    device: str = typer.Option("auto"),
    load_in_4bit: bool = typer.Option(False),
) -> None:
    """Run the A/B inference server (base vs fine-tuned, one loaded base model)."""
    import uvicorn

    from .serve import build_app

    application = build_app(
        base_model=base_model, adapter=adapter, device=device, load_in_4bit=load_in_4bit
    )
    uvicorn.run(application, host=host, port=port)


@app.command("report")
def report(
    out_path: Path = typer.Option(Path("reports/EXPERIMENT_REPORT.md")),
    artifacts: Path = typer.Option(Path("artifacts")),
    data_dir: Path = typer.Option(DEFAULT_PROCESSED),
) -> None:
    """Assemble the experiment report from whatever artifacts exist."""
    from .report import build_report

    path = build_report(artifacts_dir=artifacts, data_dir=data_dir, out_path=out_path)
    console.print(f"Report written to {path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
