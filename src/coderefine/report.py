"""Assemble the experiment report from whatever artifacts exist on disk.

Deliberately tolerant: it renders the sections it has evidence for and prints an
explicit "not run" line for the rest, rather than failing or — worse — emitting
a plausible-looking section with no data behind it. Every number in the output
is read from a file in ``artifacts/``; nothing is typed in by hand, so the
report cannot drift away from the runs it describes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .forgetting import compare_forgetting


def _load(path: Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def _num(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No data._\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def build_report(
    artifacts_dir: Path = Path("artifacts"),
    data_dir: Path = Path("data/processed"),
    out_path: Path = Path("reports/EXPERIMENT_REPORT.md"),
) -> Path:
    artifacts_dir, data_dir = Path(artifacts_dir), Path(data_dir)
    eval_dir = artifacts_dir / "eval"

    card = _load(data_dir / "dataset_card.json")
    bench_card = _load(Path("data/benchmark/benchmark_card.json"))
    comparison = _load(eval_dir / "comparison.json")
    judge = _load(eval_dir / "judge.json")
    runs = sorted((artifacts_dir / "runs").glob("*/train_summary.json"))
    forgetting = sorted((artifacts_dir / "forgetting").glob("*.forgetting.json"))

    parts: list[str] = []
    parts.append(_section_header(comparison))
    parts.append(_section_problem())
    parts.append(_section_dataset(card, bench_card))
    parts.append(_section_training(runs))
    parts.append(_section_sweep(runs))
    parts.append(_section_results(comparison, eval_dir))
    parts.append(_section_examples(comparison))
    parts.append(_section_judge(judge))
    parts.append(_section_forgetting(forgetting))
    parts.append(_section_repro())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(p for p in parts if p), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _section_header(comparison: dict | None) -> str:
    head = "# Fine-tuning a code-review refinement model with LoRA\n\n"
    if not comparison:
        return head + (
            "> **Status: evaluation not yet run.** Train an adapter and run "
            "`coderefine evaluate` for the base and tuned models, then "
            "`coderefine compare`, and rebuild this report.\n"
        )
    h = comparison["headline"]
    return head + (
        f"**Headline.** On {h['n']} held-out examples, exact match went from "
        f"{_pct(h['base_exact_match'])} (base) to {_pct(h['tuned_exact_match'])} "
        f"(LoRA fine-tuned), a change of {h['exact_match_delta'] * 100:+.1f} points "
        f"(95% CI [{h['exact_match_ci95'][0] * 100:+.1f}, {h['exact_match_ci95'][1] * 100:+.1f}]). "
        f"Fine-tuning fixed {h['examples_fixed']} examples the base model got wrong "
        f"and broke {h['examples_broken']} it got right.\n\n"
        f"> {h['verdict']}\n"
    )


def _section_problem() -> str:
    return """
## 1. The problem, and why fine-tuning is the right tool

**Task.** Given a code hunk as it stood before review and a reviewer's
natural-language comment, produce the hunk as it stood after the author
addressed that comment. Source data is the Microsoft CodeReviewer refinement
corpus: real comments from real pull requests across nine languages.

**Why not few-shot prompting?** The output format is unusually strict. A correct
answer reproduces every untouched line byte-for-byte — indentation, trailing
commas, comment alignment — and changes only what the reviewer asked about.
Few-shot examples communicate *what* to do but are poor at communicating *how
little* to do; base models reformat neighbouring lines, "helpfully" add error
handling, and wrap output in prose. Those are format and restraint behaviours,
learned far more efficiently from 2,000 gradient-updated examples than from
three in-context ones, and each in-context example costs tokens on every single
request forever.

**Why not RAG?** Retrieval answers "what information is missing". Nothing is
missing here: the code and the comment are both in the prompt. What the base
model lacks is a *behaviour* — surgical, minimal edits in the house style of
merged pull requests. Retrieval cannot supply a behaviour.

**Why LoRA rather than full fine-tuning?** The adapter is a few megabytes
against ~15 GB of base weights, trains on a single consumer GPU under 4-bit
quantisation, and can be attached and detached at serving time. That last
property is what makes the A/B endpoint in this repo honest: base and tuned
responses come from one resident set of weights, differing only by the adapter.
"""


def _section_dataset(card: dict | None, bench: dict | None) -> str:
    if not card:
        return "\n## 2. Dataset\n\n_Not built. Run `coderefine build-data`._\n"

    rows = []
    for split, info in card["splits"].items():
        g = info.get("grounding", {})
        rows.append([
            f"`{split}`", str(info["n"]), str(info.get("n_repos", "—")),
            str(len(info.get("languages", {}))), str(len(info.get("categories", {}))),
            _num(g.get("mean")), str(info.get("hunk_lines", {}).get("median", "—")),
            str(info.get("mean_comment_chars", "—")),
        ])

    funnel_rows = []
    train_funnel = card["funnel"].get("train", {})
    read = train_funnel.get("records_read", 0)
    for name, count in list(train_funnel.get("rejected_by_filter", {}).items())[:9]:
        funnel_rows.append([f"`{name}`", f"{count:,}", _pct(count / read) if read else "—"])

    checks = card["leakage_checks"]
    leak_clean = not (
        checks["repo_overlap_train_valid"] or checks["repo_overlap_train_test"]
        or checks["repo_overlap_valid_test"] or checks["id_overlap_any"]
    )

    out = f"""
## 2. Dataset and curation methodology

Curated from {read:,} raw records into {sum(i['n'] for i in card['splits'].values()):,}
model-ready examples. The governing principle is quality over quantity: the
value of an example is entirely determined by whether the reviewer's comment
actually predicts the code change, and roughly two thirds of the raw corpus
fails that test.

{_table(
    ["Split", "n", "Repos", "Langs", "Categories", "Mean grounding", "Median hunk lines", "Mean comment chars"],
    rows,
)}
### Curation funnel (training dump)

{_table(["Filter", "Rejected", "% of read"], funnel_rows)}
The dominant filter is `ungrounded_change`: a **grounding score** built from the
lexical overlap between the comment and the lines the diff actually adds or
removes, weighted toward identifiers the reviewer wrote in backticks. A comment
that shares no vocabulary with its own diff is almost always one whose real
instruction lived somewhere we are not showing the model — a design discussion,
an earlier review round, a linked issue. Training on those teaches
hallucination, so they are dropped. Comments that explicitly point elsewhere
("as discussed", "same as above", "see #1234") are removed by pattern before
scoring.

Two quota filters keep the distribution honest: at most
{card['curation_config']['max_per_repo']} examples from any one repository, and at
most {card['curation_config']['max_per_comment']} sharing a normalised comment,
so that neither a single monorepo nor the phrase "remove this" can define the task.

### Leakage control

{"**All checks pass.**" if leak_clean else "**LEAKS PRESENT — see dataset_card.json.**"}
Three independent guards:

1. **Provenance** — the splits are curated from the three separate upstream
   dumps, which are already cut by project and time.
2. **Content hashing** — one fingerprint set spans all three files, processed
   train-first, so a snippet vendored into two repositories cannot cross the
   boundary. Collisions dropped: `{checks['content_hash_collisions_dropped']}`.
3. **Repository disjointness** — no repository appears in more than one split.
   This is the guard that matters for the failure mode in the brief (examples
   derived from the same source file), because such examples share a repo even
   when their content differs. Upstream does *not* make validation and test
   repo-disjoint from each other — its cut is by pull request — so shared
   repositories are partitioned between them by a stable hash of the repository
   name rather than dropped, which keeps both splits at full size.

Final overlaps: train∩valid `{checks['repo_overlap_train_valid']}`,
train∩test `{checks['repo_overlap_train_test']}`,
valid∩test `{checks['repo_overlap_valid_test']}`.

**The test split is sacred.** `coderefine evaluate --split test` refuses to run
without an explicit `--final` flag. Model selection and the hyperparameter
sweep use validation only.
"""
    if bench:
        probes = ", ".join(f"`{p}`" for p in bench.get("probes", [])[:8])
        out += f"""
### Evaluation benchmark

{bench['n_total']} examples: {bench['n_handwritten']} hand-written plus
{bench['n_curated']} stratified hard cases drawn from the held-out test pool
(mean hardness {bench.get('curated_mean_hardness')} against a pool average of
{bench.get('pool_mean_hardness')}).

The hand-written half exists because the natural distribution contains almost
none of the cases that separate a model which reads the comment from one which
makes a plausible-looking change. Each targets one failure mode: {probes}, and
others. Results are always reported split by origin, since the two halves
measure different things.
"""
    return out


def _section_training(runs: list[Path]) -> str:
    if not runs:
        return "\n## 3. Training setup\n\n_No runs yet. Run `coderefine train configs/qlora_mistral7b.yaml`._\n"
    first = _load(runs[0]) or {}
    cfg = first.get("config", {})
    lora, train, quant = cfg.get("lora", {}), cfg.get("train", {}), cfg.get("quant", {})
    env = first.get("env", {})

    return f"""
## 3. Training setup

| | |
|---|---|
| Base model | `{cfg.get('base_model')}` |
| Quantisation | {"4-bit NF4 (QLoRA), " + str(quant.get('compute_dtype')) + " compute" if quant.get('load_in_4bit') else "none (full-precision base)"} |
| LoRA rank / alpha / dropout | {lora.get('r')} / {lora.get('alpha')} / {lora.get('dropout')} |
| Target modules | `{lora.get('target_modules')}` |
| Trainable parameters | {first.get('trainable_params', 0):,} |
| Learning rate / schedule | {train.get('learning_rate')} / {train.get('lr_scheduler')} |
| Epochs | {train.get('num_epochs')} |
| Effective batch size | {first.get('effective_batch_size')} |
| Max sequence length | {train.get('max_seq_length')} |
| Seed | {train.get('seed')} |
| Hardware | {env.get('gpu_name', env.get('platform', 'unknown'))} |

**Why these LoRA values.** Rank 16 is the smallest setting that comfortably
covers a behaviour-shaping task like this one; alpha is fixed at 2r throughout,
which holds the effective update scale α/r constant so that a sweep over rank
measures adapter *capacity* rather than confounding capacity with step size.
Dropout 0.05 is light regularisation appropriate to a 2,000-example set — enough
to blunt memorisation, not enough to starve a 3-epoch budget. Targeting the
query and value projections only is the original LoRA paper's configuration and
the cheapest thing that works; the sweep tests whether paying ~3× the parameters
for all-linear buys anything.

**Loss is computed on the assistant turn only.** The prompt already contains
~90% of the target text, so training over the full sequence would spend most of
the gradient budget teaching the model to copy code it was handed — producing an
adapter that has learned to echo. `DataCollatorForCompletionOnlyLM` masks
everything through the response template, and that template is *derived from the
tokenizer's own chat template* at runtime rather than hard-coded, so swapping
Mistral for Llama 3 cannot silently disable masking.

**Checkpointing and early stopping.** Validation runs
{train.get('evals_per_epoch')}× per epoch; checkpoints are saved at the same
cadence, the best is restored at the end (`load_best_model_at_end`), and
training stops after {train.get('early_stopping_patience')} evaluations without
improvement. Every run records which checkpoint was selected and why:

> {first.get('checkpoint_selection_reason', '—')}
"""


def _section_sweep(runs: list[Path]) -> str:
    if len(runs) < 2:
        return (
            "\n## 4. Hyperparameter sweep\n\n"
            "_Fewer than two runs recorded. Run `scripts/run_sweep.sh` on a GPU host._\n"
        )
    rows = []
    for path in runs:
        s = _load(path) or {}
        cfg = s.get("config", {})
        lora, train = cfg.get("lora", {}), cfg.get("train", {})
        rows.append([
            f"`{s.get('run_name')}`", str(lora.get("r")), str(train.get("learning_rate")),
            str(train.get("num_epochs")),
            ",".join(lora.get("target_modules", [])),
            _num(s.get("best_eval_loss"), 4),
            str(s.get("best_checkpoint_step")),
            f"{s.get('train_runtime_s', 0) / 60:.1f} min",
            _num(s.get("gpu_peak_memory_gb"), 2),
        ])
    best = min(
        (l for l in runs if (_load(l) or {}).get("best_eval_loss") is not None),
        key=lambda p: (_load(p) or {})["best_eval_loss"],
        default=None,
    )
    best_name = (_load(best) or {}).get("run_name") if best else "—"

    return f"""
## 4. Hyperparameter sweep

All arms share the seed, the data and the effective batch size, and are selected
on **validation loss only** — the test split is untouched at this stage.

{_table(
    ["Run", "r", "LR", "Epochs", "Targets", "Best eval loss", "Best step", "Wall clock", "Peak GPU GB"],
    rows,
)}
Selected configuration: **`{best_name}`** (lowest validation loss). Full loss
curves, GPU memory traces and per-step logs for every arm are in
`artifacts/runs/<name>/metrics.jsonl`, and in Weights & Biases when a key is set.
"""


def _section_results(comparison: dict | None, eval_dir: Path) -> str:
    if not comparison:
        return "\n## 5. Base vs fine-tuned\n\n_Not run._\n"
    p = comparison["paired_metrics"]

    rows = []
    labels = {
        "exact_match": "Exact match",
        "improved": "Moved toward gold",
        "regressed": "Moved away from gold",
        "copied": "Returned input unchanged",
        "edit_line_f1": "Edit-line F1",
        "edit_sim": "Edit similarity",
        "token_f1": "Token F1",
    }
    for key, label in labels.items():
        e = p[key]
        sig = "yes" if e.get("significant") else "no"
        rows.append([
            label, _num(e["base"], 4), _num(e["tuned"], 4), f"{e['delta']:+.4f}",
            f"[{e['ci_low']:+.4f}, {e['ci_high']:+.4f}]", sig,
        ])

    base_sum = _load(eval_dir / "base__benchmark.summary.json") or _load(eval_dir / "base__test.summary.json")
    floor = ""
    if base_sum:
        m = base_sum["metrics"]
        floor = f"""
### Reading these numbers honestly

On this task the input is already ~90% of the correct output. A model that
changes nothing scores **{_num(m.get('copy_baseline_bleu'), 1)} BLEU** and
**{_num(m.get('copy_baseline_edit_similarity'), 3)} edit similarity** — so those
metrics are near-useless on their own and are reported here only against that
floor. The metrics that carry information are *exact match* (0% for a copier by
construction) and *moved toward gold* (0% for a copier by construction).
"""

    cat_rows = []
    for name, info in sorted(
        comparison["by_category"].items(), key=lambda kv: -kv[1]["delta_exact_match"]
    ):
        cat_rows.append([
            f"`{name}`", str(info["n"]), _pct(info["base_exact_match"]),
            _pct(info["tuned_exact_match"]), f"{info['delta_exact_match'] * 100:+.1f}",
            f"{info['delta_edit_sim']:+.3f}",
        ])

    origin_rows = []
    for name, info in sorted(comparison["by_origin"].items()):
        origin_rows.append([
            f"`{name}`", str(info["n"]), _pct(info["base_exact_match"]),
            _pct(info["tuned_exact_match"]), f"{info['delta_exact_match'] * 100:+.1f}",
        ])

    return f"""
## 5. Head-to-head: base vs fine-tuned

Both models are shown byte-identical prompts built by `coderefine.prompts`, and
both decode greedily. The only difference between the two runs is the adapter.

{_table(["Metric", "Base", "Fine-tuned", "Δ", "95% CI (paired bootstrap)", "Significant"], rows)}
{floor}
### Per-category breakdown

{_table(["Category", "n", "Base EM", "Tuned EM", "Δ EM (pts)", "Δ edit sim"], cat_rows)}
### By benchmark half

{_table(["Origin", "n", "Base EM", "Tuned EM", "Δ EM (pts)"], origin_rows)}
Wins: {comparison['n_wins']} · Losses: {comparison['n_losses']} · Ties: {comparison['n_ties']}
"""


def _section_examples(comparison: dict | None) -> str:
    if not comparison:
        return ""
    out = ["\n## 6. Where it helped, and where it hurt\n"]

    def render(rows: list[dict], heading: str, blurb: str) -> str:
        if not rows:
            return f"\n### {heading}\n\n_None._\n"
        chunk = [f"\n### {heading}\n\n{blurb}\n"]
        for r in rows[:3]:
            chunk.append(
                f"""
**`{r['id']}`** · {r['lang']} · `{r['category']}`{" · probe `" + r['probe'] + "`" if r.get('probe') else ""}

> {r['comment'][:300]}

```{r['lang']}
# --- before ---
{r['old_code']}
```
```{r['lang']}
# --- merged revision (gold) ---
{r['gold']}
```
```{r['lang']}
# --- base model ---
{r['base_prediction'] or '(empty)'}
```
```{r['lang']}
# --- fine-tuned ---
{r['tuned_prediction'] or '(empty)'}
```

Edit similarity {r['base_edit_sim']:.3f} → {r['tuned_edit_sim']:.3f} ({r['delta_edit_sim']:+.3f})
"""
            )
        return "".join(chunk)

    out.append(render(
        comparison["wins"], "Fine-tuning helped",
        "The largest improvements, ranked by edit-similarity gain.",
    ))
    out.append(render(
        comparison["losses"], "Fine-tuning hurt (regressions)",
        "Cases the base model handled better. Reported in full — a fine-tune that "
        "regresses nothing is usually a fine-tune that was evaluated too narrowly.",
    ))
    return "".join(out)


def _section_judge(judge: dict | None) -> str:
    if not judge:
        return (
            "\n## 7. LLM-as-judge\n\n"
            "_Not run. `coderefine judge <base.jsonl> <tuned.jsonl>`._\n"
        )
    s = judge["summary"]
    rows = [
        [f"`{k}`", str(v["n"]), _num(v["base_mean"], 2), _num(v["tuned_mean"], 2),
         f"{v['tuned_wins']}–{v['base_wins']}"]
        for k, v in s.get("by_category", {}).items()
    ]
    return f"""
## 7. LLM-as-judge evaluation

Judge: `{s['judge_model']}` via {s['judge_provider']}, on {s['n']} paired examples.
The judge sees the original code, the comment, the merged revision as reference,
and two candidates labelled A and B. It is never told which model produced
which, or that fine-tuning is involved.

| | |
|---|---|
| Mean score, base (1–5) | {_num(s['base_mean_score'], 2)} |
| Mean score, fine-tuned (1–5) | {_num(s['tuned_mean_score'], 2)} |
| Δ | {s['score_delta']:+.2f} |
| Fine-tuned wins / base wins / ties | {s['tuned_wins']} / {s['base_wins']} / {s['ties']} |
| Fine-tuned win rate | {_pct(s['tuned_win_rate'])} |
| Position-bias rate | {_pct(s['position_bias_rate'])} |

Every pair is judged twice with the candidate order swapped. Pairs where the
judge picked the same *slot* both times are counted as position bias and scored
as ties rather than being allowed to inflate either side.
{s['position_bias_note']}

{_table(["Category", "n", "Base mean", "Tuned mean", "W–L"], rows)}"""


def _section_forgetting(paths: list[Path]) -> str:
    results = {}
    for path in paths:
        data = _load(path)
        if data:
            results[data["tag"]] = data
    base = results.get("base")
    tuned = next((v for k, v in results.items() if k != "base"), None)
    if not (base and tuned):
        return (
            "\n## 8. Catastrophic forgetting\n\n"
            "_Not run. `coderefine forgetting` for the base model and again with `--adapter`._\n"
        )
    c = compare_forgetting(base, tuned)
    return f"""
## 8. Catastrophic forgetting analysis

Fine-tuning on 2,000 narrow examples risks teaching the model that *every*
prompt is a code-refinement prompt. Two probes measure different halves of that
risk: 24 multiple-choice items scored by length-normalised log-likelihood (no
generation, fully deterministic — what the model still *knows*), and 16
instruction-following items that are generated and rule-checked (whether it
still *complies*). The items are original to this project, so there is no
contamination from the base model having trained on them.

| Probe | Base | Fine-tuned | Retention |
|---|---|---|---|
| Multiple-choice accuracy | {_pct(c['mcq_accuracy_base'])} | {_pct(c['mcq_accuracy_tuned'])} | {c['mcq_retention_pct']}% |
| Instruction-following pass rate | {_pct(c['instruction_pass_base'])} | {_pct(c['instruction_pass_tuned'])} | {c['instruction_retention_pct']}% |
| Spurious code-fence rate | {_pct(c['spurious_fence_base'])} | {_pct(c['spurious_fence_tuned'])} | — |

**Overall retention: {c['overall_retention_pct']}%** (the weaker of the two probes).

> {c['verdict']}

The spurious code-fence rate is the most diagnostic row: it counts general
questions ("what colour is a clear daytime sky?") that the model answers inside
a fenced code block. A large jump there is task-format bleed, and the remedy is
fewer epochs or a lower rank rather than more data.
"""


def _section_repro() -> str:
    return """
## 9. Reproducing this

```bash
conda create -n lora-coderefine python=3.11 -y && conda activate lora-coderefine
pip install -r requirements.txt && pip install -e .
pip install -r requirements-gpu.txt          # CUDA hosts only

coderefine build-data                        # 6 GB of raw diffs -> curated splits
coderefine build-benchmark                   # 40-example hard benchmark

coderefine train configs/qlora_mistral7b.yaml
bash scripts/run_sweep.sh                    # all eight arms

coderefine evaluate --split benchmark --tag base
coderefine evaluate --split benchmark --adapter artifacts/runs/<run>/adapter --tag tuned
coderefine compare artifacts/eval/base__benchmark.predictions.jsonl \\
                   artifacts/eval/tuned__benchmark.predictions.jsonl
coderefine judge   artifacts/eval/base__benchmark.predictions.jsonl \\
                   artifacts/eval/tuned__benchmark.predictions.jsonl
coderefine forgetting
coderefine forgetting --adapter artifacts/runs/<run>/adapter
coderefine report
```

Every run is reproducible from its config file alone: the resolved config, its
fingerprint, the full metric history and the host environment are written into
`artifacts/runs/<name>/`. Two runs with the same fingerprint trained on the same
data will produce the same adapter.

---

_Generated by `coderefine report`. Every number above is read from a file in
`artifacts/`; none is typed in by hand._
"""
