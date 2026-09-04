# LoRA fine-tuning pipeline — applying code review comments automatically

An end-to-end pipeline that fine-tunes an open-source LLM to do one narrow thing
well: **take a code hunk and a reviewer's comment, and produce the revised code
the author would have written.**

It covers the full lifecycle — curating 6 GB of raw pull-request diffs into a
high-signal instruction set, QLoRA training with experiment tracking and early
stopping, a hyperparameter sweep, rigorous base-vs-tuned evaluation with
confidence intervals, blind LLM-as-judge scoring, a catastrophic-forgetting
analysis, and a deployable adapter behind an A/B inference server.

```
raw diffs ──► curate ──► splits ──► QLoRA train ──► sweep ──► evaluate ──► report
   6 GB       filter      2000/       PEFT+TRL      8 arms     base vs      auto-
            + score       250/250     + tracking               tuned        generated
                                                                  │
                                                                  └──► adapter ──► vLLM / Ollama / A/B API
```

---

## The task, and why fine-tuning is the right tool for it

Given code as it stood **before** review plus the reviewer's comment, emit the
code as it stood **after**:

> **Comment:** *"This except block swallows the error and returns None, which hides real failures. Just let it propagate."*

```python
# before                                  # after
def load_config(path):                    def load_config(path):
    try:                                      with open(path) as fh:
        with open(path) as fh:                    return json.load(fh)
            return json.load(fh)
    except Exception:
        return None
```

**Not few-shot prompting.** The output format is unusually strict: a correct
answer reproduces every untouched line byte-for-byte and changes only what the
reviewer asked about. In-context examples convey *what* to do but are poor at
conveying *how little* to do — base models reformat neighbouring lines, add
unrequested error handling, and wrap the answer in prose. Those are restraint
and format behaviours, learned far more efficiently from gradient updates, and
in-context examples cost tokens on every request forever.

**Not RAG.** Retrieval answers "what information is missing". Nothing is
missing — the code and the comment are both in the prompt. What the base model
lacks is a *behaviour*, and retrieval cannot supply one.

**LoRA, not full fine-tuning.** The adapter is a few MB against ~15 GB of base
weights, trains on one consumer GPU under 4-bit quantisation, and attaches and
detaches at serving time — which is what lets the A/B endpoint serve both models
from one resident set of weights.

---

## Quick start

```bash
conda create -n lora-coderefine python=3.11 -y && conda activate lora-coderefine
pip install -r requirements.txt && pip install -e .
pip install -r requirements-gpu.txt      # CUDA hosts only (bitsandbytes, vLLM, lm-eval)

make data benchmark                      # curate the corpus, build the benchmark
make test                                # 94 tests
make smoke                               # end-to-end CPU run, small model, minutes
```

Then on a GPU (or via [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb)):

```bash
make train                               # QLoRA on Mistral-7B
make eval-base eval-tuned compare        # head-to-head
make forgetting report                   # capability retention + the report
make serve                               # A/B API on :8000
```

### Hardware

| Path | Requirement | What it produces |
|---|---|---|
| `configs/qlora_mistral7b.yaml` | CUDA, ≥16 GB VRAM | The headline 7B results |
| `notebooks/colab_train.ipynb` | Colab T4 (free) | Same, ~50–70 min |
| `configs/local_cpu_full.yaml` | Any CPU, 16 GB RAM | Real numbers on a 0.5B code model |
| `configs/local_cpu_smoke.yaml` | Any CPU | Pipeline correctness check, minutes |

> QLoRA needs `bitsandbytes`, which is CUDA-only — there is no CPU or Apple-MPS
> build. The CPU configs therefore run unquantised on a small model. The code
> path is otherwise identical, which is the point: the smoke run exercises chat
> templating, loss masking, checkpointing, early stopping, tracking and adapter
> export exactly as the GPU run does.

---

## What makes the numbers trustworthy

This is the part that matters, so it is the part with the most engineering in it.

### One prompt template, one source of truth

`coderefine.prompts` is imported by training, base evaluation, tuned evaluation
and the server. Base and fine-tuned models are shown **byte-identical** prompts
and both decode greedily. A test asserts the training prompt is a strict prefix
extension of the generation prompt, so the two can never silently diverge.

### Three independent leakage guards

1. **Provenance** — splits are curated from the three separate upstream dumps.
2. **Content hashing** — one fingerprint set spans all three files, train-first,
   so a snippet vendored into two repos cannot cross the boundary.
3. **Repository disjointness** — no repository appears in two splits. This is
   the guard that catches examples derived from the same source file, which
   share a repo even when their content differs.

Upstream does *not* make validation and test repo-disjoint from each other (its
cut is by pull request), so shared repositories are **partitioned** between them
by a stable hash rather than dropped — dropping removed 99% of the test split.

**The test split is sacred**: `evaluate --split test` refuses to run without an
explicit `--final` flag.

### Metrics that a do-nothing model cannot game

On this task the input is already ~90% of the correct output, so **a model that
echoes its input scores over 80 BLEU**. Reporting BLEU alone would make doing
nothing look excellent. So:

| Metric | What it resists |
|---|---|
| **Exact match** | 0% for a copier, by construction |
| **Improved rate** — is the prediction closer to gold than the input was? | 0% for a copier, by construction |
| **Edit-line F1** — F1 over the lines the edit *touches* | Untouched context can't inflate it |
| **Copy rate** | Reported directly, so echoing is visible |
| BLEU / edit similarity | Always printed next to `copy_baseline_*` floors |

Every headline delta carries a **paired bootstrap 95% CI** and an **exact
McNemar test**. At n=40 a 5-point difference is two examples, and the report
says so rather than quoting it as a result.

### A benchmark built to break things

40 examples in two halves, always reported separately:

- **15 hand-written**, one per expected failure mode — over-editing a distractor
  identifier, an edit that is a *deletion*, an edit that is only a change of
  *indentation*, a comment phrased as a question, a typo next to an
  already-correct twin, a reviewer `suggestion` block to be used verbatim.
- **25 stratified hard cases** from the held-out test pool (mean hardness 0.50
  vs a pool average of 0.28).

`suggested_edit` — where the reviewer pasted the answer in a ```suggestion block
— is scored as its own category, because folding those into the headline would
inflate it.

### Quality over quantity in curation

~150k raw records become 2,000 training examples. The dominant filter is a
**grounding score**: lexical overlap between the comment and the lines the diff
actually adds or removes, weighted toward identifiers the reviewer wrote in
backticks. A comment sharing no vocabulary with its own diff almost always had
its real instruction somewhere we aren't showing the model — so training on it
teaches hallucination. Quotas cap any one repository at 40 examples and any one
comment phrasing at 3.

### Loss on the answer only

The prompt contains ~90% of the target text. Training over the full sequence
would spend the gradient budget teaching the model to copy. The response
template used for masking is **derived from the tokenizer's own chat template at
runtime**, not hard-coded, so swapping Mistral for Llama 3 cannot silently
disable masking and turn the run into an expensive no-op.

---

## Repository layout

```
src/coderefine/
├── prompts.py          Prompt template + output parsing — the single source of truth
├── config.py           Typed config, recursive `extends:`, run fingerprinting
├── runtime.py          Device/dtype/quantisation resolution, model loading
├── tracking.py         W&B → MLflow → local JSONL, degrading gracefully
├── train.py            PEFT + TRL SFT, completion-only loss, early stopping
├── evaluate.py         Batched greedy generation + scoring
├── metrics.py          Task metrics, copy-baseline floors, per-slice breakdowns
├── compare.py          Paired bootstrap CIs, exact McNemar, wins/regressions
├── judge.py            Blind pairwise LLM-as-judge with position-bias control
├── forgetting.py       Log-likelihood MCQ + instruction-following probes
├── export.py           Adapter packaging, checksums, vLLM/Ollama recipes
├── serve.py            FastAPI: /refine, /ab, OpenAI-compatible, HTML console
├── report.py           Assembles the experiment report from artifacts
└── data/
    ├── curate.py       Streaming filter funnel + grounding score
    ├── build.py        Leak-free splits + dataset card
    ├── benchmark.py    Hard-benchmark assembly
    └── taxonomy.py     14-category review-comment classifier

configs/                base + QLoRA + CPU configs, 8 sweep arms
notebooks/              Colab GPU runner
scripts/                sweep + full-evaluation orchestration
docker/                 train and serve images, compose file
tests/                  94 tests
data/benchmark/         hand-written benchmark half (tracked as source)
```

---

## Serving

```bash
coderefine serve --adapter artifacts/runs/<run>/adapter --load-in-4bit
```

| Endpoint | Purpose |
|---|---|
| `POST /ab` | **Both** models on one prompt, with automatic scores and a verdict |
| `POST /refine` | The task, typed |
| `POST /v1/chat/completions` | OpenAI-compatible; model name selects the variant |
| `GET /metrics` | Rolling latency p50/p95, tokens/sec |
| `GET /` | HTML console for the demo |

The A/B endpoint loads **one** base model and toggles the PEFT adapter with
`disable_adapter()`, so the two columns provably share identical weights apart
from the LoRA delta. `coderefine export` also emits a `serve_vllm.sh` (runtime
LoRA modules, batched inference) and an Ollama `Modelfile`.

---

## Reproducibility

A run is reproducible **from its config file alone**. Each run directory holds
the resolved config, a fingerprint hash of everything that affects the result,
the full metric history, the host environment, and the adapter. Renaming a run
does not change its fingerprint; changing its learning rate does.

```bash
coderefine train configs/sweep/s03_r32_lr2e4_e3.yaml
coderefine train configs/base.yaml --set lora.r=32 train.num_epochs=1
```

---

## Notes and limitations

- **Reported honestly**: at n=40 the benchmark cannot resolve small differences.
  The report prints confidence intervals and refuses to call a straddling
  interval an improvement.
- **The taxonomy is rule-based.** ~40% of comments land in `other` or
  `design_question`. Deterministic and auditable, but not exhaustive.
- **Gold is one author's revision**, not the only correct answer. This is why
  exact match is reported alongside edit-line F1 and the LLM-as-judge scores,
  which tolerate valid alternatives.
- **The bundled forgetting probes are small** (24 MCQ + 16 instruction-following)
  and original to this project. They catch task-format collapse reliably; for
  broader coverage install `lm-eval` and run the standard suites.

## Data

[Microsoft CodeReviewer](https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer)
refinement dumps (`ref-{train,valid,test}.jsonl`) — real review comments from
public pull requests across Go, Python, Java, C++, JavaScript, PHP, C#, C and
Ruby. Place them in `Code_Refinement/` and run `make data`.
# lora-code-refinement
# lora-code-refinement-v1
# loRA-code-refinement
