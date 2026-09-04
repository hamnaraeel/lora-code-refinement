#!/usr/bin/env python
"""Side-by-side terminal demo: base model vs LoRA-tuned, on hand-picked cases.

Built for recording. It loads one base model, toggles the adapter per example,
and prints a coloured three-way diff against the merged revision — which is the
whole story of the project in one screen.

    python scripts/demo.py --adapter artifacts/runs/<run>/adapter --load-in-4bit
    python scripts/demo.py --adapter ... --ids hw-02,hw-06,hw-14   # pick cases
    python scripts/demo.py --adapter ... --pause                   # step through
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from coderefine.metrics import normalize, score_example
from coderefine.prompts import build_messages, parse_generation
from coderefine.quant import QuantSpec
from coderefine.runtime import load_model

console = Console()

#: Default showcase. Chosen because each is a *different* kind of failure, and
#: because base models reliably get them wrong in an interesting way rather than
#: an unwatchable one.
DEFAULT_IDS = ["hw-02", "hw-06", "hw-01", "hw-14", "hw-11"]

_FENCE_LANG = {"py": "python", "js": "javascript", "rb": "ruby", ".cs": "csharp"}


def _syntax(code: str, lang: str) -> Syntax:
    return Syntax(
        code or "(empty)", _FENCE_LANG.get(lang, lang or "text"),
        theme="ansi_dark", word_wrap=False, background_color="default",
    )


def _verdict(base_s, tuned_s) -> tuple[str, str]:
    if tuned_s.exact_match and not base_s.exact_match:
        return "FINE-TUNED WINS", "bold green"
    if base_s.exact_match and not tuned_s.exact_match:
        return "BASE WINS", "bold red"
    if tuned_s.edit_sim > base_s.edit_sim + 0.02:
        return "fine-tuned closer", "green"
    if base_s.edit_sim > tuned_s.edit_sim + 0.02:
        return "base closer", "red"
    return "tie", "yellow"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--benchmark", type=Path, default=Path("data/benchmark/benchmark.jsonl"))
    ap.add_argument("--ids", default="", help="Comma-separated example ids.")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--pause", action="store_true", help="Wait for a keypress between examples.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.benchmark.read_text().splitlines() if l.strip()]
    by_id = {r["id"]: r for r in rows}
    wanted = [i.strip() for i in args.ids.split(",") if i.strip()] or DEFAULT_IDS
    selected = [by_id[i] for i in wanted if i in by_id][: args.limit]
    if not selected:
        selected = rows[: args.limit]

    console.rule("[bold]Loading model[/]")
    console.print(f"base: [cyan]{args.base_model}[/]   adapter: [cyan]{args.adapter}[/]")
    loaded = load_model(
        base_model=args.base_model, device=args.device,
        quant=QuantSpec(load_in_4bit=args.load_in_4bit), adapter=args.adapter,
    )
    model, tok = loaded.model, loaded.tokenizer
    console.print(f"device: [green]{loaded.device}[/]  dtype: [green]{loaded.dtype}[/]\n")

    tally = {"tuned": 0, "base": 0, "tie": 0}

    for n, rec in enumerate(selected, 1):
        console.rule(f"[bold]{n}/{len(selected)}  {rec['id']}  ·  {rec['lang']}  ·  {rec['category']}[/]")
        if rec.get("probe"):
            console.print(f"[dim]failure mode probed: {rec['probe']}[/]")
        console.print(Panel(rec["comment"], title="reviewer comment", border_style="yellow"))
        console.print(Panel(_syntax(rec["old_code"], rec["lang"]), title="code under review", border_style="dim"))

        messages = build_messages(rec["old_code"], rec["comment"], rec["lang"])
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tok(prompt, return_tensors="pt", add_special_tokens=False).to(loaded.device)

        outputs = {}
        for variant in ("base", "tuned"):
            ctx = model.disable_adapter() if variant == "base" else _null()
            with ctx:
                generated = model.generate(
                    **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                    num_beams=1, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                )
            text = tok.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            outputs[variant] = parse_generation(text).code

        base_s = score_example(outputs["base"], rec["new_code"], rec["old_code"], rec["lang"])
        tuned_s = score_example(outputs["tuned"], rec["new_code"], rec["old_code"], rec["lang"])
        label, style = _verdict(base_s, tuned_s)
        tally["tuned" if "FINE-TUNED" in label or label == "fine-tuned closer"
              else "base" if "BASE" in label or label == "base closer" else "tie"] += 1

        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(ratio=1); grid.add_column(ratio=1); grid.add_column(ratio=1)
        grid.add_row(
            Panel(_syntax(rec["new_code"], rec["lang"]), title="[bold]merged revision (gold)", border_style="green"),
            Panel(_syntax(outputs["base"], rec["lang"]),
                  title=f"base  ·  {'EXACT' if base_s.exact_match else f'sim {base_s.edit_sim:.2f}'}"
                        f"{'  ·  UNCHANGED' if base_s.copied else ''}",
                  border_style="grey50"),
            Panel(_syntax(outputs["tuned"], rec["lang"]),
                  title=f"fine-tuned  ·  {'EXACT' if tuned_s.exact_match else f'sim {tuned_s.edit_sim:.2f}'}"
                        f"{'  ·  UNCHANGED' if tuned_s.copied else ''}",
                  border_style="blue"),
        )
        console.print(grid)
        console.print(f"[{style}]▸ {label}[/]\n")

        if args.pause and n < len(selected):
            console.input("[dim]press enter for the next example…[/]")

    console.rule("[bold]Summary[/]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("outcome"); table.add_column("count", justify="right")
    for key, label in [("tuned", "fine-tuned better"), ("base", "base better"), ("tie", "tie")]:
        table.add_row(label, str(tally[key]))
    console.print(table)
    console.print(
        "[dim]This is a hand-picked demo of failure modes, not a benchmark. "
        "The measured results are in reports/EXPERIMENT_REPORT.md.[/]"
    )


class _null:
    """No-op context manager, so the two branches read symmetrically."""

    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
