"""FastAPI inference server with a built-in A/B comparison endpoint.

One base model is resident in memory. PEFT adapters can be enabled and disabled
in place, so the base and fine-tuned responses come from *the same weights plus
or minus the adapter* — no second model, no second GPU, and no chance of the
comparison being confounded by a differently-loaded base.

Endpoints
---------
``POST /v1/chat/completions``  OpenAI-compatible, so existing clients work.
``POST /refine``               The task, typed: code + comment in, revision out.
``POST /ab``                   Both models on one prompt, plus automatic scores.
``GET  /healthz``              Liveness and what is loaded.
``GET  /metrics``              Rolling latency and throughput.
``GET  /``                     A small HTML console for the demo.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .metrics import percentile, score_example
from .prompts import build_messages, parse_generation
from .quant import QuantSpec
from .runtime import load_model

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RefineRequest(BaseModel):
    old_code: str = Field(..., description="The code as it stands before review.")
    comment: str = Field(..., description="The reviewer's comment.")
    lang: str = Field("py", description="Language tag: py, java, go, js, php, rb, c, cpp, .cs")
    variant: Literal["tuned", "base"] = "tuned"
    max_new_tokens: int = 512


class ABRequest(BaseModel):
    old_code: str
    comment: str
    lang: str = "py"
    #: Optional gold revision. When supplied, both outputs are scored against it
    #: and the response carries the automatic metrics alongside the text.
    gold: str | None = None
    max_new_tokens: int = 512


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "coderefine"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Engine:
    """Holds the resident model and toggles the adapter around each request."""

    def __init__(self, base_model: str, adapter: Path | None, device: str, load_in_4bit: bool):
        self.base_model_name = base_model
        self.adapter_path = str(adapter) if adapter else None
        loaded = load_model(
            base_model=base_model,
            device=device,
            quant=QuantSpec(load_in_4bit=load_in_4bit),
            adapter=adapter,
        )
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self.device = loaded.device
        self.has_adapter = adapter is not None
        self.latencies: deque[float] = deque(maxlen=500)
        self.token_counts: deque[int] = deque(maxlen=500)

    @contextmanager
    def as_variant(self, variant: str):
        """Run a block with the adapter on ("tuned") or off ("base").

        ``disable_adapter()`` is a PEFT context manager that zeroes the adapter
        contribution without unloading anything, so switching costs nothing and
        both variants demonstrably share one set of base weights.
        """
        if variant == "base" and self.has_adapter:
            with self.model.disable_adapter():
                yield
        else:
            if variant == "tuned" and not self.has_adapter:
                raise HTTPException(
                    status_code=400,
                    detail="No adapter is loaded; start the server with --adapter to serve the tuned variant.",
                )
            yield

    @torch.no_grad()
    def generate(self, messages: list[dict[str, str]], max_new_tokens: int, variant: str) -> tuple[str, dict]:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

        start = time.time()
        with self.as_variant(variant):
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - start

        prompt_len = encoded["input_ids"].shape[1]
        new_tokens = int(output.shape[1] - prompt_len)
        text = self.tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)

        self.latencies.append(elapsed)
        self.token_counts.append(new_tokens)
        usage = {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": new_tokens,
            "total_tokens": int(prompt_len) + new_tokens,
            "latency_s": round(elapsed, 3),
            "tokens_per_second": round(new_tokens / elapsed, 2) if elapsed > 0 else None,
        }
        return text, usage


def build_app(
    base_model: str,
    adapter: Path | None = None,
    device: str = "auto",
    load_in_4bit: bool = False,
) -> FastAPI:
    engine = Engine(base_model, adapter, device, load_in_4bit)
    app = FastAPI(
        title="Code Refinement A/B Server",
        description="Base vs LoRA-tuned code refinement from one resident model.",
        version="0.1.0",
    )

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "base_model": engine.base_model_name,
            "adapter": engine.adapter_path,
            "adapter_loaded": engine.has_adapter,
            "device": engine.device,
            "variants": ["tuned", "base"] if engine.has_adapter else ["base"],
        }

    @app.get("/metrics")
    def metrics() -> dict:
        lat = sorted(engine.latencies)
        if not lat:
            return {"requests": 0}
        total_tokens = sum(engine.token_counts)
        total_time = sum(engine.latencies)
        return {
            "requests": len(lat),
            "latency_p50_s": round(percentile(lat, 0.50), 3),
            "latency_p95_s": round(percentile(lat, 0.95), 3),
            "latency_max_s": round(lat[-1], 3),
            "mean_tokens_per_second": round(total_tokens / total_time, 2) if total_time else None,
            "completion_tokens_total": total_tokens,
        }

    @app.post("/refine")
    def refine(req: RefineRequest) -> dict:
        messages = build_messages(req.old_code, req.comment, req.lang)
        raw, usage = engine.generate(messages, req.max_new_tokens, req.variant)
        parsed = parse_generation(raw)
        return {
            "variant": req.variant,
            "revised_code": parsed.code,
            "parse_mode": parsed.how,
            "raw_output": raw,
            "usage": usage,
        }

    @app.post("/ab")
    def ab(req: ABRequest) -> dict:
        """Run both variants on one prompt. This is the demo endpoint."""
        if not engine.has_adapter:
            raise HTTPException(
                status_code=400,
                detail="A/B comparison needs an adapter; start the server with --adapter.",
            )
        messages = build_messages(req.old_code, req.comment, req.lang)

        results: dict[str, Any] = {}
        for variant in ("base", "tuned"):
            raw, usage = engine.generate(messages, req.max_new_tokens, variant)
            parsed = parse_generation(raw)
            entry: dict[str, Any] = {
                "revised_code": parsed.code,
                "parse_mode": parsed.how,
                "changed_input": parsed.code.strip() != req.old_code.strip(),
                "usage": usage,
            }
            if req.gold:
                score = score_example(parsed.code, req.gold, req.old_code, req.lang)
                entry["scores"] = {
                    "exact_match": score.exact_match,
                    "edit_similarity": round(score.edit_sim, 4),
                    "moved_toward_gold": score.improved,
                    "edit_line_f1": round(score.changed_right_lines, 4),
                }
            results[variant] = entry

        verdict = None
        if req.gold:
            b, t = results["base"]["scores"], results["tuned"]["scores"]
            if t["exact_match"] and not b["exact_match"]:
                verdict = "fine-tuned model matched the merged revision exactly; base model did not"
            elif b["exact_match"] and not t["exact_match"]:
                verdict = "base model matched exactly; fine-tuned model regressed"
            elif t["edit_similarity"] > b["edit_similarity"]:
                verdict = "fine-tuned model is closer to the merged revision"
            elif t["edit_similarity"] < b["edit_similarity"]:
                verdict = "base model is closer to the merged revision"
            else:
                verdict = "both models scored identically"

        return {
            "request": {"lang": req.lang, "comment": req.comment, "old_code": req.old_code},
            "base": results["base"],
            "tuned": results["tuned"],
            "verdict": verdict,
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest) -> dict:
        """OpenAI-compatible surface.

        The requested model name selects the variant: anything containing
        "base" runs with the adapter disabled, everything else runs tuned. That
        mirrors how vLLM addresses runtime LoRA modules by name, so a client
        written against this server works unchanged against `serve_vllm.sh`.
        """
        variant = "base" if "base" in req.model.lower() else "tuned"
        messages = [m.model_dump() for m in req.messages]
        raw, usage = engine.generate(messages, req.max_tokens, variant)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": raw},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            },
        }

    @app.get("/", response_class=HTMLResponse)
    def console() -> str:
        return _CONSOLE_HTML

    return app


_CONSOLE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Code Refinement A/B</title>
<style>
 :root{color-scheme:light dark}
 body{font:14px/1.5 ui-sans-serif,system-ui,sans-serif;margin:0;padding:24px;max-width:1100px}
 h1{font-size:18px;margin:0 0 4px} p.sub{margin:0 0 20px;opacity:.7}
 label{display:block;font-weight:600;margin:14px 0 4px}
 textarea,input,select{width:100%;font:13px ui-monospace,SFMono-Regular,Menlo,monospace;
   padding:8px;border:1px solid #8884;border-radius:6px;background:transparent;color:inherit;box-sizing:border-box}
 textarea{min-height:120px;resize:vertical}
 button{margin-top:14px;padding:9px 18px;border-radius:6px;border:1px solid #8886;
   background:#2563eb;color:#fff;font-weight:600;cursor:pointer}
 button:disabled{opacity:.5;cursor:wait}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:20px}
 @media(max-width:800px){.cols{grid-template-columns:1fr}}
 .card{border:1px solid #8884;border-radius:8px;padding:12px;overflow-x:auto}
 .card h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;margin:0 0 8px;opacity:.7}
 pre{margin:0;font:12px/1.45 ui-monospace,Menlo,monospace;white-space:pre}
 .badge{display:inline-block;font-size:11px;padding:2px 7px;border-radius:99px;border:1px solid #8886;margin-right:6px}
 .verdict{margin-top:16px;padding:10px 12px;border-left:3px solid #2563eb;background:#2563eb14;border-radius:0 6px 6px 0}
</style></head><body>
<h1>Code Refinement — base vs LoRA-tuned</h1>
<p class="sub">One resident base model. The adapter is toggled per request, so both columns share identical weights apart from the LoRA delta.</p>
<label>Language</label>
<select id="lang"><option>py</option><option>java</option><option>go</option><option>js</option>
<option>php</option><option>rb</option><option>c</option><option>cpp</option><option>.cs</option></select>
<label>Code under review</label>
<textarea id="old">def load_config(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None</textarea>
<label>Reviewer comment</label>
<textarea id="comment" style="min-height:60px">This except block swallows the error and returns None, which hides real failures. Just let it propagate.</textarea>
<label>Reference revision (optional — enables automatic scoring)</label>
<textarea id="gold" style="min-height:60px"></textarea>
<button id="go" onclick="run()">Compare</button>
<div class="cols">
  <div class="card"><h2>Base model</h2><div id="bmeta"></div><pre id="base">—</pre></div>
  <div class="card"><h2>Fine-tuned</h2><div id="tmeta"></div><pre id="tuned">—</pre></div>
</div>
<div id="verdict"></div>
<script>
async function run(){
  const btn=document.getElementById('go'); btn.disabled=true; btn.textContent='Running…';
  document.getElementById('verdict').innerHTML='';
  const gold=document.getElementById('gold').value.trim();
  const body={old_code:document.getElementById('old').value,
              comment:document.getElementById('comment').value,
              lang:document.getElementById('lang').value};
  if(gold) body.gold=gold;
  try{
    const r=await fetch('/ab',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!r.ok) throw new Error((await r.json()).detail||r.statusText);
    const d=await r.json();
    for(const [k,elId,metaId] of [['base','base','bmeta'],['tuned','tuned','tmeta']]){
      document.getElementById(elId).textContent=d[k].revised_code||'(empty)';
      const u=d[k].usage, s=d[k].scores;
      let m=`<span class="badge">${u.latency_s}s</span><span class="badge">${u.tokens_per_second??'–'} tok/s</span>`;
      m+=`<span class="badge">${d[k].changed_input?'edited':'unchanged'}</span>`;
      if(s) m+=`<span class="badge">${s.exact_match?'exact match':'edit sim '+s.edit_similarity}</span>`;
      document.getElementById(metaId).innerHTML=m;
    }
    if(d.verdict) document.getElementById('verdict').innerHTML='<div class="verdict"><b>Verdict:</b> '+d.verdict+'</div>';
  }catch(e){ document.getElementById('verdict').innerHTML='<div class="verdict">Error: '+e.message+'</div>'; }
  btn.disabled=false; btn.textContent='Compare';
}
</script></body></html>"""
