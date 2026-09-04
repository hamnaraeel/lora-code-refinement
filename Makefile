# Convenience targets. Every one is a thin wrapper over the `coderefine` CLI,
# so nothing here is a second way of doing something.

PY          ?= python
BASE_MODEL  ?= mistralai/Mistral-7B-Instruct-v0.3
ADAPTER     ?= artifacts/runs/qlora-mistral7b-r16/adapter
FOURBIT     ?= --load-in-4bit

.PHONY: help setup data benchmark test lint train train-cpu sweep eval-base eval-tuned \
        compare judge forgetting report serve export clean-artifacts smoke

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:            ## Install the package and its dependencies
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

data:             ## Curate raw diffs into leak-free splits
	coderefine build-data

benchmark:        ## Assemble the 40-example hard benchmark
	coderefine build-benchmark

test:             ## Run the test suite
	$(PY) -m pytest tests/ -q

smoke:            ## End-to-end CPU smoke run (small model, minutes)
	coderefine train configs/local_cpu_smoke.yaml

train:            ## Train the primary QLoRA configuration (needs CUDA)
	coderefine train configs/qlora_mistral7b.yaml

train-cpu:        ## Full CPU run on the small code model (slow but real)
	coderefine train configs/local_cpu_full.yaml

sweep:            ## Run all eight hyperparameter sweep arms
	bash scripts/run_sweep.sh

eval-base:        ## Score the base model on the benchmark
	coderefine evaluate --split benchmark --base-model $(BASE_MODEL) --tag base $(FOURBIT)

eval-tuned:       ## Score the fine-tuned model on the benchmark
	coderefine evaluate --split benchmark --base-model $(BASE_MODEL) --adapter $(ADAPTER) --tag tuned $(FOURBIT)

compare:          ## Head-to-head comparison with confidence intervals
	coderefine compare artifacts/eval/base__benchmark.predictions.jsonl \
	                   artifacts/eval/tuned__benchmark.predictions.jsonl

judge:            ## Blind pairwise LLM-as-judge scoring
	coderefine judge artifacts/eval/base__benchmark.predictions.jsonl \
	                 artifacts/eval/tuned__benchmark.predictions.jsonl

forgetting:       ## Probe general capability, base and tuned
	coderefine forgetting --base-model $(BASE_MODEL) $(FOURBIT)
	coderefine forgetting --base-model $(BASE_MODEL) --adapter $(ADAPTER) $(FOURBIT)

report:           ## Rebuild the experiment report from artifacts
	coderefine report

serve:            ## Run the A/B inference server on :8000
	coderefine serve --base-model $(BASE_MODEL) --adapter $(ADAPTER) $(FOURBIT)

export:           ## Package the adapter for deployment
	coderefine export $(ADAPTER) --out-dir artifacts/release --base-model $(BASE_MODEL)

all: data benchmark train eval-base eval-tuned compare forgetting report  ## Full pipeline

clean-artifacts:  ## Delete generated artifacts (keeps curated data)
	rm -rf artifacts/runs artifacts/eval artifacts/forgetting artifacts/release
