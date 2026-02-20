````markdown
# ClusterLLM-local — DVC CLI Cheat Sheet (Data vs Models, 2-machine workflow)

This repo is organized around two separate concerns:

- **Data pipeline (HF → JSONL/CSV):** produces versioned dataset artifacts in stable paths.
- **Model pipeline (2 LLMs):** consumes the prepared dataset artifacts and writes model/run outputs.

The goal is: you run **only DVC commands** (no manual bash), and `dvc pull` restores the exact same files in the exact same locations.

---

## 0) Preconditions (one-time)

### Required stages
Your `dvc.yaml` should expose these stages:
- `prepare_small` (foreach → `prepare_small@<dataset>`)
- `prepare_large` (foreach → `prepare_large@<dataset>`)
- `full_pipeline_2llms`

List available stages:
```bash
dvc stage list
````

### Required params

Your `params.yaml` should contain:

* `run.dataset`, `run.scale`, `run.seed`, `run.llm_triplet`, `run.llm_pairs`
* `datasets.all: [ ... ]`

### Important Git rule (data must NOT be tracked by Git)

Generated data files must be tracked by **DVC**, not Git:

* `src/clusterllm/datasets/*/*.jsonl`
* `src/clusterllm/datasets/*/*_train.jsonl`
* `src/clusterllm/datasets/*/*_eval.jsonl`
* `data/raw/*_train.csv`
* `data/raw/*_eval.csv`

If you ever see: “output is already tracked by SCM (e.g. Git)”, fix it once:

```bash
git rm -r --cached src/clusterllm/datasets/*/*.jsonl
git rm -r --cached src/clusterllm/datasets/*/*_train.jsonl src/clusterllm/datasets/*/*_eval.jsonl
git rm -r --cached data/raw/*_train.csv data/raw/*_eval.csv
git add .gitignore
git commit -m "Stop tracking generated data in Git; track via DVC"
```

---

## A) Data pipeline (HF → JSONL/CSV)

### A1) Prepare ONE dataset (small)

Preferred (safe): stage target is positional; override only what you need.

```bash
dvc exp run prepare_small@mtop_intent -S run.seed=42
```

### A2) Prepare ONE dataset (large)

```bash
dvc exp run prepare_large@mtop_intent -S run.seed=42
```

### A3) Prepare ALL datasets (small)

```bash
dvc repro prepare_small
```

### A4) Prepare ALL datasets (large)

```bash
dvc repro prepare_large
```

Expected outputs (stable paths):

* `src/clusterllm/datasets/<ds>/small.jsonl` or `src/clusterllm/datasets/<ds>/large.jsonl`
* `data/raw/<ds>_<scale>_train.csv`
* `data/raw/<ds>_<scale>_eval.csv`

---

## B) Model pipeline (Full pipeline: 1 dataset, 2 LLMs)

### B1) Run full pipeline (explicit params on CLI)

```bash
dvc exp run full_pipeline_2llms \
  -S run.dataset=mtop_intent \
  -S run.scale=small \
  -S run.llm_triplet="deepseek-r1:32b" \
  -S run.llm_pairs="qwen2.5:32b" \
  -S run.seed=42
```

### B2) Run full pipeline with auto-pull if data is missing

Use this when the dataset artifacts might not be present locally.

```bash
dvc exp run --pull full_pipeline_2llms \
  -S run.dataset=mtop_intent \
  -S run.scale=small \
  -S run.llm_triplet="deepseek-r1:32b" \
  -S run.llm_pairs="qwen2.5:32b" \
  -S run.seed=42
```

Stable metric output:

* `runs/summary/full_pipeline_2llms/<dataset>/<scale>/summary.json`

---

## C) Sync to remote (push/pull)

Keep sync commands short and standard:

### C1) Push everything tracked by DVC

```bash
dvc push
```

### C2) Pull everything tracked by DVC

```bash
dvc pull
```

### C3) Pull only one dataset file (fast)

```bash
dvc pull src/clusterllm/datasets/mtop_intent/large.jsonl
```

After `dvc pull`, the model pipeline will find:
`src/clusterllm/datasets/<ds>/<small|large>.jsonl` exactly where it expects it.

---

## D) “Git stash apply merge conflicts” (what it means and the clean fix)

`dvc exp run` uses **git stash** under the hood. If your workspace is not clean (modified/untracked files),
stash re-apply can conflict (often on `dvc.lock` or internal `.dvc/*` files).

### D1) Make the workspace clean (recommended)

```bash
rm -f .dvc/dvc.md  # remove untracked file if it exists
git add .dvc/.gitignore dvc.lock
git commit -m "DVC bookkeeping (.dvc/.gitignore, dvc.lock)"
```

Then rerun:

```bash
dvc exp run prepare_large@mtop_intent -S run.seed=42
```

### D2) Avoid stash entirely (use a temp workspace)

```bash
dvc exp run --temp prepare_large@mtop_intent -S run.seed=42
```

### D3) If you do not need CLI overrides, prefer `dvc repro`

When `run.seed` is already set in `params.yaml`:

```bash
dvc repro prepare_large@mtop_intent
```

---

## E) Two-machine workflow (recommended practice)

### Machine A (prepare + version + push)

1. Build data with DVC:

```bash
dvc exp run prepare_large@mtop_intent -S run.seed=42
```

2. Commit metadata and push artifacts:

```bash
git add dvc.lock dvc.yaml params.yaml
git commit -m "Prepare mtop_intent large (seed=42)"
dvc push
git push
```

### Machine B (pull the exact same data + run models)

1. Get code + DVC metadata:

```bash
git pull
```

2. Restore artifacts (either all or just what you need):

```bash
dvc pull
# or minimal:
dvc pull src/clusterllm/datasets/mtop_intent/large.jsonl
```

3. Run the model pipeline (consumes the pulled JSONL):

```bash
dvc exp run --pull full_pipeline_2llms \
  -S run.dataset=mtop_intent \
  -S run.scale=large \
  -S run.llm_triplet="deepseek-r1:32b" \
  -S run.llm_pairs="qwen2.5:32b" \
  -S run.seed=42
```

---

## F) Quick diagnostics

Check what will be reproduced:

```bash
dvc status
```

Visualize dependencies:

```bash
dvc dag
```

List tracked artifacts:

```bash
dvc list . --dvc-only
```

```
```
