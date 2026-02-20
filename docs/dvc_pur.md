# ClusterLLM-local — DVC CLI Cheat Sheet (Data vs Models, 2-machine workflow)

This repo is organized around two separate concerns:

- **Data pipeline (HF → JSONL/CSV):** produces versioned dataset artifacts in stable paths.
- **Model pipeline (2 LLMs):** consumes the prepared dataset artifacts and writes model outputs.

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

```

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

*(Note: Ensure your parameters like `run.seed` are set in `params.yaml` before running these reproduction commands.)*

### A1) Prepare ONE dataset (small)

```bash
dvc repro prepare_small@mtop_intent

```

### A2) Prepare ONE dataset (large)

```bash
dvc repro prepare_large@mtop_intent

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

### B1) Reproduce full pipeline

First, ensure your `params.yaml` contains the desired configuration (e.g., `run.dataset=mtop_intent`, `run.scale=small`, `run.llm_triplet="deepseek-r1:32b"`, `run.llm_pairs="qwen2.5:32b"`, `run.seed=42`).

```bash
dvc repro full_pipeline_2llms

```

### B2) Reproduce full pipeline with auto-pull if data is missing

Use this when the dataset artifacts might not be present locally.

```bash
dvc repro --pull full_pipeline_2llms

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

## D) Bypassing the "Git stash apply merge conflicts" (The clean fix)

Because `dvc exp run` uses **git stash** under the hood, it causes stash re-apply conflicts (often on `dvc.lock` or internal `.dvc/*` files) if your workspace is not clean.

By switching entirely to **`dvc repro`**, we skip the DVC experimentation stash mechanism.

### D1) The Golden Rule

To ensure stash conflicts never reproduce again:

1. Edit your parameter values directly in `params.yaml` (do not use `-S` CLI overrides).
2. Save the file.
3. Run standard reproduction:

```bash
dvc repro prepare_large@mtop_intent

```

---

## E) Two-machine workflow (recommended practice)

### Machine A (prepare + version + push)

1. Set `run.seed: 42` in `params.yaml`, then build data with DVC:

```bash
dvc repro prepare_large@mtop_intent

```

2. Commit metadata and push artifacts:

```bash
git add dvc.lock dvc.yaml params.yaml
git commit -m "Prepare mtop_intent large (seed=42)"
dvc push
git push

```

### Machine B (pull the exact same data + reproduce models)

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

3. Update `params.yaml` for your target models, then reproduce the model pipeline (consumes the pulled JSONL):

```bash
dvc repro --pull full_pipeline_2llms

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