# 🚀 DVC Sweep Guide – Full Pipeline (2 LLMs)

This guide explains how to:

* Queue 100 scenarios
* Execute them in parallel
* Reproduce a single scenario
* View the results
* Clean the queue
* Reproduce a specific scenario among all

---

# 📂 Prerequisites

Expected structure:

```text
scripts/
  repro_full_pipeline_one_dataset_two_llms.sh
  sweep_full_pipeline_2llms.sh

configs/
  grid_small.txt
  grid_large.txt

```

Expected DVC stage in `dvc.yaml`:

```text
full_pipeline_2llms

```

---

# 🧩 Grid file format

`configs/grid_small.txt`

Each line = 1 scenario:

```text
# dataset scale llm_triplet llm_pairs seed
clinc150 small deepseek-r1:32b qwen2.5:32b 42
bank77 small llama3.2:3b-instruct-q8_0 qwen2.5:7b 42
reddit small deepseek-r1:32b llama3.1:8b-instruct-q8_0 42

```

* 5 mandatory columns
* Empty lines are ignored
* Lines starting with `#` are ignored

---

# 🔁 Main commands

---

## A) Queue all 100 scenarios (without reproducing)

```bash
bash scripts/sweep_full_pipeline_2llms.sh queue configs/grid_100.txt

```

This:

* Adds all scenarios to your script's queue
* Does not reproduce anything yet

---

## B) Reproduce all scenarios in parallel (4 jobs)

```bash
bash scripts/sweep_full_pipeline_2llms.sh repro configs/grid_small.txt 4

```

This:

1. Queues all scenarios
2. Reproduces all pipelines in parallel using 4 workers

---

* This also applies to `configs/grid_large.txt`

## C) Reproduce a single scenario "by hand"

*(Note: Since `dvc repro` does not accept `-S` for inline overrides, update your `params.yaml` file with the desired values first.)*

```bash
dvc repro full_pipeline_2llms

```

---

## D) View the results

```bash
bash scripts/sweep_full_pipeline_2llms.sh show

```

Equivalent to (if your script registers experiments):

```bash
dvc exp show

```

---

## E) Clean the queue

```bash
bash scripts/sweep_full_pipeline_2llms.sh clean

```

---

# 🎯 Reproduce a single scenario out of the 100

---

## Option 1 – Create a temporary file

Create:

```text
configs/one_small.txt

```

Content:

```text
clinc150 small deepseek-r1:32b qwen2.5:32b 42

```

Then:

```bash
bash scripts/sweep_full_pipeline_2llms.sh repro configs/one_small.txt 4

```

* Same for `configs/one_large.txt`

---

## Option 2 – Temporarily modify grid_small.txt

Delete all lines except the one you want, then:

```bash
bash scripts/sweep_full_pipeline_2llms.sh repro configs/grid_small.txt 4

```

* Same for `configs/one_large.txt`

---

# 🧠 Recommended Workflow

### 1️⃣ Generate or modify grid

```text
configs/grid_small.txt
- or configs/grid_large.txt

```

### 2️⃣ Queue

```bash
bash scripts/sweep_full_pipeline_2llms.sh queue configs/grid_small.txt
# or 
bash scripts/sweep_full_pipeline_2llms.sh queue configs/grid_large.txt

```

### 3️⃣ Reproduce in parallel

```bash
bash scripts/sweep_full_pipeline_2llms.sh repro configs/grid_small.txt 4

```

*(Using your custom script to handle the parallel jobs instead of standard DVC experiment queues)*

### 4️⃣ View results

```bash
dvc exp show

```

---

# ⚡ Real-world Example

```bash
bash scripts/sweep_full_pipeline_2llms.sh repro configs/grid_small.txt 4
# or 
bash scripts/sweep_full_pipeline_2llms.sh repro configs/grid_large.txt 4

```

This:

* Enqueues 100 configs
* Reproduces 4 in parallel
* Continues until finished
* All outputs are in `runs/`

---

# 📌 Summary

| Action | Command |
| --- | --- |
| Queue | `sweep_full_pipeline_2llms.sh queue` |
| Parallel repro | `sweep_full_pipeline_2llms.sh repro` |
| Single repro | Update `params.yaml` then `dvc repro ...` |
| Show results | `dvc exp show` |
| Clean queue | `sweep_full_pipeline_2llms.sh clean` |

---

# ✅ Best Practices

* Always verify that the script correctly accepts the arguments
* Do not overwrite variables after the args block
* Use `--jobs` (or script worker count) adapted to your VRAM
* Do not put `runs/` in the DVC cache if it is too large

---

