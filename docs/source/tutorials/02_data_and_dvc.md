# Data Management and DVC Pipelines

ClusterLLM-Local treats data and model artifacts as first-class, versioned objects using Data Version Control (DVC).

## Deterministic Dataset Reconstruction

Class imbalance and row ordering significantly impact clustering outcomes. We use a deterministic reconstruction layer that ensures:

* Global seed propagation (default: 42).


* Stable sorting of examples before any sampling.


* Explicit tracking of class imbalance ratios ($r$).



## DVC Stage Organization

The pipeline is divided into two distinct stages to prevent accidental coupling:

1. **Data Preparation**: Generation of deterministic `.jsonl` files.
2. **Model Supervision**: Extraction of triplet/pair predictions and training files.

## Running Data Pipelines

You can reproduce the data preparation for various scales (Small or Large):

### Small Split

```bash
# Individual dataset
poetry run dvc repro prepare_small@bank77

# All datasets
poetry run dvc repro prepare_small

```

### Remote Storage

Data is versioned and stored on a Google Drive remote. To synchronize:

```bash
dvc pull  # Retrieve existing artifacts
dvc push  # Upload new artifacts

```

---
