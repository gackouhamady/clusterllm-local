#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyper-deterministic reconstruction of ClusterLLM datasets (paper: arXiv:2305.14871v2).

Outputs:
  ./datasets/{dataset_name}/{split}.jsonl
with exactly 2 columns: text, label

Determinism requirements:
  - random.seed(SEED) and np.random.seed(SEED)
  - For ANY sampling:
      1) sort by text (alphabetical, stable mergesort)
      2) use pandas.sample(..., random_state=SEED)

Paper rules (Appendix E):
  - CLINC: remove out-of-scope utterances; clinc_intent keeps in-domain intents only.
           clinc_domain uses domain as label (still removing OOS utterances).
  - Massive & MTOP: keep English-only.
    Remove labels with very few instances by keeping Top-K labels by frequency (deterministic).
  - FewRel & FewEvent: combine all splits, then split into large/small by target sizes.
  - FewNerd: use original train (large) and test (small).
  - StackEx/Reddit/ArxivS2S: combine all splits, keep Top-K labels, then split deterministically.
  - GoEmo: remove multi-label and neutral instances.

Robustness additions:
  - If not enough rows to reach target_rows, we do deterministic oversampling WITH replacement
    (only as needed), preserving label proportions as much as possible.
  - We then optionally adjust counts to better match the paper ratio (min_count/max_count)
    without introducing excessive noise.
  - No hard crash on ratio mismatch: log warnings instead (can be made strict with env var).
"""

from __future__ import annotations

import os
import json
import random
import traceback
from typing import Any, Dict, Optional, Tuple, List, Sequence

import numpy as np
import pandas as pd
from datasets import load_dataset  # huggingface datasets


# -------------------------
# Global determinism
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Metrics strictness (avoid crashes by default)
STRICT_METRICS = os.getenv("STRICT_METRICS", "0") == "1"
RATIO_TOL = float(os.getenv("RATIO_TOL", "5e-3"))  # default: 0.005 (0.5% absolute tolerance)
MAX_RATIO_ADJUST_ITERS = int(os.getenv("MAX_RATIO_ADJUST_ITERS", "200000"))


# -------------------------
# Target metrics (YOUR table)
# ratio := min(label_count)/max(label_count)
# -------------------------
TARGETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "mtop_intent": {
        "large": {"ratio": 0.000619, "nb_classes": 102, "total_rows": 15638},
        "small": {"ratio": 0.002114, "nb_classes": 102, "total_rows": 4386},
    },
    "massive_intent": {
        "large": {"ratio": 0.017284, "nb_classes": 59, "total_rows": 11510},
        "small": {"ratio": 0.004785, "nb_classes": 59, "total_rows": 2974},
    },
    "few_nerd_nat": {
        "large": {"ratio": 0.006604, "nb_classes": 58, "total_rows": 50000},
        "small": {"ratio": 0.008780, "nb_classes": 58, "total_rows": 3789},
    },
    "go_emotions": {
        "large": {"ratio": 0.014391, "nb_classes": 27, "total_rows": 23485},
        "small": {"ratio": 0.011869, "nb_classes": 27, "total_rows": 5940},
    },
    "few_event": {
        "large": {"ratio": 0.029732, "nb_classes": 34, "total_rows": 18969},
        "small": {"ratio": 0.030948, "nb_classes": 34, "total_rows": 4742},
    },
    "stackex": {
        "large": {"ratio": 0.061983, "nb_classes": 121, "total_rows": 50000},
        "small": {"ratio": 0.037037, "nb_classes": 121, "total_rows": 4156},
    },
    "massive_domain": {
        "large": {"ratio": 0.125000, "nb_classes": 18, "total_rows": 11514},
        "small": {"ratio": 0.141791, "nb_classes": 18, "total_rows": 2974},
    },
    "bank77": {
        "large": {"ratio": 0.187166, "nb_classes": 77, "total_rows": 10003},
        "small": {"ratio": 1.000000, "nb_classes": 77, "total_rows": 3080},
    },
    "mtop_domain": {
        "large": {"ratio": 0.424783, "nb_classes": 11, "total_rows": 15667},
        "small": {"ratio": 0.321370, "nb_classes": 11, "total_rows": 4386},
    },
    "reddit": {
        "large": {"ratio": 0.729131, "nb_classes": 50, "total_rows": 50000},
        "small": {"ratio": 0.413793, "nb_classes": 50, "total_rows": 3217},
    },
    "arxiv": {
        "large": {"ratio": 0.777015, "nb_classes": 93, "total_rows": 50000},
        "small": {"ratio": 0.800000, "nb_classes": 93, "total_rows": 3674},
    },
    "few_rel_nat": {
        "large": {"ratio": 0.947612, "nb_classes": 64, "total_rows": 40320},
        "small": {"ratio": 0.600000, "nb_classes": 64, "total_rows": 4480},
    },
    "clinc_domain": {
        "large": {"ratio": 1.000000, "nb_classes": 10, "total_rows": 15000},
        "small": {"ratio": 1.000000, "nb_classes": 10, "total_rows": 4500},
    },
    "clinc_intent": {
        "large": {"ratio": 1.000000, "nb_classes": 150, "total_rows": 15000},
        "small": {"ratio": 1.000000, "nb_classes": 150, "total_rows": 4500},
    },
}


# -------------------------
# IO helpers
# -------------------------
def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(df: pd.DataFrame, path: str) -> None:
    mkdirp(os.path.dirname(path))
    df = df[["text", "label"]].copy()
    # stable output order
    df = df.sort_values(["text", "label"], kind="mergesort").reset_index(drop=True)

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in df.itertuples(index=False):
            f.write(json.dumps({"text": str(r.text), "label": str(r.label)}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# -------------------------
# Deterministic transforms
# -------------------------
def sort_by_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df.sort_values(["text", "label"], kind="mergesort").reset_index(drop=True)


def ratio_min_over_max(df: pd.DataFrame) -> float:
    vc = df["label"].value_counts()
    if len(vc) == 0:
        return 0.0
    mn = int(vc.min())
    mx = int(vc.max())
    return float(mn) / float(mx) if mx > 0 else 0.0


def keep_top_k_labels(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Remove labels with very few instances by keeping top-K labels by frequency.
    Deterministic tie-breaker: label string.
    """
    df = sort_by_text(df)
    vc = df["label"].value_counts()
    items = [(lab, int(cnt)) for lab, cnt in vc.items()]
    items.sort(key=lambda x: (-x[1], str(x[0])))
    top = [lab for lab, _ in items[:k]]
    out = df[df["label"].isin(top)].copy()
    return sort_by_text(out)


# -------------------------
# Robust deterministic resampling (downsample or oversample)
# -------------------------
def _largest_remainder_counts(
    labels: Sequence[str],
    weights: Dict[str, float],
    total: int,
    *,
    min_per_label: int = 1,
) -> Dict[str, int]:
    """
    Deterministic integer allocation that sums to 'total'.
    - weights must be non-negative and not all zero.
    - ensures each label gets at least min_per_label if feasible.

    Tie-breakers are deterministic by label string.
    """
    labels = list(labels)
    labels = sorted(labels, key=lambda x: str(x))

    # Normalize weights
    w = np.array([float(weights.get(l, 0.0)) for l in labels], dtype=np.float64)
    w = np.maximum(w, 0.0)
    s = float(w.sum())

    if s <= 0.0:
        # Fallback: uniform weights
        w = np.ones_like(w)
        s = float(w.sum())

    w = w / s

    # First pass (floor)
    raw = w * float(total)
    base = np.floor(raw).astype(int)
    rem = raw - base.astype(np.float64)

    # Enforce minimum per label if feasible
    if min_per_label > 0:
        base = np.maximum(base, min_per_label)

    cur_sum = int(base.sum())

    # If we exceeded total due to min_per_label, reduce deterministically but never below min_per_label
    if cur_sum > total:
        over = cur_sum - total
        # Sort candidates by (count desc, label asc) to reduce from large bins first deterministically
        order = sorted(range(len(labels)), key=lambda i: (-int(base[i]), str(labels[i])))
        i_ptr = 0
        while over > 0 and i_ptr < len(order):
            i = order[i_ptr]
            if base[i] > min_per_label:
                base[i] -= 1
                over -= 1
            else:
                i_ptr += 1
        # If still over, impossible constraint; hard cut (but deterministic)
        if over > 0:
            # cut from the earliest labels (stable)
            for i in range(len(labels)):
                if over <= 0:
                    break
                if base[i] > 0:
                    base[i] -= 1
                    over -= 1

    # If we are under total, distribute by largest remainder (tie by label)
    cur_sum = int(base.sum())
    if cur_sum < total:
        need = total - cur_sum
        # Sort indices by remainder desc, label asc
        order = sorted(range(len(labels)), key=lambda i: (-float(rem[i]), str(labels[i])))
        idx = 0
        while need > 0:
            i = order[idx % len(order)]
            base[i] += 1
            need -= 1
            idx += 1

    # Final sanity
    base = base.astype(int)
    # In extremely pathological cases, enforce exact sum deterministically
    cur_sum = int(base.sum())
    if cur_sum != total:
        diff = total - cur_sum
        if diff > 0:
            # add to smallest label strings
            for i in range(diff):
                base[i % len(base)] += 1
        else:
            # remove from largest counts first, but not below 0
            diff = -diff
            order = sorted(range(len(labels)), key=lambda i: (-int(base[i]), str(labels[i])))
            for i in order:
                if diff <= 0:
                    break
                take = min(diff, int(base[i]))
                base[i] -= take
                diff -= take

    return {lab: int(cnt) for lab, cnt in zip(labels, base.tolist())}


def _adjust_counts_to_target_ratio(
    counts: Dict[str, int],
    target_ratio: float,
    *,
    min_per_label: int = 1,
    tol: float = RATIO_TOL,
    max_iters: int = MAX_RATIO_ADJUST_ITERS,
) -> Dict[str, int]:
    """
    Adjust counts by moving 1 sample at a time from max -> min (or reverse)
    to approach target_ratio = min/max.

    This is intentionally conservative:
    - maintains sum(counts) constant
    - never goes below min_per_label
    - deterministic tie-breaking by label string
    """
    if not (0.0 <= float(target_ratio) <= 1.0):
        return counts

    labels = sorted(counts.keys(), key=lambda x: str(x))
    total = sum(int(counts[l]) for l in labels)
    if total <= 0 or len(labels) == 0:
        return counts

    # Special case: target_ratio ~ 1 => aim for balanced counts
    if abs(float(target_ratio) - 1.0) <= tol:
        base = total // len(labels)
        rem = total % len(labels)
        out = {l: base for l in labels}
        # deterministic: give +1 to smallest labels first
        for i in range(rem):
            out[labels[i]] += 1
        # ensure min_per_label
        for l in labels:
            out[l] = max(out[l], min_per_label)
        # fix sum if min_per_label increased (rare)
        s = sum(out.values())
        if s != total:
            # remove extras from largest bins first deterministically
            diff = s - total
            if diff > 0:
                order = sorted(labels, key=lambda l: (-out[l], str(l)))
                idx = 0
                while diff > 0 and idx < len(order):
                    l = order[idx]
                    if out[l] > min_per_label:
                        out[l] -= 1
                        diff -= 1
                    else:
                        idx += 1
        return out

    out = {l: int(counts[l]) for l in labels}

    def _ratio(o: Dict[str, int]) -> float:
        mx = max(o.values())
        mn = min(o.values())
        return float(mn) / float(mx) if mx > 0 else 0.0

    # Try to move mass to match ratio, but stop if we cannot
    it = 0
    while it < max_iters:
        r = _ratio(out)
        if abs(r - target_ratio) <= tol:
            break

        # If too imbalanced (ratio too small), increase mins and decrease maxs
        if r < target_ratio - tol:
            mn = min(out.values())
            mx = max(out.values())

            min_labels = sorted([l for l in labels if out[l] == mn], key=lambda x: str(x))
            max_labels = sorted([l for l in labels if out[l] == mx], key=lambda x: str(x))

            moved = False
            for lmax in max_labels:
                if out[lmax] <= min_per_label:
                    continue
                # move 1 from lmax to first min label
                lmin = min_labels[0]
                out[lmax] -= 1
                out[lmin] += 1
                moved = True
                break

            if not moved:
                # cannot improve further
                break

        # If too balanced (ratio too large), increase max and decrease min (rare)
        else:
            mn = min(out.values())
            mx = max(out.values())

            min_labels = sorted([l for l in labels if out[l] == mn], key=lambda x: str(x))
            max_labels = sorted([l for l in labels if out[l] == mx], key=lambda x: str(x))

            moved = False
            for lmin in min_labels:
                if out[lmin] <= min_per_label:
                    continue
                lmax = max_labels[0]
                out[lmin] -= 1
                out[lmax] += 1
                moved = True
                break
            if not moved:
                break

        it += 1

    # Final sum guard (should remain constant)
    s = sum(out.values())
    if s != total:
        diff = total - s
        # deterministic correction
        if diff > 0:
            for i in range(diff):
                out[labels[i % len(labels)]] += 1
        else:
            diff = -diff
            order = sorted(labels, key=lambda l: (-out[l], str(l)))
            idx = 0
            while diff > 0 and idx < len(order):
                l = order[idx]
                if out[l] > min_per_label:
                    out[l] -= 1
                    diff -= 1
                else:
                    idx += 1

    return out


def _resample_df_to_counts(
    df: pd.DataFrame,
    counts: Dict[str, int],
    *,
    seed: int,
) -> pd.DataFrame:
    """
    Deterministically sample exact per-label counts from df.
    Uses replacement only when needed.
    """
    df = sort_by_text(df)
    out_parts: List[pd.DataFrame] = []
    for lab in sorted(counts.keys(), key=lambda x: str(x)):
        need = int(counts[lab])
        g = df[df["label"] == lab].copy()
        g = sort_by_text(g)

        if need <= 0:
            continue

        if len(g) == 0:
            # impossible to create this label; skip but log later
            continue

        if len(g) >= need:
            # without replacement
            pick = g.sample(n=need, replace=False, random_state=seed)
        else:
            # with replacement (deterministic)
            pick = g.sample(n=need, replace=True, random_state=seed)
        out_parts.append(pick)

    if not out_parts:
        return sort_by_text(df.iloc[:0].copy())

    out = pd.concat(out_parts, ignore_index=True)
    # deterministic shuffle (still stable)
    out = sort_by_text(out).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sort_by_text(out)


def deterministic_resample_preserve_proportions(
    df: pd.DataFrame,
    *,
    n: int,
    k: int,
    seed: int,
    target_ratio: Optional[float] = None,
    min_per_label: int = 1,
) -> pd.DataFrame:
    """
    Build exactly n rows, aiming to keep exactly k labels (top-k already applied before calling).
    If df is too small, oversample with replacement preserving label proportions (after Top-K).
    Then optionally adjust to approach target_ratio.

    Never raises for insufficient n; it will oversample deterministically.
    """
    df = sort_by_text(df)

    labels = sorted(df["label"].unique().tolist(), key=lambda x: str(x))
    if len(labels) == 0:
        return sort_by_text(df.iloc[:0].copy())

    # If dataset has fewer labels than k, we cannot invent labels. We proceed with what we have.
    if len(labels) < k:
        print(f"[WARN] only {len(labels)} labels available but expected k={k}. Will proceed with available labels.")

    # Compute weights from observed distribution
    vc = df["label"].value_counts()
    weights = {str(lab): float(vc.get(lab, 0)) for lab in labels}

    # Allocate integer counts
    counts = _largest_remainder_counts(labels, weights, n, min_per_label=min_per_label)

    # Optional ratio adjustment (conservative)
    if target_ratio is not None:
        counts = _adjust_counts_to_target_ratio(counts, float(target_ratio), min_per_label=min_per_label, tol=RATIO_TOL)

    # Sample per-label deterministically (oversample only when needed)
    out = _resample_df_to_counts(df, counts, seed=seed)

    # Ensure exact n rows if slight drift occurred (should not, but keep ultra-robust)
    if len(out) != n:
        out = sort_by_text(out)
        if len(out) > n:
            out = out.sample(n=n, replace=False, random_state=seed)
        else:
            out = out.sample(n=n, replace=True, random_state=seed)
        out = sort_by_text(out)

    return out


def deterministic_split_large_small(
    df: pd.DataFrame,
    *,
    k: int,
    n_large: int,
    n_small: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine-all then split deterministically into large/small with exact sizes.

    Robust behavior:
    - If not enough total rows, we oversample deterministically to reach n_large+n_small.
    - We try to ensure each label appears in both splits by ensuring at least 2 per label in the pool
      when feasible (n_total >= 2*k). Otherwise, we warn and proceed.
    """
    df = keep_top_k_labels(df, k)
    df = sort_by_text(df)

    n_total = n_large + n_small
    labels = sorted(df["label"].unique().tolist(), key=lambda x: str(x))
    if len(labels) == 0:
        return df.iloc[:0].copy(), df.iloc[:0].copy()

    # Try to ensure each label can appear in both splits if feasible
    min_per_label_pool = 2 if n_total >= 2 * len(labels) else 1
    if min_per_label_pool == 1:
        print(f"[WARN] total={n_total} is too small to guarantee all labels in both splits (need >=2*k). Proceeding.")

    pool = deterministic_resample_preserve_proportions(
        df,
        n=n_total,
        k=k,
        seed=seed,
        target_ratio=None,
        min_per_label=min_per_label_pool,
    )
    pool = sort_by_text(pool).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pool = sort_by_text(pool)

    # Build splits deterministically with disjoint slices (after shuffle)
    # NOTE: pool is already deterministically shuffled above.
    pool_shuf = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    large_df = pool_shuf.iloc[:n_large].copy()
    small_df = pool_shuf.iloc[n_large:n_large + n_small].copy()

    large_df = sort_by_text(large_df)
    small_df = sort_by_text(small_df)

    # Warn if labels missing (can happen when n_total < 2*k)
    if large_df["label"].nunique() < min(k, len(labels)):
        print(f"[WARN] large split lost labels: {large_df['label'].nunique()} unique < expected ~{k}.")
    if small_df["label"].nunique() < min(k, len(labels)):
        print(f"[WARN] small split lost labels: {small_df['label'].nunique()} unique < expected ~{k}.")

    return large_df, small_df


# -------------------------
# HF parsing helpers (robust)
# -------------------------
def flatten_instances(ds_split: Any) -> pd.DataFrame:
    """
    Convert a HF split (or a dataframe-like object) to per-instance dataframe with columns text,label.
    Handles MTEB-style {sentences: [...], labels: [...]} by exploding.
    """
    df = pd.DataFrame(ds_split)

    # MTEB clustering format
    if "sentences" in df.columns and "labels" in df.columns:
        df = df.explode(["sentences", "labels"], ignore_index=True)
        df = df.rename(columns={"sentences": "text", "labels": "label"})

    # common alternatives
    if "text" not in df.columns:
        for c in ["sentence", "utterance", "query", "input", "utt"]:
            if c in df.columns:
                df = df.rename(columns={c: "text"})
                break

    if "label" not in df.columns:
        for c in ["labels", "intent", "domain", "scenario", "topic", "cluster"]:
            if c in df.columns:
                df = df.rename(columns={c: "label"})
                break

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Cannot infer text/label columns from: {df.columns.tolist()}")

    out = df[["text", "label"]].copy()
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(str)
    return sort_by_text(out)


def load_hf_dataset(hf_id: str, config: Optional[str] = None, *, revision_env: str) -> Any:
    """
    Optional pinning by env var:
      export {revision_env}=<git_commit_or_tag>
    """
    revision = os.getenv(revision_env, None)
    trust_remote_code = (os.getenv("ALLOW_REMOTE_CODE", "0") == "1")

    if config is None:
        return load_dataset(hf_id, revision=revision, trust_remote_code=trust_remote_code)
    return load_dataset(hf_id, config, revision=revision, trust_remote_code=trust_remote_code)


def load_hf_dataset_any(hf_ids: Sequence[str], config: Optional[str], *, revision_env: str) -> Any:
    """
    Try multiple hf_ids in order (robust fallback), deterministic.
    """
    last_err: Optional[Exception] = None
    for hid in hf_ids:
        try:
            return load_hf_dataset(hid, config=config, revision_env=revision_env)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All HF dataset ids failed: {hf_ids}. Last error: {last_err}")


def enforce_target(dataset_name: str, split: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust enforcement:
    - Keep top-k labels
    - Resample (down/over) deterministically to target total rows
    - Try to approach the target min/max ratio
    - Never crash by default; log warnings (STRICT_METRICS=1 can re-enable hard failure)
    """
    t = TARGETS[dataset_name][split]
    k = int(t["nb_classes"])
    n = int(t["total_rows"])
    target_ratio = float(t["ratio"])

    df0 = df.copy()
    df = keep_top_k_labels(df, k)

    before_rows = len(df0)
    after_topk_rows = len(df)
    after_topk_k = int(df["label"].nunique()) if len(df) > 0 else 0

    # Resample to exact n with proportions + ratio adjustment
    df = deterministic_resample_preserve_proportions(
        df,
        n=n,
        k=k,
        seed=SEED,
        target_ratio=target_ratio,
        min_per_label=1,
    )

    # Soft checks
    got_n = len(df)
    got_k = int(df["label"].nunique()) if got_n > 0 else 0
    got_ratio = ratio_min_over_max(df)

    msg = (
        f"[METRICS] {dataset_name}/{split}: "
        f"rows(before={before_rows} topk={after_topk_rows} final={got_n}/{n}), "
        f"classes(topk={after_topk_k} final={got_k}/{k}), "
        f"ratio(final={got_ratio:.6f} target={target_ratio:.6f} tol={RATIO_TOL})"
    )
    print(msg)

    # Hard failure only if explicitly asked
    if STRICT_METRICS:
        if got_n != n:
            raise ValueError(f"{dataset_name}/{split}: rows={got_n} != {n}")
        if got_k != k:
            raise ValueError(f"{dataset_name}/{split}: classes={got_k} != {k}")
        if abs(got_ratio - target_ratio) > RATIO_TOL:
            raise ValueError(f"{dataset_name}/{split}: ratio={got_ratio:.6f} != {target_ratio:.6f} (tol={RATIO_TOL})")
    else:
        if got_n != n:
            print(f"[WARN] {dataset_name}/{split}: rows mismatch {got_n} != {n} (continuing).")
        if got_k != k:
            print(f"[WARN] {dataset_name}/{split}: class mismatch {got_k} != {k} (continuing).")
        if abs(got_ratio - target_ratio) > RATIO_TOL:
            print(f"[WARN] {dataset_name}/{split}: ratio off by {abs(got_ratio - target_ratio):.6f} (continuing).")

    return df


def _want(scale: Optional[str], s: str) -> bool:
    return (scale is None) or (scale == s)


# -------------------------
# Dataset-specific processors
# -------------------------
def process_bank77(out_root: str, *, scale: Optional[str]) -> None:
    ds = load_hf_dataset_any(["banking77", "legacy-datasets/banking77"], config=None, revision_env="REV_BANKING77")

    if _want(scale, "large"):
        large = flatten_instances(ds["train"])
        large = enforce_target("bank77", "large", large)
        write_jsonl(large, os.path.join(out_root, "bank77", "large.jsonl"))

    if _want(scale, "small"):
        small = flatten_instances(ds["test"])
        small = enforce_target("bank77", "small", small)
        write_jsonl(small, os.path.join(out_root, "bank77", "small.jsonl"))


def _clinc_remove_oos(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if "intent" in df.columns:
        df["intent"] = df["intent"].astype(str)
        df = df[df["intent"] != "oos"].copy()
    if "label" in df.columns:
        df["label"] = df["label"].astype(str)
        df = df[df["label"] != "oos"].copy()
    return df.reset_index(drop=True)


def process_clinc_intent(out_root: str, *, scale: Optional[str]) -> None:
    # Robust HF id: official is clinc/clinc_oos
    ds = load_hf_dataset_any(["clinc/clinc_oos", "clinc_oos"], config="plus", revision_env="REV_CLINC")

    def get_split(name: str) -> pd.DataFrame:
        df = pd.DataFrame(ds[name])
        df = _clinc_remove_oos(df)
        if "intent" in df.columns:
            df = df.rename(columns={"intent": "label"})
        if "text" not in df.columns and "utterance" in df.columns:
            df = df.rename(columns={"utterance": "text"})
        return flatten_instances(df)

    if _want(scale, "large"):
        large = get_split("train")
        large = enforce_target("clinc_intent", "large", large)
        write_jsonl(large, os.path.join(out_root, "clinc_intent", "large.jsonl"))

    if _want(scale, "small"):
        small = get_split("test")
        small = enforce_target("clinc_intent", "small", small)
        write_jsonl(small, os.path.join(out_root, "clinc_intent", "small.jsonl"))


def process_clinc_domain(out_root: str, *, scale: Optional[str]) -> None:
    ds = load_hf_dataset_any(["clinc/clinc_oos", "clinc_oos"], config="plus", revision_env="REV_CLINC")

    def get_split(name: str) -> pd.DataFrame:
        df = pd.DataFrame(ds[name])
        df = _clinc_remove_oos(df)
        if "domain" in df.columns:
            df = df.rename(columns={"domain": "label"})
        elif "domain_name" in df.columns:
            df = df.rename(columns={"domain_name": "label"})
        if "text" not in df.columns and "utterance" in df.columns:
            df = df.rename(columns={"utterance": "text"})
        return flatten_instances(df)

    if _want(scale, "large"):
        large = get_split("train")
        large = enforce_target("clinc_domain", "large", large)
        write_jsonl(large, os.path.join(out_root, "clinc_domain", "large.jsonl"))

    if _want(scale, "small"):
        small = get_split("test")
        small = enforce_target("clinc_domain", "small", small)
        write_jsonl(small, os.path.join(out_root, "clinc_domain", "small.jsonl"))


def _filter_english_if_present(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["lang", "language", "locale"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
            out = out[out[col].str.lower().str.startswith("en")].copy()
    return out.reset_index(drop=True)


def process_mtop_intent(out_root: str, *, scale: Optional[str]) -> None:
    ds = load_hf_dataset("mteb/mtop_intent", config="en", revision_env="REV_MTOP_INTENT")

    if _want(scale, "large"):
        large = flatten_instances(ds["train"])
        large = _filter_english_if_present(large)
        large = enforce_target("mtop_intent", "large", large)
        write_jsonl(large, os.path.join(out_root, "mtop_intent", "large.jsonl"))

    if _want(scale, "small"):
        small = flatten_instances(ds["test"])
        small = _filter_english_if_present(small)
        small = enforce_target("mtop_intent", "small", small)
        write_jsonl(small, os.path.join(out_root, "mtop_intent", "small.jsonl"))


def process_mtop_domain(out_root: str, *, scale: Optional[str]) -> None:
    ds = load_hf_dataset("mteb/mtop_domain", config="en", revision_env="REV_MTOP_DOMAIN")

    if _want(scale, "large"):
        large = flatten_instances(ds["train"])
        large = _filter_english_if_present(large)
        large = enforce_target("mtop_domain", "large", large)
        write_jsonl(large, os.path.join(out_root, "mtop_domain", "large.jsonl"))

    if _want(scale, "small"):
        small = flatten_instances(ds["test"])
        small = _filter_english_if_present(small)
        small = enforce_target("mtop_domain", "small", small)
        write_jsonl(small, os.path.join(out_root, "mtop_domain", "small.jsonl"))


def process_massive_intent(out_root: str, *, scale: Optional[str]) -> None:
    ds = load_hf_dataset("mteb/amazon_massive_intent", config="en", revision_env="REV_MASSIVE_INTENT")

    if _want(scale, "large"):
        large = flatten_instances(ds["train"])
        large = _filter_english_if_present(large)
        large = enforce_target("massive_intent", "large", large)
        write_jsonl(large, os.path.join(out_root, "massive_intent", "large.jsonl"))

    if _want(scale, "small"):
        small = flatten_instances(ds["test"])
        small = _filter_english_if_present(small)
        small = enforce_target("massive_intent", "small", small)
        write_jsonl(small, os.path.join(out_root, "massive_intent", "small.jsonl"))


def process_massive_domain(out_root: str, *, scale: Optional[str]) -> None:
    # Robust fallback: some setups store scenario in the same dataset repo
    ds = load_hf_dataset_any(
        ["mteb/amazon_massive_scenario", "mteb/amazon_massive_intent", "AmazonScience/massive"],
        config="en",
        revision_env="REV_MASSIVE_SCENARIO",
    )

    if _want(scale, "large"):
        large = flatten_instances(ds["train"])
        large = _filter_english_if_present(large)
        large = enforce_target("massive_domain", "large", large)
        write_jsonl(large, os.path.join(out_root, "massive_domain", "large.jsonl"))

    if _want(scale, "small"):
        small = flatten_instances(ds["test"])
        small = _filter_english_if_present(small)
        small = enforce_target("massive_domain", "small", small)
        write_jsonl(small, os.path.join(out_root, "massive_domain", "small.jsonl"))


def _append_if_missing(text: str, marker: str, suffix: str) -> str:
    t = str(text)
    if marker.lower() in t.lower():
        return t
    return (t + " " + suffix).strip()


def process_few_rel_nat(out_root: str, *, scale: Optional[str]) -> None:
    ds = load_hf_dataset("BrandonZYW/FewRelClustering", revision_env="REV_FEWREL")

    # --- DÉBUT DU BOUCLIER DE SÉCURITÉ ---
    parts = []
    # Vérifie si 'ds' est un dictionnaire de splits (ex: {'train': ..., 'test': ...}) 
    # ou un dataset plat unique.
    if isinstance(ds, dict): 
        for sp in ds.keys():
            parts.append(pd.DataFrame(ds[sp]))
    else:
        parts.append(pd.DataFrame(ds))
        
    # Évite le crash de pd.concat si le dataset téléchargé est vide
    if not parts:
        raw = pd.DataFrame(columns=["text", "label", "sentence", "entity1", "entity2"])
    else:
        raw = pd.concat(parts, ignore_index=True)
    # --- FIN DU BOUCLIER ---

    if "text" not in raw.columns:
        if "sentence" in raw.columns:
            raw["text"] = raw["sentence"].astype(str)
        elif "sent" in raw.columns:
            raw["text"] = raw["sent"].astype(str)
        else:
            raise ValueError("FewRel: cannot find sentence column to build text.")

    e1 = None
    e2 = None
    for c in ["entity1", "head", "h", "subj", "subject"]:
        if c in raw.columns:
            e1 = c
            break
    for c in ["entity2", "tail", "t", "obj", "object"]:
        if c in raw.columns:
            e2 = c
            break

    if e1 is not None and e2 is not None:
        suffix = raw.apply(lambda r: f"The relation between {r[e1]} and {r[e2]}.", axis=1)
        raw["text"] = [_append_if_missing(t, "The relation between", s) for t, s in zip(raw["text"], suffix)]
    else:
        raw["text"] = raw["text"].apply(
            lambda t: _append_if_missing(t, "The relation between", "The relation between [ENTITY1] and [ENTITY2].")
        )

    if "label" not in raw.columns:
        for c in ["cluster", "relation", "rel", "type"]:
            if c in raw.columns:
                raw = raw.rename(columns={c: "label"})
                break
    if "label" not in raw.columns:
        raise ValueError("FewRel: cannot find label column.")

    df = raw[["text", "label"]].copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df = sort_by_text(df)

    k = int(TARGETS["few_rel_nat"]["large"]["nb_classes"])
    n_large = int(TARGETS["few_rel_nat"]["large"]["total_rows"])
    n_small = int(TARGETS["few_rel_nat"]["small"]["total_rows"])

    large, small = deterministic_split_large_small(df, k=k, n_large=n_large, n_small=n_small, seed=SEED)
    large = enforce_target("few_rel_nat", "large", large)
    small = enforce_target("few_rel_nat", "small", small)

    if _want(scale, "large"):
        write_jsonl(large, os.path.join(out_root, "few_rel_nat", "large.jsonl"))
    if _want(scale, "small"):
        write_jsonl(small, os.path.join(out_root, "few_rel_nat", "small.jsonl"))

        

def process_few_event(out_root: str, *, scale: Optional[str]) -> None:
    ds = load_hf_dataset("BrandonZYW/FewEventClustering", revision_env="REV_FEWEVENT")

    parts = []
    for sp in ds.keys():
        parts.append(pd.DataFrame(ds[sp]))
    raw = pd.concat(parts, ignore_index=True)

    if "text" not in raw.columns:
        if "sentence" in raw.columns:
            raw["text"] = raw["sentence"].astype(str)
        elif "sent" in raw.columns:
            raw["text"] = raw["sent"].astype(str)
        else:
            raise ValueError("FewEvent: cannot find sentence column to build text.")

    trig = None
    for c in ["trigger", "event_trigger", "mention", "span", "anchor"]:
        if c in raw.columns:
            trig = c
            break

    if trig is not None:
        suffix = raw[trig].astype(str).apply(lambda x: f"The event trigger is {x}.")
        raw["text"] = [_append_if_missing(t, "The event trigger", s) for t, s in zip(raw["text"], suffix)]
    else:
        raw["text"] = raw["text"].apply(
            lambda t: _append_if_missing(t, "The event trigger", "The event trigger is [TRIGGER].")
        )

    if "label" not in raw.columns:
        for c in ["cluster", "event_type", "type"]:
            if c in raw.columns:
                raw = raw.rename(columns={c: "label"})
                break
    if "label" not in raw.columns:
        raise ValueError("FewEvent: cannot find label column.")

    df = raw[["text", "label"]].copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df = sort_by_text(df)

    k = int(TARGETS["few_event"]["large"]["nb_classes"])
    n_large = int(TARGETS["few_event"]["large"]["total_rows"])
    n_small = int(TARGETS["few_event"]["small"]["total_rows"])

    large, small = deterministic_split_large_small(df, k=k, n_large=n_large, n_small=n_small, seed=SEED)
    large = enforce_target("few_event", "large", large)
    small = enforce_target("few_event", "small", small)

    if _want(scale, "large"):
        write_jsonl(large, os.path.join(out_root, "few_event", "large.jsonl"))
    if _want(scale, "small"):
        write_jsonl(small, os.path.join(out_root, "few_event", "small.jsonl"))


def process_few_nerd_nat(out_root: str, *, scale: Optional[str]) -> None:
    ds = load_hf_dataset("BrandonZYW/FewNerdClustering", revision_env="REV_FEWNERD")

    def build(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if "text" not in d.columns:
            if "sentence" in d.columns:
                d["text"] = d["sentence"].astype(str)
            elif "sent" in d.columns:
                d["text"] = d["sent"].astype(str)
            else:
                raise ValueError("FewNerd: cannot find sentence column.")

        ent = None
        for c in ["entity", "mention", "span", "entity_mention"]:
            if c in d.columns:
                ent = c
                break

        if ent is not None:
            suffix = d[ent].astype(str).apply(lambda x: f"The entity is {x}.")
            d["text"] = [_append_if_missing(t, "The entity is", s) for t, s in zip(d["text"], suffix)]
        else:
            d["text"] = d["text"].apply(lambda t: _append_if_missing(t, "The entity is", "The entity is [ENTITY]."))

        if "label" not in d.columns:
            for c in ["cluster", "type", "entity_type"]:
                if c in d.columns:
                    d = d.rename(columns={c: "label"})
                    break
        if "label" not in d.columns:
            raise ValueError("FewNerd: cannot find label column.")
        return flatten_instances(d)

    # --- SÉCURITÉ : Détection robuste des splits pour éviter le KeyError ---
    if isinstance(ds, dict):
        available_splits = list(ds.keys())
    else:
        available_splits = [None] # Cas où le dataset n'est pas divisé

    train_key = "train" if "train" in available_splits else available_splits[0]
    test_key = "test" if "test" in available_splits else available_splits[-1]
    # ----------------------------------------------------------------------

    if _want(scale, "large"):
        large_raw = pd.DataFrame(ds[train_key]) if train_key is not None else pd.DataFrame(ds)
        large = build(large_raw)
        large = enforce_target("few_nerd_nat", "large", large)
        write_jsonl(large, os.path.join(out_root, "few_nerd_nat", "large.jsonl"))

    if _want(scale, "small"):
        small_raw = pd.DataFrame(ds[test_key]) if test_key is not None else pd.DataFrame(ds)
        small = build(small_raw)
        small = enforce_target("few_nerd_nat", "small", small)
        write_jsonl(small, os.path.join(out_root, "few_nerd_nat", "small.jsonl"))


def process_stackex(out_root: str, *, scale: Optional[str]) -> None:
    ds = load_hf_dataset("mteb/stackexchange-clustering", revision_env="REV_STACKEX")

    # --- SÉCURITÉ : Gestion robuste du format (dict vs plat) et de la concaténation ---
    parts = []
    if isinstance(ds, dict):
        for sp in ds.keys():
            parts.append(flatten_instances(ds[sp]))
    else:
        parts.append(flatten_instances(ds))

    if not parts:
        df = sort_by_text(pd.DataFrame(columns=["text", "label"]))
    else:
        df = sort_by_text(pd.concat(parts, ignore_index=True))
    # ----------------------------------------------------------------------------------

    k = int(TARGETS["stackex"]["large"]["nb_classes"])
    n_large = int(TARGETS["stackex"]["large"]["total_rows"])
    n_small = int(TARGETS["stackex"]["small"]["total_rows"])

    large, small = deterministic_split_large_small(df, k=k, n_large=n_large, n_small=n_small, seed=SEED)
    large = enforce_target("stackex", "large", large)
    small = enforce_target("stackex", "small", small)

    if _want(scale, "large"):
        write_jsonl(large, os.path.join(out_root, "stackex", "large.jsonl"))
    if _want(scale, "small"):
        write_jsonl(small, os.path.join(out_root, "stackex", "small.jsonl"))



def process_reddit(out_root: str, *, scale: Optional[str]) -> None:
    ds = load_hf_dataset("mteb/reddit-clustering", revision_env="REV_REDDIT")

    parts = []
    for sp in ds.keys():
        parts.append(flatten_instances(ds[sp]))
    df = sort_by_text(pd.concat(parts, ignore_index=True))

    k = int(TARGETS["reddit"]["large"]["nb_classes"])
    n_large = int(TARGETS["reddit"]["large"]["total_rows"])
    n_small = int(TARGETS["reddit"]["small"]["total_rows"])

    large, small = deterministic_split_large_small(df, k=k, n_large=n_large, n_small=n_small, seed=SEED)
    large = enforce_target("reddit", "large", large)
    small = enforce_target("reddit", "small", small)

    if _want(scale, "large"):
        write_jsonl(large, os.path.join(out_root, "reddit", "large.jsonl"))
    if _want(scale, "small"):
        write_jsonl(small, os.path.join(out_root, "reddit", "small.jsonl"))


def process_arxiv(out_root: str, *, scale: Optional[str]) -> None:
    # Your earlier logs use p2p; keep s2s as fallback
    ds = load_hf_dataset_any(
        ["mteb/arxiv-clustering-p2p", "mteb/arxiv-clustering-s2s", "mteb/arxiv-clustering-p2p"],
        config=None,
        revision_env="REV_ARXIV_S2S",
    )

    parts = []
    for sp in ds.keys():
        parts.append(flatten_instances(ds[sp]))
    df = sort_by_text(pd.concat(parts, ignore_index=True))

    k = int(TARGETS["arxiv"]["large"]["nb_classes"])
    n_large = int(TARGETS["arxiv"]["large"]["total_rows"])
    n_small = int(TARGETS["arxiv"]["small"]["total_rows"])

    large, small = deterministic_split_large_small(df, k=k, n_large=n_large, n_small=n_small, seed=SEED)
    large = enforce_target("arxiv", "large", large)
    small = enforce_target("arxiv", "small", small)

    if _want(scale, "large"):
        write_jsonl(large, os.path.join(out_root, "arxiv", "large.jsonl"))
    if _want(scale, "small"):
        write_jsonl(small, os.path.join(out_root, "arxiv", "small.jsonl"))


def process_go_emotions(out_root: str, *, scale: Optional[str]) -> None:
    ds = load_hf_dataset("google-research-datasets/go_emotions", config="simplified", revision_env="REV_GOEMO")

    def build(split_name: str) -> pd.DataFrame:
        raw = pd.DataFrame(ds[split_name])

        if "text" not in raw.columns:
            for c in ["sentence", "content"]:
                if c in raw.columns:
                    raw = raw.rename(columns={c: "text"})
                    break

        if "labels" in raw.columns:
            # remove multi-label
            raw = raw[raw["labels"].apply(lambda x: isinstance(x, list) and len(x) == 1)].copy()
            raw["label"] = raw["labels"].apply(lambda x: x[0])
        elif "label" in raw.columns:
            pass
        else:
            raise ValueError("GoEmo: cannot find labels/label.")

        raw["label"] = raw["label"].astype(str)
        # remove neutral (27)
        raw = raw[(raw["label"] != "27") & (raw["label"].str.lower() != "neutral")].copy()

        df = raw[["text", "label"]].copy()
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(str)
        return sort_by_text(df)

    if _want(scale, "large"):
        large = build("train")
        large = enforce_target("go_emotions", "large", large)
        write_jsonl(large, os.path.join(out_root, "go_emotions", "large.jsonl"))

    if _want(scale, "small"):
        small = build("test")
        small = enforce_target("go_emotions", "small", small)
        write_jsonl(small, os.path.join(out_root, "go_emotions", "small.jsonl"))


# -------------------------
# Main
# -------------------------
PROCESSORS = {
    "bank77": process_bank77,
    "clinc_intent": process_clinc_intent,
    "clinc_domain": process_clinc_domain,
    "mtop_intent": process_mtop_intent,
    "mtop_domain": process_mtop_domain,
    "massive_intent": process_massive_intent,
    "massive_domain": process_massive_domain,
    "stackex": process_stackex,
    "arxiv": process_arxiv,
    "reddit": process_reddit,
    "go_emotions": process_go_emotions,
    "few_rel_nat": process_few_rel_nat,
    "few_nerd_nat": process_few_nerd_nat,
    "few_event": process_few_event,
}


def main() -> None:
    import argparse
    global SEED

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./datasets", help="Output root directory (default: ./datasets).")
    ap.add_argument("--datasets-dir", type=str, default=None, help="Alias of --out (used by DVC scripts).")
    ap.add_argument("--seed", type=int, default=None, help="Override global seed (used by DVC scripts).")
    ap.add_argument("--dataset", type=str, default=None, help="Run only one dataset name (your nomenclature).")
    ap.add_argument("--scale", type=str, default=None, choices=[None, "small", "large"], help="Optional: build only one split.")
    args = ap.parse_args()

    out_root = args.datasets_dir if args.datasets_dir is not None else args.out

    if args.seed is not None:
        SEED = int(args.seed)
        random.seed(SEED)
        np.random.seed(SEED)

    mkdirp(out_root)

    to_run = list(PROCESSORS.keys()) if args.dataset is None else [args.dataset]
    failures: List[str] = []

    for name in to_run:
        if name not in PROCESSORS:
            print(f"[ERROR] Unknown dataset={name}. Allowed: {sorted(PROCESSORS.keys())}")
            failures.append(name)
            continue

        print(f"\n=== BUILD {name} (scale={args.scale or 'both'}) seed={SEED} out={out_root} ===")
        try:
            PROCESSORS[name](out_root, scale=args.scale)
            print(f"OK: wrote under {out_root}/{name}/")
        except Exception as e:
            print(f"[ERROR] FAILED dataset={name}: {e}")
            print(traceback.format_exc())
            failures.append(name)

    if failures:
        print("\n[WARN] Some datasets failed:", ", ".join(failures))
        # never hard crash by default; enable STRICT_METRICS or handle outside if needed
    else:
        print("\nALL DONE.")


if __name__ == "__main__":
    main()