import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =========================================================
# CONFIG
# =========================================================
DATASETS_ROOT = Path("/home/hamady_gackou_work/clusterllm-local/src/clusterllm/datasets")
OUTPUT_FILE = "all_dataset_results.txt"

VALID_EXTENSIONS = {".json", ".jsonl", ".txt", ".yaml", ".yml", ".log"}

INTERESTING_NAME_PARTS = [
    "measure", "metric", "summary", "result", "eval", "train", "embeds"
]

DATASET_KEYS = ["dataset", "dataset_name", "data", "task"]
VERSION_KEYS = ["version", "size", "scale"]

LLM1_KEYS = [
    "llm_phase1", "phase1_llm", "llm1", "teacher_llm", "gen_llm",
    "embed_llm", "train_llm", "llm_tr", "llm_train", "t"
]

LLM2_KEYS = [
    "llm_phase2", "phase2_llm", "llm2", "student_llm", "pred_llm",
    "pair_llm", "eval_llm", "llm_pred", "llm_pe", "p"
]

METRIC_PREFIXES = {
    "ACC", "NMI", "ARI", "F1", "PREC", "RECALL", "PURITY", "AMI", "VMEASURE"
}

PRIMARY_MATCH_METRICS = ["ACC_mean", "NMI_mean", "ARI_mean"]
SECONDARY_MATCH_METRICS = ["ACC_std", "NMI_std", "ARI_std"]

# seuil empirique pour accepter un matching par proximité
MAX_MATCH_DISTANCE = 8.0

# =========================================================
# OUTILS DE BASE
# =========================================================
def is_candidate_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False

    name = path.name.lower()
    return any(part in name for part in INTERESTING_NAME_PARTS)


def detect_version_from_filename(filename: str) -> Optional[str]:
    low = filename.lower()
    if low.startswith("small_"):
        return "small"
    if low.startswith("large_"):
        return "large"
    return None


def detect_result_type_from_filename(filename: str) -> str:
    low = filename.lower()
    if "_cont" in low:
        return "contribution"
    return "reproduction_fidele"


def detect_dataset_from_path(path: Path, datasets_root: Path) -> Optional[str]:
    try:
        rel = path.relative_to(datasets_root)
        parts = rel.parts
        if len(parts) >= 2:
            return parts[0]
    except Exception:
        pass
    return None


def safe_read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return ""


def try_load_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def try_load_jsonl(path: Path) -> Optional[List[Any]]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows if rows else None
    except Exception:
        return None


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    flat = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            flat.update(flatten_json(v, new_prefix))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            flat.update(flatten_json(v, new_prefix))
    else:
        flat[prefix] = obj

    return flat


def find_value_by_keys(flat: Dict[str, Any], possible_keys: List[str]) -> Optional[str]:
    normalized = {k.lower(): v for k, v in flat.items()}

    for wanted in possible_keys:
        wanted = wanted.lower()
        for full_k, v in normalized.items():
            last = full_k.split(".")[-1]
            if last == wanted or full_k == wanted:
                if isinstance(v, (str, int, float, bool)):
                    return str(v)
    return None


def clean_model_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    s = str(value).strip()
    s = s.replace("\\", "/")
    s = re.sub(r"\s+", "", s)
    s = s.strip("_-=:")
    return s or None


# =========================================================
# EXTRACTION DES LLMs
# =========================================================
def infer_llms_from_filename(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Cherche t=... et p=... dans le nom de fichier.
    Retourne (llm1, llm2, source)
    """
    stem = Path(filename).stem

    llm1 = None
    llm2 = None

    mt = re.search(r"(?:^|__)t=([^=]+?)(?=__p=|$)", stem, flags=re.IGNORECASE)
    mp = re.search(r"(?:^|__)p=([^=]+?)(?=__|$)", stem, flags=re.IGNORECASE)

    if mt:
        llm1 = mt.group(1)
    if mp:
        llm2 = mp.group(1)

    llm1 = clean_model_name(llm1)
    llm2 = clean_model_name(llm2)

    source = "direct_filename" if llm1 and llm2 else None
    return llm1, llm2, source


def infer_llms_from_text(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    llm1 = None
    llm2 = None

    patterns = [
        (r'["\']?(llm_phase1|phase1_llm|llm1|teacher_llm|gen_llm|embed_llm|train_llm|llm_tr|t)["\']?\s*[:=]\s*["\']([^"\']+)["\']', 1),
        (r'["\']?(llm_phase2|phase2_llm|llm2|student_llm|pred_llm|pair_llm|eval_llm|llm_pred|llm_pe|p)["\']?\s*[:=]\s*["\']([^"\']+)["\']', 2),
    ]

    for pattern, phase in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            if phase == 1 and not llm1:
                llm1 = m.group(2)
            elif phase == 2 and not llm2:
                llm2 = m.group(2)

    llm1 = clean_model_name(llm1)
    llm2 = clean_model_name(llm2)
    source = "direct_content" if llm1 and llm2 else None
    return llm1, llm2, source


# =========================================================
# EXTRACTION DES METRIQUES
# =========================================================
def is_metric_key(key: str) -> bool:
    k = key.strip()

    if re.fullmatch(r"[A-Za-z]+_(mean|std)", k):
        return True

    if re.fullmatch(r"[A-Za-z0-9]+_[A-Za-z0-9]+", k):
        first = k.split("_")[0].upper()
        if first in METRIC_PREFIXES:
            return True

    return False


def normalize_metric_key(key: str) -> str:
    return key.strip()


def to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def extract_metrics_from_flat(flat: Dict[str, Any]) -> Dict[str, float]:
    metrics = {}
    for k, v in flat.items():
        last = k.split(".")[-1]
        if is_metric_key(last):
            val = to_float(v)
            if val is not None:
                metrics[normalize_metric_key(last)] = val
    return metrics


def extract_metrics_from_text(text: str) -> Dict[str, float]:
    metrics = {}
    pattern = r'["\']([A-Za-z0-9]+_(?:mean|std))["\']\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'

    for m in re.finditer(pattern, text):
        key = normalize_metric_key(m.group(1))
        val = to_float(m.group(2))
        if val is not None:
            metrics[key] = val

    return metrics


# =========================================================
# ANALYSE DE FICHIER
# =========================================================
def analyze_json_file(path: Path) -> Optional[Dict[str, Any]]:
    data = try_load_json(path)
    if data is None:
        return None

    flat = flatten_json(data)

    dataset = find_value_by_keys(flat, DATASET_KEYS)
    version = find_value_by_keys(flat, VERSION_KEYS)

    llm1_flat = find_value_by_keys(flat, LLM1_KEYS)
    llm2_flat = find_value_by_keys(flat, LLM2_KEYS)

    metrics = extract_metrics_from_flat(flat)

    text = json.dumps(data, ensure_ascii=False)
    llm1_txt, llm2_txt, text_source = infer_llms_from_text(text)

    llm1 = clean_model_name(llm1_flat) or llm1_txt
    llm2 = clean_model_name(llm2_flat) or llm2_txt

    llm_source = None
    if clean_model_name(llm1_flat) and clean_model_name(llm2_flat):
        llm_source = "direct_content"
    elif text_source:
        llm_source = text_source

    return {
        "dataset": dataset,
        "version": version,
        "llm_phase1": llm1,
        "llm_phase2": llm2,
        "llm_source": llm_source,
        "metrics": metrics,
    }


def analyze_jsonl_file(path: Path) -> Optional[Dict[str, Any]]:
    rows = try_load_jsonl(path)
    if rows is None:
        return None

    merged_flat = {}
    metrics = {}

    for idx, row in enumerate(rows):
        flat = flatten_json(row, prefix=f"row{idx}")
        merged_flat.update(flat)
        metrics.update(extract_metrics_from_flat(flat))

    dataset = find_value_by_keys(merged_flat, DATASET_KEYS)
    version = find_value_by_keys(merged_flat, VERSION_KEYS)

    llm1_flat = find_value_by_keys(merged_flat, LLM1_KEYS)
    llm2_flat = find_value_by_keys(merged_flat, LLM2_KEYS)

    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    llm1_txt, llm2_txt, text_source = infer_llms_from_text(text)

    llm1 = clean_model_name(llm1_flat) or llm1_txt
    llm2 = clean_model_name(llm2_flat) or llm2_txt

    llm_source = None
    if clean_model_name(llm1_flat) and clean_model_name(llm2_flat):
        llm_source = "direct_content"
    elif text_source:
        llm_source = text_source

    return {
        "dataset": dataset,
        "version": version,
        "llm_phase1": llm1,
        "llm_phase2": llm2,
        "llm_source": llm_source,
        "metrics": metrics,
    }


def analyze_text_file(path: Path) -> Optional[Dict[str, Any]]:
    text = safe_read_text(path)
    if not text.strip():
        return None

    dataset = None
    version = None

    for key in DATASET_KEYS:
        m = re.search(
            rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            text,
            flags=re.IGNORECASE
        )
        if m:
            dataset = m.group(1).strip()
            break

    for key in VERSION_KEYS:
        m = re.search(
            rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            text,
            flags=re.IGNORECASE
        )
        if m:
            version = m.group(1).strip()
            break

    llm1, llm2, llm_source = infer_llms_from_text(text)
    metrics = extract_metrics_from_text(text)

    return {
        "dataset": dataset,
        "version": version,
        "llm_phase1": llm1,
        "llm_phase2": llm2,
        "llm_source": llm_source,
        "metrics": metrics,
    }


def analyze_file(path: Path, datasets_root: Path) -> Optional[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        result = analyze_json_file(path)
    elif path.suffix.lower() == ".jsonl":
        result = analyze_jsonl_file(path)
    else:
        result = analyze_text_file(path)

    if result is None:
        return None

    result["dataset"] = result["dataset"] or detect_dataset_from_path(path, datasets_root)
    result["version"] = result["version"] or detect_version_from_filename(path.name)

    llm1_name, llm2_name, filename_source = infer_llms_from_filename(path.name)

    if not result["llm_phase1"] and llm1_name:
        result["llm_phase1"] = llm1_name
    if not result["llm_phase2"] and llm2_name:
        result["llm_phase2"] = llm2_name
    if not result["llm_source"] and filename_source:
        result["llm_source"] = filename_source

    result["result_type"] = detect_result_type_from_filename(path.name)

    result["dataset"] = result["dataset"].strip() if result["dataset"] else None
    result["version"] = result["version"].strip().lower() if result["version"] else None
    result["llm_phase1"] = clean_model_name(result["llm_phase1"])
    result["llm_phase2"] = clean_model_name(result["llm_phase2"])

    if not result["dataset"]:
        return None
    if result["version"] not in {"small", "large"}:
        return None
    if not result["metrics"]:
        return None

    result["file"] = str(path)
    result["relative_file"] = str(path.relative_to(datasets_root))
    result["has_ft"] = "ft" in path.name.lower()
    result["is_cont"] = "_cont" in path.name.lower()

    return result


# =========================================================
# DISTANCE ENTRE METRIQUES
# =========================================================
def metric_distance(metrics_a: Dict[str, float], metrics_b: Dict[str, float]) -> Optional[float]:
    """
    Distance robuste entre deux ensembles de métriques.
    Plus la distance est petite, plus les résultats sont proches.
    """
    compared = []

    ordered_keys = PRIMARY_MATCH_METRICS + SECONDARY_MATCH_METRICS
    for key in ordered_keys:
        if key in metrics_a and key in metrics_b:
            compared.append((key, metrics_a[key], metrics_b[key]))

    if len(compared) < 2:
        # fallback : intersection générale
        inter = sorted(set(metrics_a.keys()) & set(metrics_b.keys()))
        for key in inter:
            compared.append((key, metrics_a[key], metrics_b[key]))

    if len(compared) < 2:
        return None

    total = 0.0
    count = 0

    for key, va, vb in compared:
        scale = max(abs(va), abs(vb), 1.0)
        diff = abs(va - vb) / scale

        # on donne plus de poids aux métriques _mean
        weight = 2.0 if key.endswith("_mean") else 1.0
        total += weight * diff * 100.0
        count += weight

    if count == 0:
        return None

    return total / count


# =========================================================
# RESOLUTION DES LLMs POUR CONTRIBUTION
# =========================================================
def build_reference_index(results: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
    Indexe les résultats fidèles avec LLMs connus par (dataset, version).
    """
    ref_index = defaultdict(list)

    for r in results:
        if r["result_type"] != "reproduction_fidele":
            continue
        if not r.get("llm_phase1") or not r.get("llm_phase2"):
            continue

        key = (r["dataset"], r["version"])
        ref_index[key].append(r)

    return ref_index


def choose_best_reference(
    cont_result: Dict[str, Any],
    candidates: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """
    Choisit le candidat fidèle le plus proche en métriques.
    """
    best = None
    best_dist = None

    for cand in candidates:
        # Heuristique douce : si contribution a ft, on préfère aussi un fidèle avec ft
        # sans exclure les autres.
        dist = metric_distance(cont_result["metrics"], cand["metrics"])
        if dist is None:
            continue

        bonus = 0.0
        if cont_result.get("has_ft") and cand.get("has_ft"):
            bonus -= 0.5

        effective_dist = dist + bonus

        if best is None or effective_dist < best_dist:
            best = cand
            best_dist = effective_dist

    return best, best_dist


def resolve_missing_llms_for_contributions(results: List[Dict[str, Any]]) -> None:
    """
    Pour les résultats contribution sans LLMs explicites,
    récupère les LLMs par rapprochement métrique avec la reproduction fidèle.
    """
    ref_index = build_reference_index(results)

    for r in results:
        if r["result_type"] != "contribution":
            continue

        if r.get("llm_phase1") and r.get("llm_phase2"):
            continue

        key = (r["dataset"], r["version"])
        candidates = ref_index.get(key, [])
        if not candidates:
            r["match_status"] = "no_reference_found"
            continue

        best, best_dist = choose_best_reference(r, candidates)
        if best is None or best_dist is None:
            r["match_status"] = "no_metric_match"
            continue

        if best_dist <= MAX_MATCH_DISTANCE:
            r["llm_phase1"] = best["llm_phase1"]
            r["llm_phase2"] = best["llm_phase2"]
            r["llm_source"] = "matched_by_metrics"
            r["matched_reference_file"] = best["file"]
            r["match_distance"] = round(best_dist, 6)
            r["match_status"] = "matched"
        else:
            r["match_status"] = "match_too_weak"
            r["match_distance"] = round(best_dist, 6)


# =========================================================
# COLLECTE
# =========================================================
def collect_all_results(datasets_root: Path) -> List[Dict[str, Any]]:
    results = []

    for dataset_dir in sorted(datasets_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        for path in sorted(dataset_dir.iterdir()):
            if not is_candidate_file(path):
                continue

            result = analyze_file(path, datasets_root)
            if result is not None:
                results.append(result)

    return results


# =========================================================
# FILTRAGE FINAL
# =========================================================
def keep_only_usable_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept = []

    for r in results:
        if not r.get("dataset"):
            continue
        if r.get("version") not in {"small", "large"}:
            continue
        if not r.get("metrics"):
            continue
        if not r.get("llm_phase1") or not r.get("llm_phase2"):
            continue
        kept.append(r)

    return kept


# =========================================================
# ECRITURE
# =========================================================
def write_output(results: List[Dict[str, Any]], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("==== RESULTATS CONSOLIDES DES DATASETS ====\n")
        f.write(f"Nombre de résultats gardés : {len(results)}\n\n")

        for i, r in enumerate(results, 1):
            f.write("=" * 100 + "\n")
            f.write(f"RESULTAT #{i}\n")
            f.write(f"DATASET            : {r['dataset']}\n")
            f.write(f"VERSION            : {r['version']}\n")
            f.write(f"TYPE_RESULTAT      : {r['result_type']}\n")
            f.write(f"LLM PHASE 1        : {r['llm_phase1']}\n")
            f.write(f"LLM PHASE 2        : {r['llm_phase2']}\n")
            f.write(f"SOURCE_LLM         : {r.get('llm_source', 'unknown')}\n")
            f.write(f"FT_PRESENT         : {r.get('has_ft', False)}\n")
            f.write(f"FICHIER            : {r['file']}\n")
            f.write(f"REL_PATH           : {r['relative_file']}\n")

            if r.get("matched_reference_file"):
                f.write(f"REFERENCE_MATCH    : {r['matched_reference_file']}\n")
            if r.get("match_distance") is not None:
                f.write(f"MATCH_DISTANCE     : {r['match_distance']}\n")
            if r.get("match_status"):
                f.write(f"MATCH_STATUS       : {r['match_status']}\n")

            f.write("METRIQUES          :\n")
            for mk, mv in sorted(r["metrics"].items()):
                f.write(f"  - {mk}: {mv}\n")
            f.write("\n")


def print_summary(raw_results: List[Dict[str, Any]], final_results: List[Dict[str, Any]]) -> None:
    total = len(raw_results)
    kept = len(final_results)

    faithful = sum(1 for r in final_results if r["result_type"] == "reproduction_fidele")
    contrib = sum(1 for r in final_results if r["result_type"] == "contribution")
    matched = sum(1 for r in final_results if r.get("llm_source") == "matched_by_metrics")

    print(f"[OK] Résultats analysés : {total}")
    print(f"[OK] Résultats gardés   : {kept}")
    print(f"[OK] Fidèles gardés     : {faithful}")
    print(f"[OK] Contribution gardés: {contrib}")
    print(f"[OK] LLMs déduits par matching métrique : {matched}")


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    if not DATASETS_ROOT.exists():
        print(f"[ERREUR] Dossier introuvable : {DATASETS_ROOT}")
        return

    raw_results = collect_all_results(DATASETS_ROOT)

    # Etape clé : déduction robuste des LLMs manquants
    resolve_missing_llms_for_contributions(raw_results)

    final_results = keep_only_usable_results(raw_results)
    write_output(final_results, OUTPUT_FILE)

    print(f"[OK] Fichier créé : {OUTPUT_FILE}")
    print_summary(raw_results, final_results)


if __name__ == "__main__":
    main()