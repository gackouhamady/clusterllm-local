import json
import re
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
RUN_CONFIGS = [
    {
        "root": Path("/home/hamady_gackou_work/clusterllm-local/runs/summary/full_pipeline_2llms"),
        "result_type": "reproduction_fidele",
        "summary_filenames": ["summary.json"],
    },
    {
        "root": Path("/home/hamady_gackou_work/clusterllm-local/runs_cont/summary/contrib_2llms_cont"),
        "result_type": "contribution",
        "summary_filenames": ["summary_cont.json", "summary.json"],
    },
]

OUTPUT_FILE = "all_granularities_results.txt"
EXPECTED_VERSIONS = {"small", "large"}

# =========================================================
# OUTILS GENERAUX
# =========================================================
def safe_load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def flatten_json(obj, prefix=""):
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


def clean_text(value):
    if value is None:
        return None
    s = str(value).strip()
    s = re.sub(r"\s+", " ", s)
    return s or None


def clean_model_name(value):
    if value is None:
        return None
    s = str(value).strip()
    s = s.replace("\\", "/")
    s = re.sub(r"\s+", "", s)
    s = s.strip("_-=:")
    return s or None


# =========================================================
# EXTRACTION DATASET / VERSION
# =========================================================
def infer_dataset_and_version_from_path(summary_path: Path, root_path: Path):
    """
    Structure attendue :
    root_path / <dataset> / <version> / summary.json
    ou
    root_path / <dataset> / <version> / summary_cont.json
    """
    try:
        rel = summary_path.relative_to(root_path)
        parts = rel.parts
        if len(parts) >= 3:
            dataset = parts[0]
            version = parts[1].lower()
            return dataset, version
    except Exception:
        pass
    return None, None


# =========================================================
# EXTRACTION LLMs
# =========================================================
LLM1_KEYS = [
    "llm_phase1", "phase1_llm", "llm1", "teacher_llm", "gen_llm",
    "embed_llm", "train_llm", "llm_tr", "llm_train", "t"
]

LLM2_KEYS = [
    "llm_phase2", "phase2_llm", "llm2", "student_llm", "pred_llm",
    "pair_llm", "eval_llm", "llm_pred", "llm_pe", "p"
]


def find_value_by_keys(flat, possible_keys):
    normalized = {k.lower(): v for k, v in flat.items()}

    for wanted in possible_keys:
        wanted = wanted.lower()
        for full_k, v in normalized.items():
            last = full_k.split(".")[-1]
            if last == wanted or full_k == wanted:
                if isinstance(v, (str, int, float, bool)):
                    return str(v)
    return None


def infer_llms_from_text(text: str):
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

    return clean_model_name(llm1), clean_model_name(llm2)


def extract_llms(data):
    flat = flatten_json(data)
    llm1 = find_value_by_keys(flat, LLM1_KEYS)
    llm2 = find_value_by_keys(flat, LLM2_KEYS)

    text = json.dumps(data, ensure_ascii=False)
    llm1_txt, llm2_txt = infer_llms_from_text(text)

    llm1 = clean_model_name(llm1) or llm1_txt
    llm2 = clean_model_name(llm2) or llm2_txt

    return llm1, llm2


# =========================================================
# EXTRACTION DES GRANULARITES
# =========================================================
def looks_like_granularity_key(key: str) -> bool:
    k = key.lower()

    keywords = [
        "granularity",
        "granularities",
        "granularite",
        "granularites",
        "grain",
        "cluster_size",
        "cluster_sizes",
        "num_clusters",
        "n_clusters",
        "levels",
        "level",
        "k",
    ]
    return any(word in k for word in keywords)


def looks_like_granularity_value(value):
    return isinstance(value, (int, float, str, list, dict, bool))


def extract_granularities(data):
    flat = flatten_json(data)
    found = {}

    for k, v in flat.items():
        last = k.split(".")[-1]
        if looks_like_granularity_key(last) or looks_like_granularity_key(k):
            if looks_like_granularity_value(v):
                found[k] = v

    return found


def summarize_granularities(granularities: dict):
    if not granularities:
        return "aucune granularité reconnue"

    lines = []
    for k, v in sorted(granularities.items()):
        if isinstance(v, (dict, list)):
            try:
                v_str = json.dumps(v, ensure_ascii=False)
            except Exception:
                v_str = str(v)
        else:
            v_str = str(v)
        lines.append(f"  - {k}: {v_str}")
    return "\n".join(lines)


# =========================================================
# ANALYSE D'UN FICHIER SUMMARY
# =========================================================
def analyze_summary_file(summary_path: Path, root_path: Path, result_type: str):
    data = safe_load_json(summary_path)
    if data is None:
        return None

    dataset, version = infer_dataset_and_version_from_path(summary_path, root_path)
    if not dataset or version not in EXPECTED_VERSIONS:
        return None

    llm1, llm2 = extract_llms(data)
    granularities = extract_granularities(data)

    return {
        "dataset": dataset,
        "version": version,
        "type_resultat": result_type,
        "llm_phase1": llm1,
        "llm_phase2": llm2,
        "file": str(summary_path),
        "granularities": granularities,
    }


# =========================================================
# RECHERCHE ROBUSTE DU FICHIER SUMMARY
# =========================================================
def find_existing_summary_file(version_dir: Path, candidate_names):
    for filename in candidate_names:
        path = version_dir / filename
        if path.exists() and path.is_file():
            return path
    return None


# =========================================================
# COLLECTE
# =========================================================
def collect_from_config(run_config):
    results = []

    root_path = run_config["root"]
    result_type = run_config["result_type"]
    summary_filenames = run_config["summary_filenames"]

    if not root_path.exists():
        print(f"[INFO] Racine absente, ignorée : {root_path}")
        return results

    for dataset_dir in sorted(root_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        for version in sorted(EXPECTED_VERSIONS):
            version_dir = dataset_dir / version
            if not version_dir.exists() or not version_dir.is_dir():
                continue

            summary_path = find_existing_summary_file(version_dir, summary_filenames)
            if summary_path is None:
                continue

            result = analyze_summary_file(summary_path, root_path, result_type)
            if result is not None:
                results.append(result)

    return results


def collect_all_results():
    results = []

    for run_config in RUN_CONFIGS:
        results.extend(collect_from_config(run_config))

    return results


# =========================================================
# ECRITURE
# =========================================================
def write_output(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("==== GRANULARITES CONSOLIDEES ====\n")
        f.write(f"Nombre de résultats trouvés : {len(results)}\n\n")

        for i, r in enumerate(results, 1):
            f.write("=" * 100 + "\n")
            f.write(f"RESULTAT #{i}\n")
            f.write(f"DATASET            : {r['dataset']}\n")
            f.write(f"VERSION            : {r['version']}\n")
            f.write(f"TYPE_RESULTAT      : {r['type_resultat']}\n")
            f.write(f"LLM PHASE 1        : {r['llm_phase1'] or 'inconnu'}\n")
            f.write(f"LLM PHASE 2        : {r['llm_phase2'] or 'inconnu'}\n")
            f.write(f"FICHIER            : {r['file']}\n")
            f.write("GRANULARITES       :\n")
            f.write(summarize_granularities(r["granularities"]))
            f.write("\n\n")


# =========================================================
# MAIN
# =========================================================
def main():
    results = collect_all_results()
    write_output(results, OUTPUT_FILE)

    print(f"[OK] Fichier créé : {OUTPUT_FILE}")
    print(f"[OK] Nombre de résultats trouvés : {len(results)}")


if __name__ == "__main__":
    main()