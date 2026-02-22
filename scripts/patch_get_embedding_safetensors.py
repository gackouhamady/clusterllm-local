#!/usr/bin/env python3
import argparse
import pathlib
import re

def main(repo_root: str) -> None:
    root = pathlib.Path(repo_root).resolve()
    path = root / "src/clusterllm/perspective/finetuning/get_embedding.py"
    if not path.exists():
        raise FileNotFoundError(f"get_embedding.py not found: {path}")

    txt = path.read_text(encoding="utf-8")

    # 1) Ensure safetensors import
    if "from safetensors.torch import load_file as safe_load_file" not in txt:
        m = re.search(r"(^\s*import\s+torch\s*$)", txt, flags=re.MULTILINE)
        if m:
            insert_at = m.end()
            txt = txt[:insert_at] + "\nfrom safetensors.torch import load_file as safe_load_file\n" + txt[insert_at:]
        else:
            txt = "from safetensors.torch import load_file as safe_load_file\n" + txt

    # 2) Prefer .bin, fallback to model.safetensors if missing
    pat_state_path = r'(^[ \t]*)state_path\s*=\s*os\.path\.join\(args\.checkpoint,\s*"pytorch_model\.bin"\)'
    m = re.search(pat_state_path, txt, flags=re.MULTILINE)
    if m and "model.safetensors" not in txt:
        indent = m.group(1)
        repl = (
            f'{indent}state_path = os.path.join(args.checkpoint, "pytorch_model.bin")\n'
            f'{indent}if not os.path.exists(state_path):\n'
            f'{indent}    alt = os.path.join(args.checkpoint, "model.safetensors")\n'
            f'{indent}    if os.path.exists(alt):\n'
            f'{indent}        state_path = alt'
        )
        txt = re.sub(pat_state_path, repl, txt, count=1, flags=re.MULTILINE)

    # 3) Load safetensors when applicable
    pat_load = r'(^[ \t]*)state_dict\s*=\s*torch\.load\(state_path,\s*map_location="cpu"\)'
    m = re.search(pat_load, txt, flags=re.MULTILINE)
    if m:
        indent = m.group(1)
        repl = (
            f"{indent}if str(state_path).endswith('.safetensors'):\n"
            f"{indent}    state_dict = safe_load_file(str(state_path))\n"
            f"{indent}else:\n"
            f"{indent}    state_dict = torch.load(state_path, map_location=\"cpu\")"
        )
        txt = re.sub(pat_load, repl, txt, count=1, flags=re.MULTILINE)

    path.write_text(txt, encoding="utf-8")
    print("PATCHED:", path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, required=True)
    args = ap.parse_args()
    main(args.repo_root)