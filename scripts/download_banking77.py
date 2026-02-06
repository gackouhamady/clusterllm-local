#!/usr/bin/env python3
import os, json
from datasets import load_dataset

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    out_dir = os.path.join("datasets", "banking77")
    ensure_dir(out_dir)

    print("📥 Downloading Hugging Face dataset: banking77 ...")
    ds = load_dataset("banking77")  # train + test

    train_rows = [{"text": x["text"], "label": int(x["label"])} for x in ds["train"]]
    test_rows  = [{"text": x["text"], "label": int(x["label"])} for x in ds["test"]]

    write_jsonl(train_rows, os.path.join(out_dir, "train.jsonl"))
    write_jsonl(test_rows,  os.path.join(out_dir, "test.jsonl"))
    write_jsonl(train_rows + test_rows, os.path.join(out_dir, "full.jsonl"))

    small_n = 2000 if len(train_rows) >= 2000 else len(train_rows)
    write_jsonl(train_rows[:small_n], os.path.join(out_dir, "small.jsonl"))

    print("✅ Done. Files created:")
    for fn in ["train.jsonl","test.jsonl","full.jsonl","small.jsonl"]:
        print(" -", os.path.join(out_dir, fn))

if __name__ == "__main__":
    main()
