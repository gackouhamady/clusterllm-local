#!/usr/bin/env python3
import json, os
import numpy as np
import h5py
from sentence_transformers import SentenceTransformer

IN_PATH  = "datasets/banking77/small.jsonl"
OUT_PATH = "datasets/banking77/small_embeds.hdf5"
MODEL    = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def main():
    texts = []
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    model = SentenceTransformer(MODEL)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with h5py.File(OUT_PATH, "w") as h:
        h.create_dataset("embeddings", data=emb, compression="gzip")

    print("✅ wrote", OUT_PATH, "shape", emb.shape)

if __name__ == "__main__":
    main()
