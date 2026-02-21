# 1️⃣ Fix Geometry for Text (Cosine Instead of Raw Euclidean)

##  Problem

Ward + KMeans use **Euclidean distance**, but text embeddings behave better in **angular (cosine) space**.

---

##  Stage 1 & Evaluation

###  File

```
src/clusterllm/clustering_utils/evaluator.py
```

###  Change

Inside `ClusteringEvaluator.__call__` (right after embeddings are computed):

```python
corpus_embeddings = np.asarray(model.encode(new_sentences))
```

Replace with:

```python
corpus_embeddings = np.asarray(
    model.encode(new_sentences, normalize_embeddings=True)
)
```

OR normalize manually:

```python
corpus_embeddings = corpus_embeddings / (
    np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-12
)
```

---

##  Expected Gain

* +1–4% ARI/NMI typical for text datasets
* More stable granularity tree
* Better separation for semantic clusters

---

# 2️⃣ Replace Ward Hierarchy (Granularity Stage)

##  Problem

Ward linkage assumes Euclidean variance minimization → wrong geometry for cosine embeddings.

---

##  Stage 7 (Granularity Hierarchy)

###  File

Find where hierarchy is built (likely in):

```
src/clusterllm/granularity/
```

Look for:

```python
AgglomerativeClustering(linkage="ward")
```

---

###  Replace With

```python
AgglomerativeClustering(
    n_clusters=None,
    metric="cosine",
    linkage="average",
    distance_threshold=0
)
```

---

##  Expected Gain

* More meaningful dendrogram
* Better cluster merging decisions
* Improved final K selection stability

---

# 3️⃣ Improve Triplet Sampling Geometry (Stage 1 Perspective)

##  Problem

Student-t soft assignment uses squared Euclidean norm.

Paper uses:

$$ [
||z_i - \mu_k||^2
] $$

Not ideal for normalized embeddings.

---

## Stage 2 (Triplet Sampling)

###  File

```
src/clusterllm/perspective/predict_triplet/triplet_sampling.py
```

Find where distances to centroids are computed.

---

### 🔄 Replace Euclidean Distance With Cosine

Instead of:

```python
dist = np.linalg.norm(z_i - mu_k)**2
```

Use:

```python
dist = 1 - np.dot(z_i, mu_k)
```

(Ensure embeddings are normalized.)

---

##  Expected Gain

* Better entropy ranking
* More informative anchors
* Less noisy LLM supervision

---

#  4️⃣ Make Fine-Tuning Loss More Robust to LLM Noise

## Problem

Softmax contrastive loss is sensitive to noisy positives.

---

##  Stage 5 (Finetune)

### File

```
src/clusterllm/perspective/finetuning/finetune.py
```

---

###  Add Label Smoothing

Instead of strict positive probability = 1:

```python
loss = CrossEntropy(...)
```

Use:

```python
loss = CrossEntropy(label_smoothing=0.1)
```

OR scale negatives:

$$ [
\ell = -\log \frac{e^{s(a,p)/\tau}}
{e^{s(a,p)/\tau} + \alpha \sum e^{s(a,n)/\tau}}
] $$

with α = 0.7–0.9.

---

## Expected Gain

* More stable training
* Better robustness to LLM mistakes
* Improved downstream clustering

---

# 5️⃣ Reduce LLM Bias in Pair Judgments

##  Problem

LLM pair answers are noisy.

---

##  Stage 7 Pair Queries

### File

```
src/clusterllm/granularity/predict_pairs.py
```

---

###  Add Majority Voting

For each pair:

* Query 3 times
* Shuffle order of options
* Use majority decision

---

## Expected Gain

* Less variance in granularity decision
* Better cluster count estimation

---

#  6️⃣ Optional: Replace KMeans Entirely (Strong Upgrade)

If you want a publishable improvement:

### Replace KMeans with:

* **Spherical KMeans**
* OR **Spectral Clustering (cosine graph)**

Modify inside:

```
_clustering_utils/evaluator.py
```

---