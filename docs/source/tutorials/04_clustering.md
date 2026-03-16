# Text Clustering and Granularity Selection

The clustering process follows a two-stage approach: improving Perspective and selecting the optimal Granularity.

## Stage 1: Perspective Learning

1. 
**Informative Sampling**: Instead of random selection, we sample high-entropy triplets using provisional clusters to identify ambiguous data points.


2. 
**LLM Preferences**: The local Triplet LLM decides which candidates are semantically closer to an anchor based on user instructions.


3. 
**Fine-Tuning**: The base embedder (Instructor-Large) is fine-tuned for one epoch using these preferences to align its representation space with the user's view.



## Stage 2: Granularity Selection

1. 
**Hierarchy Building**: An agglomerative clustering hierarchy is built from refined embeddings.


2. 
**Pairwise Judgments**: The local Pairwise LLM answers "Yes/No" questions about whether two samples belong to the same group.


3. 
**Consistency Maximization**: The final number of clusters ($\hat{K}$) is determined by selecting the hierarchy cut that maximizes consistency with the LLM's pairwise judgments.



## Evaluation Metrics

We report standard clustering performance metrics:

* **ACC**: Accuracy
* **NMI**: Normalized Mutual Information
* **ARI**: Adjusted Rand Index

---