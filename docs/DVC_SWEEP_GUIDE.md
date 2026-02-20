# 🚀 DVC Sweep Guide – Full Pipeline (2 LLMs)

Ce guide explique comment :

* Mettre en queue 100 scénarios
* Les exécuter en parallèle
* Lancer un seul scénario
* Voir les résultats
* Nettoyer la queue
* Lancer un scénario spécifique parmi tous

---

# 📂 Prérequis

Structure attendue :

```
scripts/
  run_full_pipeline_one_dataset_two_llms.sh
  sweep_full_pipeline_2llms.sh

configs/
  grid_small.txt
  grid_large.txt
```

Stage DVC attendu dans `dvc.yaml` :

```
full_pipeline_2llms
```

---

# 🧩 Format du fichier grid

`configs/grid_small.txt`

Chaque ligne = 1 scénario :

```
# dataset scale llm_triplet llm_pairs seed
clinc150 small deepseek-r1:32b qwen2.5:32b 42
bank77 small llama3.2:3b-instruct-q8_0 qwen2.5:7b 42
reddit small deepseek-r1:32b llama3.1:8b-instruct-q8_0 42
```

* 5 colonnes obligatoires
* Les lignes vides sont ignorées
* Les lignes commençant par `#` sont ignorées

---

# 🔁 Commandes principales

---

## A) Mettre en queue les 100 scénarios (sans lancer)

```bash
bash scripts/sweep_full_pipeline_2llms.sh queue configs/grid_100.txt
```

Cela :

* Ajoute tous les scénarios à la queue DVC
* Ne lance rien

---

## B) Lancer tous les scénarios en parallèle (4 jobs)

```bash
bash scripts/sweep_full_pipeline_2llms.sh run configs/grid_small.txt 4
```

Cela :

1. Met en queue tous les scénarios
2. Lance tous les runs en parallèle avec 4 workers

---
- Ceci va de  meme  pour  le  configs/grid_large.txt


## C) Lancer un seul scénario “à la main”

```bash
dvc exp run -s full_pipeline_2llms \
  -S run.dataset=clinc150 \
  -S run.scale=small \
  -S run.llm_triplet=deepseek-r1:32b \
  -S run.llm_pairs=qwen2.5:32b \
  -S run.seed=42
```

---

## D) Voir les résultats

```bash
bash scripts/sweep_full_pipeline_2llms.sh show
```

Équivalent à :

```bash
dvc exp show
```

---

## E) Nettoyer la queue

```bash
bash scripts/sweep_full_pipeline_2llms.sh clean
```

---

# 🎯 Lancer un seul scénario parmi les 100

---

## Option 1 – Créer un fichier temporaire

Créer :

```
configs/one_small.txt
```

Contenu :

```
clinc150 small deepseek-r1:32b qwen2.5:32b 42
```

Puis :

```bash
bash scripts/sweep_full_pipeline_2llms.sh run configs/one_small.txt 4
```
- Meme   pour  configs/one_large.txt
---

## Option 2 – Modifier temporairement grid_small.txt

Supprimer toutes les lignes sauf celle voulue, puis :

```bash
bash scripts/sweep_full_pipeline_2llms.sh run configs/grid_small.txt 4
```
- Meme  pour   configs/one_large.txt
---

# 🧠 Workflow recommandé

### 1️⃣ Générer ou modifier grid

```
configs/grid_small.txt
- ou configs/grid_small.txt
```

### 2️⃣ Mettre en queue

```bash
bash scripts/sweep_full_pipeline_2llms.sh queue configs/grid_small.txt
# or 
bash scripts/sweep_full_pipeline_2llms.sh queue configs/grid_large.txt

```

### 3️⃣ Lancer en parallèle

```bash
dvc exp run --run-all --jobs 4
```

### 4️⃣ Voir résultats

```bash
dvc exp show
```

---

# ⚡ Exemple réel

```bash
bash scripts/sweep_full_pipeline_2llms.sh run configs/grid_small.txt 4
# or 
bash scripts/sweep_full_pipeline_2llms.sh run configs/grid_large.txt 4
```

Cela :

* Enqueue 100 configs
* Lance 4 en parallèle
* Continue jusqu’à fin
* Tous les outputs sont dans `runs/`

---

# 📌 Résumé

| Action       | Commande                                 |
| ------------ | ---------------------------------------- |
| Queue        | `sweep_full_pipeline_2llms.sh queue`     |
| Run parallel | `sweep_full_pipeline_2llms.sh run`       |
| Single run   | `dvc exp run -s full_pipeline_2llms ...` |
| Show results | `dvc exp show`                           |
| Clean queue  | `sweep_full_pipeline_2llms.sh clean`     |

---

# ✅ Bonnes pratiques

* Toujours vérifier que le script accepte bien les args
* Ne pas écraser les variables après le bloc args
* Utiliser `--jobs` adapté à ta VRAM
* Ne pas mettre `runs/` en cache DVC si volumineux

---