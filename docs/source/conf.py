import os
import sys
from unittest.mock import MagicMock

# FIX: On remonte de deux niveaux pour que Sphinx voie 'src' comme un package
# Cela permet de faire 'from clusterllm import ...' sans erreur
sys.path.insert(0, os.path.abspath("../.."))

# --- Informations du Projet ---
project = "ClusterLLM-Local"
author = "Hamady Gackou"
copyright = "2026, Hamady Gackou"
release = "0.1.0"

# --- Extension des Mocks (Pour supprimer TOUS les warnings d'import) ---
# Nous incluons ici toutes les dépendances lourdes, système, ou conflictuelles
autodoc_mock_imports = [
    "torch",
    "torch.utils",
    "torch.utils.data",
    "transformers",
    "transformers.utils",
    "datasets",
    "h5py",
    "numpy",
    "pandas",
    "sklearn",
    "sklearn.cluster",
    "sklearn.metrics",
    "scipy",
    "sentence_transformers",
    "InstructorEmbedding",
    "safetensors",
    "pynvml",
    "Cython",
    "mlflow",
    # Mocks pour vos modules locaux utilitaires qui posent problème lors du build
    "tools",
    "clustering_utils",
    "e5_utils",
    "hierarchy",
]

# --- Configuration des Extensions ---
extensions = [
    "sphinx.ext.autodoc",      # Extraction des docstrings
    "sphinx.ext.napoleon",     # Support format Google/NumPy
    "sphinx.ext.viewcode",     # Liens vers le code source
    "myst_parser",             # Support des fichiers Markdown .md
    "sphinx_copybutton",       # Bouton copier pour les blocs de code
    "sphinx_design",           # Composants UI (grilles, boutons)
    "sphinx_tabs.tabs",        # Onglets
    "sphinxcontrib.mermaid",   # Graphiques Mermaid
]

# Configuration Markdown + Mathématiques
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "tasklist",
    "attrs_block",
]

# Paramètres HTML
html_theme = "furo"
html_static_path = ["_static"]

# Paramètres Autodoc
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}