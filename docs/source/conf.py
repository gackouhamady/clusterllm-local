import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "ClusterLLM-Local"
author = "Hamady Gackou"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_tabs.tabs",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# MyST options (Markdown)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "attrs_block",
]
