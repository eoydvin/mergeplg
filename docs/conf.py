from __future__ import annotations

import importlib.metadata

project = "mergeplg"
copyright = "2024, Christian Chwala"
author = "Christian Chwala"
release = importlib.metadata.version("mergeplg")

# Keep the sidebar header clean by showing only the project name. The full
# build version is available via ``release`` and shown on the index page.
version = ""
html_title = "mergeplg"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "nbsphinx",
]

autosummary_generate = True

# Custom autosummary templates (docs/_templates/autosummary/*.rst) add
# ``:toctree:`` to the functions/classes/attributes/exceptions blocks so that
# per-member detail pages are generated (the default module.rst template only
# adds ``:toctree:`` to the submodules block).
templates_path = ["_templates"]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "furo"

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True
