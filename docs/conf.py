# Configuration file for the Sphinx documentation builder.
# Run with: cd docs && make html
# (or: uv run sphinx-build -b html docs docs/_build/html)

import sys
from pathlib import Path

# Make the source package importable so autodoc can read docstrings
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "pychannel_lab"))

# ── Project info ──────────────────────────────────────────────────────────────
project = "pyChanneLab"
copyright = "2026, Luca Sagresti"
author = "Luca Sagresti"
release = "0.1.0"

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",  # pull docstrings automatically
    "sphinx.ext.napoleon",  # Google / NumPy docstring style
    "sphinx.ext.viewcode",  # add [source] links to rendered docs
    "sphinx.ext.autosummary",  # generate summary tables
    "sphinx.ext.intersphinx",  # cross-link to NumPy / SciPy / Python docs
]

# Napoleon settings (Google-style docstrings preferred)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# autodoc defaults
autodoc_default_options = {
    "members": True,
    "undoc-members": False,  # skip members with no docstring
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

# autosummary: generate stub files automatically
autosummary_generate = True

# intersphinx: link to upstream docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# ── HTML output ───────────────────────────────────────────────────────────────
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "titles_only": False,
}

# ── Misc ──────────────────────────────────────────────────────────────────────
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
