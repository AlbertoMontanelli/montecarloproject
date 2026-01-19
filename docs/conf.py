# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# --- Silence ROOT during documentation build ---
os.environ["ROOTSYS"] = os.environ.get("ROOTSYS", "")
os.environ["CLING_STANDARD_PCH"] = "none"

import warnings
warnings.filterwarnings(
    "ignore",
    message="Particle with PDGcode=.* already defined",
    category=RuntimeWarning,
)

try:
    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kError  # ignore warnings & infos
except Exception:
    pass

project = "montecarloproject"
author = "Alberto Montanelli"
release = "0.1.0"

# --- General configuration ---
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",  # markdown support
    "sphinx.ext.autosummary",  # auto API stubs
]
autoclass_content = "class"
autodoc_class_signature = "separated"

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "private-members": False,
}
todo_include_todos = True

# Options for syntax highlighting.
pygments_style = "default"
pygments_dark_style = "default"

# Options for internationalization.
language = "en"

# Options for markup.
rst_prolog = """
.. |Python| replace:: `Python <https://www.python.org/>`__
.. |Sphinx| replace:: `Sphinx <https://www.sphinx-doc.org/en/master/>`__
.. |numpy| replace:: `NumPy <https://numpy.org/>`__
.. |GitHub| replace:: `GitHub <https://github.com/>`__
"""

# Options for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Options for templating.
templates_path = ["_templates"]

myst_enable_extensions = ["colon_fence", "dollarmath"]

# --- Options for HTML output ---
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_theme_options = {
    "awesome_external_links": True,
}
html_permalinks_icon = "<span>#</span>"
html_static_path = ["_static"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
