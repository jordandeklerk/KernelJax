"""Sphinx configuration for the KernelJAX documentation."""

project = "KernelJAX"
copyright = "2026, Jordan DeKlerk"
author = "Jordan DeKlerk"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = []

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "KernelJAX"

autosummary_generate = True
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

myst_enable_extensions = ["linkify", "colon_fence"]
nb_execution_mode = "off"
