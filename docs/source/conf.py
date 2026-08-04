"""Sphinx configuration for the KernelJax documentation."""

project = "KernelJax"
copyright = "2026, Jordan DeKlerk"
author = "Jordan DeKlerk"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_immaterial",
]

exclude_patterns = []
templates_path = ["_templates"]

html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "KernelJax"
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "repo_url": "https://github.com/jordandeklerk/KernelJax",
    "repo_name": "KernelJax",
    "icon": {"repo": "fontawesome/brands/github"},
    "features": [
        "navigation.sections",
        "navigation.expand",
        "navigation.instant",
        "navigation.top",
        "navigation.tracking",
        "search.highlight",
        "search.share",
        "toc.follow",
        "content.code.copy",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "blue",
            "accent": "red",
            "toggle": {"icon": "material/weather-night", "name": "Switch to dark mode"},
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "blue",
            "accent": "red",
            "toggle": {"icon": "material/weather-sunny", "name": "Switch to light mode"},
        },
    ],
}

autosummary_generate = True

# WLS/wls and Summary/summary differ only in case, which collide as stub filenames
# on a case-insensitive filesystem.
autosummary_filename_map = {
    "kerneljax.wls": "kerneljax.wls-function",
    "kerneljax.summary": "kerneljax.summary-function",
}

autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

myst_enable_extensions = ["linkify", "colon_fence", "dollarmath"]
myst_heading_anchors = 3
nb_execution_mode = "off"
