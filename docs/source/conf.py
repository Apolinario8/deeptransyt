# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',       # Auto-generate API documentation from docstrings
    'sphinx.ext.napoleon',      # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',      # Add links to source code in the documentation
    'sphinx.ext.githubpages'    # Adds .nojekyll file for GitHub Pages compatibility
]

# -- Paths setup -------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Ensure Sphinx can find your package

# -- HTML theme --------------------------------------------------------------

html_theme = "sphinx_rtd_theme"  # Use Read the Docs theme

templates_path = ['_templates']
exclude_patterns = []




# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
