# conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Adjust path to find source code
sys.path.insert(0, os.path.abspath('../..'))  # Adjust path to find source code
sys.path.insert(0, os.path.abspath('../../..'))  # Adjust path to find source code

project = 'funcsim'
author = 'Henry Bryant'
copyright = '2017-2025, Henry Bryant'
release = '0.1.0'

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',  # Optional: renders type hints more cleanly
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'  # or 'sphinx_rtd_theme', 'alabaster', etc.
# html_static_path = ['_static']

napoleon_google_docstring = False  # Make sure Google style is off
napoleon_numpy_docstring = True   # Enable NumPy style