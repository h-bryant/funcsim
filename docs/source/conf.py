# Sphinx configuration for the funcsim documentation.
#
# Build locally from the repository root with
#     sphinx-build -W -b html docs/source docs/build
# Read the Docs runs the same configuration (see ../../.readthedocs.yaml).

import os
import sys
import typing

# make the funcsim package in this repository importable, so that autodoc
# documents the working tree rather than any installed copy
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

import funcsim  # noqa: E402  (after sys.path manipulation)

project = 'funcsim'
author = 'Henry Bryant'
copyright = '2017-2026, Henry Bryant'

# single source of truth for the version: funcsim.version()
release = funcsim.version()
version = ".".join(release.split(".")[:2])

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = []

# -- autodoc / napoleon ------------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
# the long conversions.VectorLike / ArrayLike unions render as unreadable
# walls of text; show them by their alias names instead
autodoc_type_aliases = {
    'VectorLike': 'funcsim.conversions.VectorLike',
    'ArrayLike': 'funcsim.conversions.ArrayLike',
}
typehints_fully_qualified = False
always_document_param_types = False
typehints_defaults = 'comma'


def _alias_formatter(annotation, config):
    """Render the conversions.VectorLike / ArrayLike unions by name.

    autodoc_type_aliases only applies to string (PEP 563) annotations, and
    sphinx_autodoc_typehints expands the aliases into their member types
    when it writes a "Return type" line.  typing also flattens a union that
    contains an alias, as in the return type of funcsim.shrink, so the alias
    is recovered from the member set.  Returning None falls back to the
    extension's default formatting.
    """
    from sphinx_autodoc_typehints import format_annotation
    from funcsim import conversions

    if typing.get_origin(annotation) is not typing.Union:
        return None
    args = typing.get_args(annotation)
    # VectorLike is a superset of ArrayLike, so it is tested first
    for name in ("VectorLike", "ArrayLike"):
        members = set(typing.get_args(getattr(conversions, name)))
        if members <= set(args):
            rest = [a for a in args if a not in members]
            parts = [f"``{name}``"] + [format_annotation(a, config)
                                       for a in rest]
            return " | ".join(parts)
    return None


typehints_formatter = _alias_formatter
# a callable configuration value cannot be pickled into the environment
# cache; Sphinx warns about that, and -W would turn the warning into an error
suppress_warnings = ['config.cache']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
}

# -- HTML output -------------------------------------------------------------

html_theme = 'alabaster'
html_title = f"funcsim {release} documentation"
html_theme_options = {
    'description': f"Functional, simple stochastic simulation. "
                   f"Version {release}.",
    'github_user': 'h-bryant',
    'github_repo': 'funcsim',
    'github_button': False,
    'github_banner': False,
    'page_width': '1000px',
    'sidebar_width': '240px',
}
html_sidebars = {
    '**': ['about.html', 'navigation.html', 'searchbox.html'],
}
