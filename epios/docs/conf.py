# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import sphinx
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EpiOS'
copyright = '2023, Yunli Qi, Tom Reed, Antonio Mastromarino'
author = 'Yunli Qi, Tom Reed, Antonio Mastromarino'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon']

# Autodoc defaults
if int(sphinx.__version__.split('.')[1]) < 8:
    autodoc_default_flags = ['members', 'inherited-members']
else:
    autodoc_default_options = {
        'members': None,
        'inherited-members': None,
    }

master_doc = 'index'

# Napoleon settings
napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output ------------------------------------------------- # noqa
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []
