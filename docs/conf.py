"""Sphinx configuration."""
import datetime

project = 'validator'
author = 'Ethan Welty'
copyright = f'{datetime.date.today().year}, {author}'
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]
html_theme = 'sphinx_rtd_theme'
autodoc_member_order = 'bysource'
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
html_static_path = ['_static']
html_css_files = ['custom.css']
copybutton_exclude = '.linenos, .gp, .go'
