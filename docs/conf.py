# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyspoc'
copyright = '2026, Garry Cotton, Hetvi Jethwani'
author = 'Garry Cotton, Hetvi Jethwani'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]
root_doc = "index"

templates_path = ['_templates']
exclude_patterns = ['_build',
                    'Thumbs.db',
                    '.DS_Store',
                    "api/pyspoc.benchmarks*",
                    "api/pyspoc.lib*",
                    "api/pyspoc.debugging*",]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

napoleon_type_aliases = {
    "Dataset": "~pyspoc.dataset.Dataset",
    "Config": "~pyspoc.config.Config",
    "Component": "~pyspoc.base.Component",
    "Statistic": "~pyspoc.statistic.Statistic",
    "PairwiseStatistic": "~pyspoc.statistic.PairwiseStatistic",
    "ReducedStatistic": "~pyspoc.statistic.ReducedStatistic",
    "Calculator": "~pyspoc.calculator.Calculator",
    "np.ndarray": "numpy.ndarray",
    "pd.DataFrame": "pandas.DataFrame",
}

autodoc_inherit_docstrings = True
autodoc_typehints = "signature"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_preprocess_types = True
napoleon_use_param = True
napoleon_use_rtype = True
