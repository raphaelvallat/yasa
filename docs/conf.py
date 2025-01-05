"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

For the full list of extension configuration values, see respective sites.
"""
import sys
from pathlib import Path

import yasa


# -- Path setup --------------------------------------------------------------

sys.path.append(str(Path("sphinxext").resolve()))


#####################################################################
# -- Core Sphinx settings -------------------------------------------
#####################################################################

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "yasa"
project_alt = project.upper()  # not used by sphinx, just below as a variable
author = "Raphael Vallat"
project_copyright = "2018-%Y, Dr. Raphael Vallat, Center for Human Sleep Science, UC Berkeley"
version = yasa.__version__  # full project version (e.g., 4.2.1b0)
release = version[:version.index(".", 2)]  # major project version (e.g., 4.2)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = "index"
source_suffix = {".rst": "restructuredtext"}
source_encoding = "utf-8"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
add_function_parentheses = False
add_module_names = True
toc_object_entries = True
toc_object_entries_show_parents = "hide"  # domain|hide|all
extensions = [
    "matplotlib.sphinxext.plot_directive",  # includes matplotlib plots
    "notfound.extension",  # adds 404 page
    "numpydoc",  # generates numPy style docstrings
    "sphinx_copybutton",  # adds copy-to-clipboard button on code blocks
    "sphinx_design",  # offers directives for badges, dropdowns, tabs, etc
    "sphinx.ext.autodoc",  # includes documentation from docstrings
    "sphinx.ext.autosummary",  # generates autodoc summaries
    # "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",  # links to other package docs
    "sphinx.ext.mathjax",  # LaTeX math display
    "sphinx.ext.viewcode",  # Provides source links to code
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Built-in sphinx theme configuration options
html_theme = "pydata_sphinx_theme"
html_logo = "https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/yasa_128x128.png"
html_favicon = "https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/favicon.ico"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
# html_title = f"{project} v{version}"  # defaults to "<project> v<revision> documentation"
# html_short_title = f"{project} v{release}"  # used in the navbar if not using pydata logo
html_show_sphinx = False
html_show_copyright = True
html_show_sourcelink = False
html_copy_source = False
html_permalinks = True
html_domain_indices = True
html_use_index = False  # True (default) or False, adds index to the HTML documents
html_sidebars = {
    # "**": ["localtoc.html", "globaltoc.html", "searchbox.html"],
    "**": [],  # remove sidebar from all pages
    # "api": [],
    # "quickstart": [],  # remove sidebar from quickstart page
    # "faq": [],  # remove sidebar from FAQ page
    # "contributing": [],  # remove sidebar from contributing page
    # "changelog": [],  # remove sidebar from changelog page
    # "index": ["sidebar-quicklinks.html"],
}
html_context = {
    # "github_url": "https://github.com",
    "github_user": "raphaelvallat",
    "github_repo": "yasa",
    "github_version": "master",
    "doc_path": "doc",
    "default_mode": "dark",  # auto|light|dark
}

# -- Options for HTML output (PyData theme) ----------------------------------
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html#references

html_theme_options = {
    # This overwrites the `pygments_style` set in prior sphinx configuration.
    # This is overwritten if specified below for light and dark modes separately.
    "pygments_style": "tango",  # 'tango' (default) | other pygments style

    # General configuration

    "sidebar_includehidden": True, # True (default) | False
    "use_edit_page_button": False,  # True | False (default)
    "external_links": [],
    # "github_url": "",
    "icon_links_label": "Quick Links",  # accessibility feature, 'Quick Links' by default
    "icon_links": [
        {
            "name": f"{project} on GitHub",  # text that shows on hover
            "url": "https://github.com/raphaelvallat/yasa",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "show_prev_next": False,  # True (default) | False
    "search_bar_text": "Search",  # defaults to "Search the docs ..."
    "navigation_with_keys": False,  # True | False (default)
    "collapse_navigation": True,  # True | False (default)
    "navigation_depth": 2,  # defaults to 4
    "show_nav_level": 2,  # defaults to 1
    "show_toc_level": 2,  # defaults to 1
    "navbar_align": "left",  # content (default) | left | right
    "header_links_before_dropdown": 5,  # defaults to 5
    "header_dropdown_text": "More",  # defaults to 'More'
    "pygments_light_style": "github-light-colorblind",  # defaults to 'a11y-high-contrast-light'
    "pygments_dark_style": "github-dark-colorblind",  # defaults to 'a11y-high-contrast-dark'
    "logo": {
        "alt_text": f"{project} - Home",  # read first by screen readers
        "text": f"{project} v{release}",  # optional text placed alongside logo image
        # If these are the same, can just set one with sphinx's `html_logo`
        "image_light": "https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/yasa_128x128.png",
        "image_dark": "https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/yasa_128x128.png",
        # "link": "",  # optional URL location to override default of index
    },
    # "link": "",  # optional URL location to override default of index
    "surface_warnings": True,  # True (default) | False
    "back_to_top_button": False,  # True (default) | False

    # Template placement in theme layouts
    # See list of built-in components that can be inserted in these sections:
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html#built-in-components-to-insert-into-sections
    # Can specify a list of items to include in all sidebars,
    # or a dictionary to specify items to include in sidebars of specific pages.
    # "secondary_sidebar_items": {"**": []},

    # Navigation bar aka header
    "navbar_start": ["navbar-logo"],  # defaults to ['navbar-logo']
    "navbar_center": ["navbar-nav"],  # defaults to ['navbar-nav']
    "navbar_end": ["theme-switcher", "navbar-icon-links"],  # defaults to ['theme-switcher', 'navbar-icon-links']
    "navbar_persistent": ["search-button"],  # defaults to ['search-button-field']
    # Article/content
    "article_header_start": [],  # defaults to ['breadcrumbs']
    "article_header_end": [],  # defaults to []
    "article_footer_items": [],  # defaults to []
    "content_footer_items": [],  # defaults to []
    # Primary sidebar (left side)
    "primary_sidebar_end": [],  # defaults to ["sidebar-ethical-ads"]
    # Footer
    "footer_start": ["copyright"],  # defaults to ['copyright', 'sphinx-version']
    "footer_center": [],  # defaults to []
    "footer_end": [],  # defaults to ['theme-version']
    # Secondary sidebar (right side)
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],  # defaults to ['page-toc', 'edit-this-page', 'sourcelink']
    # Announcement banner
    "show_version_warning_banner": True,  # True | False (default)
    "announcement": "<span style='font-family: Consolas, monospace;'>pip install yasa --upgrade</span> &#127881;"
}


#####################################################################
# -- Core extension settings ----------------------------------------
#####################################################################

# -- Options for sphinx.ext.autodoc ------------------------------------------

# autodoc_default_options = {
#     "members": True,
#     "member-order": "groupwise",
#     "undoc-members": False,
#     # "members": "var1, var2",
#     # "member-order": "bysource",
#     # "special-members": "__init__",
#     # "undoc-members": None,
#     # "exclude-members": "__weakref__"
#     # "private-members": "__init__",
#     # "inherited-members": "__init__",
#     # "show-inheritance":
#     # "ignore-module-all":
# }

# # -- Options for sphinx.ext.autosectionlabel ---------------------------------

# autosectionlabel_prefix_document = True  # to make sure each target is unique
# # autosectionlabel_maxdepth = 1

# -- Options for sphinx.ext.autosummary --------------------------------------

# autosummary_generate = True  # generate the API documentation when building
# autoclass_content = "class"  # class (default), init, both
# # autodoc_typehints = "description"
# # autodoc_member_order = "groupwise"  # alphabetical (default), groupwise, bysource

# -- Options for sphinx.ext.intersphinx --------------------------------------

intersphinx_mapping = {
    "antropy": ("https://raphaelvallat.com/antropy/build/html", None),
    "lightgbm": ("https://lightgbm.readthedocs.io/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "mne": ("https://mne.tools/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pyriemann": ("https://pyriemann.readthedocs.io/en/latest", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "seaborn": ("https://seaborn.pydata.org", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "tensorpac": ("https://etiennecmb.github.io/tensorpac", None),
}


#####################################################################
# -- External extension settings ------------------------------------
#####################################################################

# -- Options for sphinx-copybutton -------------------------------------------
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True  # JavaScript regular expression
copybutton_only_copy_prompt_lines = True
copybutton_copy_empty_lines = True

# -- Options for matplotlib.sphinxext.plot_directive extension ---------------
# https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html#configuration-options

plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False

# -- Options for numpydoc extension -----------------------------------------
# https://numpydoc.readthedocs.io/en/latest/install.html#configuration

numpydoc_use_plots = True
numpydoc_show_class_members = False
numpydoc_class_members_toctree = True
numpydoc_attributes_as_param_list = True

# # -- Options for myst-nb -----------------------------------------------------

# # nb_execution_excludepatterns = ['list', 'of', '*patterns']
# nb_execution_cache_path = "path/to/mycache"
# nb_execution_mode = "off"  # off, force, auto, cache, inline
