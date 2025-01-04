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
author = "Raphael Vallat"
project_copyright = "2018-%Y, Dr. Raphael Vallat, Center for Human Sleep Science, UC Berkeley"
# Note: `sphinx` will replace %Y with the current year
# Note: `project_copyright` is an alias for `copyright`
#       to avoid overriding the built-in `copyright`
version = yasa.__version__  # The full (long) project version, like "4.2.1b0"
release = version[:version.index(".", 2)]  # The major (short) project version, like "4.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = "index"
source_suffix = {".rst": "restructuredtext"}
source_encoding = "utf-8"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"  # syntax highlighting style
add_function_parentheses = False
add_module_names = True
toc_object_entries = True
toc_object_entries_show_parents = "hide"  # domain|hide|all
extensions = [
    "matplotlib.sphinxext.plot_directive",  # include matplotlib plots
    "numpydoc",  # generates numPy style docstrings
    "sphinx_copybutton",  # adds copy-to-clipboard button on code blocks
    "sphinx_design",  # offers directives for badges, dropdowns, tabs, etc
    "sphinx.ext.autodoc",  # includes documentation from docstrings
    "sphinx.ext.autosummary",  # generates autodoc summaries
    # "sphinx.ext.doctest",  # marks docstrings examples as tests
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
html_css_files = []
html_title = f"YASA v{release}"  # defaults to "<project> v<revision> documentation"
html_short_title = "YASA"  # used in the navbar
html_last_updated_fmt = "%b, %Y"  # empty string is equivalent to "%b %d, %Y"
html_permalinks = True
html_domain_indices = True
html_use_index = False
html_copy_source = False
html_show_sourcelink = True
html_show_copyright = True
html_show_sphinx = False
html_output_encoding = "utf-8"
html_sidebars = {
    # "**": ["localtoc.html", "globaltoc.html", "searchbox.html"],
    # "**": [],  # remove sidebar from all pages
    "api": [],
    "quickstart": [],  # remove sidebar from quickstart page
    "faq": [],  # remove sidebar from FAQ page
    "contributing": [],  # remove sidebar from contributing page
    "changelog": [],  # remove sidebar from changelog page
    # "index": ["sidebar-quicklinks.html"],
}
html_context = {
    # "github_url": "https://github.com",
    "github_user": "raphaelvallat",
    "github_repo": "yasa",
    "github_version": "master",
    "doc_path": "doc",
    "default_mode": "auto",  # light, dark, auto
}


# -- Options for HTML output (PyData theme) ----------------------------------
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html#references

html_theme_options = {
    "logo": {
        "text": "YASA",
        "image_light": "https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/yasa_128x128.png",
        "image_dark": "https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/yasa_128x128.png",
        "alt_text": "YASA homepage",
        # "url": "index",
    },
    "header_links_before_dropdown": 5,
    "navigation_with_keys": False,
    # "external_links": [{"name": "Releases", "url": "https://github.com/raphaelvallat/yasa/releases"}],
    "show_prev_next": False,
    "back_to_top_button": True,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    # "navbar_persistent": [],  # Default is a nice search bubble that I otherwise don't get
    "navbar_persistent": ["search-button"],
    "navbar_align": "left",  # left/content/right
    # "search_bar_text": "Search...",
    "article_header_start": [],  # disable breadcrumbs
    # "article_header_start": ["breadcrumbs"],
    # "article_header_end": [],
    # "article_footer_items": [],
    "footer_start": ["copyright"],  # "search-field" "search-button"
    "footer_center": [],
    "footer_end": ["last-updated"],  # "theme-switcher"
    "content_footer_items": [],
    # "sidebarwidth": 230,
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    # "secondary_sidebar_items": {"**": []},
    # "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "show_version_warning_banner": True,
    "announcement": "<a href='https://raphaelvallat.com/yasa/build/html/changelog.html#v0-7'>v0.7 released!</a> &#127881;<br><span style='font-family: Consolas, monospace;'>pip install yasa --upgrade</span>",
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navigation_depth": 2,
    "collapse_navigation": True,
    "use_edit_page_button": False,
    # "use_repository_button": True,
    # "icon_links_label": "Quick Links",
    "pygments_light_style": "vs",
    "pygments_dark_style": "monokai",
    "icon_links": [
        {
            "name": "YASA on GitHub",  # text that shows on hover
            "url": "https://github.com/raphaelvallat/yasa",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
}


#####################################################################
# -- Core extension settings ----------------------------------------
#####################################################################

# -- Options for sphinx.ext.autodoc ------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "groupwise",
    "undoc-members": False,
    # "members": "var1, var2",
    # "member-order": "bysource",
    # "special-members": "__init__",
    # "undoc-members": None,
    # "exclude-members": "__weakref__"
    # "private-members": "__init__",
    # "inherited-members": "__init__",
    # "show-inheritance":
    # "ignore-module-all":
}

# # -- Options for sphinx.ext.autosectionlabel ---------------------------------

# autosectionlabel_prefix_document = True  # to make sure each target is unique
# # autosectionlabel_maxdepth = 1

# -- Options for sphinx.ext.autosummary --------------------------------------

autosummary_generate = True  # generate the API documentation when building
autoclass_content = "class"  # class (default), init, both
# autodoc_typehints = "description"
# autodoc_member_order = "groupwise"  # alphabetical (default), groupwise, bysource

# -- Options for sphinx.ext.intersphinx --------------------------------------

intersphinx_mapping = {
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

numpydoc_show_class_members = False
numpydoc_class_members_toctree = True
numpydoc_attributes_as_param_list = True

# # -- Options for myst-nb -----------------------------------------------------

# # nb_execution_excludepatterns = ['list', 'of', '*patterns']
# nb_execution_cache_path = "path/to/mycache"
# nb_execution_mode = "off"  # off, force, auto, cache, inline
