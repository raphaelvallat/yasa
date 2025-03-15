"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

For the full list of extension configuration values, see respective sites.
"""
# import sys
# from pathlib import Path

import yasa


# -- Path setup --------------------------------------------------------------

# sys.path.append(str(Path("sphinxext").resolve()))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# The documented project's name.
# Defaults to "Project name not set"
project = "yasa"

# The project's author(s).
# Defaults to "Author name not set"
author = "Raphael Vallat"

# A copyright statement or list of statements. '%Y' is replaced with the current four-digit year.
# See also ``html_show_copyright`` in HTML output options.
# Defaults to ""
project_copyright = "2018-%Y, Dr. Raphael Vallat"

# The major project version, used as the replacement for the `|version|` default substitution.
# This may be something like version = '4.2'. The short X.Y version.
# If your project does not draw a meaningful distinction between between
# a 'full' and 'major' version, set both version and release to the same value.
# Defaults to ""
version = yasa.__version__

# The full project version, used as the replacement for the `|release|` default substitution,
# and e.g. in the HTML templates.
# This may be something like release = '4.2.1b0'. The full version, including alpha/beta/rc tags.
# The major (short) project version is defined in version.
# If your project does not draw a meaningful distinction between between
# a 'full' and 'major' version, set both version and release to the same value.
# Defaults to ""
release = version


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# A list of strings that are module names of Sphinx extensions.
# These can be extensions bundled with Sphinx (named sphinx.ext.*)
# or custom first-party or third-party extensions.
extensions = [

    # Extensions bundled with Sphinx
    "sphinx.ext.autodoc",       # Includes documentation from docstrings
    "sphinx.ext.autosummary",   # Generates autodoc summaries
    "sphinx.ext.intersphinx",   # Links to other package docs
    "sphinx.ext.mathjax",       # LaTeX math display
    # "sphinx.ext.viewcode",

    # Third-party extensions
    "matplotlib.sphinxext.plot_directive",  # Includes matplotlib plots
    "notfound.extension",       # Adds 404 page
    "numpydoc",                 # Generates numPy style docstrings (Needs to be loaded *after* autodoc)
    "sphinx_copybutton",        # Adds copy-to-clipboard button in code blocks
    "sphinx_design",            # Adds directives for badges, dropdowns, tabs, etc
    "sphinx_reredirects",       # Generates redirects for moved pages

]

# -- General configuration ---------------------------------------------------
# -- -> Options for internationalisation -------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-internationalisation

# The code for the language the documents are written in.
# Defaults to "en"
language = "en"


# -- General configuration ---------------------------------------------------
# -- -> Options for object signatures ----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-object-signatures

# A boolean that decides whether parentheses are appended to function and method role text
# (e.g. the content of :func:`input`) to signify that the name is callable.
# Options: True (default) | False
add_function_parentheses = False

# If a signature's length in characters exceeds the number set,
# each parameter within the signature will be displayed on an individual logical line.
# Defaults to None
maximum_signature_line_length = None

# When backslash stripping is enabled then every occurrence of \\ in a domain directive
# will be changed to \, even within string literals.
# Options: True | False (default)
strip_signature_backslash = False

# Create table of contents entries for domain objects (e.g. functions, classes, attributes, etc.).
# Options: True (default) | False
toc_object_entries = True

# A string that determines how domain objects (functions, classes, attributes, etc.)
# are displayed in their table of contents entry.
# Options: domain (default) | hide | all
toc_object_entries_show_parents = "hide"

# -- General configuration ---------------------------------------------------
# -- -> Options for source files ---------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-source-files

# A list of glob-style patterns that should be excluded when looking for source files.
# ``exclude_patterns`` has priority over ``include_patterns``
# Defaults to []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# A list of glob-style patterns that are used to find source files.
# ``exclude_patterns`` has priority over ``include_patterns``
# Defaults to ["**"]
include_patterns = ["**"]

# Sphinx builds a tree of documents based on the `toctree` directives contained
# within the source files. This sets the name of the document containing the
# master toctree directive, and hence the root of the entire tree.
# Defaults to "index"
master_doc = "index"

# The file encoding of all source files.
# Defaults to "utf-8-sig"
source_encoding = "utf-8"

# A dictionary mapping the file extensions (suffixes) of source files to their file types.
# Sphinx considers all files files with suffixes in `source_suffix` to be source files.
# If the value is a string or sequence of strings,
# Sphinx will consider that they are all 'restructuredtext' files.
# Defaults to {".rst": "restructuredtext"}
source_suffix = {".rst": "restructuredtext"}


# -- General configuration ---------------------------------------------------
# -- -> Options for the Python domain ----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-the-python-domain

# A boolean that decides whether module names are prepended to all object names
# (for object types where a "module" of some kind is defined), e.g. for py:function directives.
# Options: True (default) | False
add_module_names = True


# -- Builder options ---------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#builder-options

# -- Builder options ---------------------------------------------------------
# -- -> Options for HTML output ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme for HTML output.
# Defaults to "alabaster"
html_theme = "pydata_sphinx_theme"

# A dictionary of options that influence the look and feel of the selected theme. These are theme-specific.
# See here for list of available (key, value) pairs in PyData theme:
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html#references
html_theme_options = {

    # -- General -------------------------------

    # Options: True (default) | False
    "sidebar_includehidden": True,

    # Options: True | False (default)
    "use_edit_page_button": False,

    # Defaults to []
    "external_links": [],

    # Defaults to None
    # "github_url": "",

    # Defaults to "Quick Links"
    "icon_links_label": "Quick Links",

    # Defaults to []
    "icon_links": [
        {
            "name": f"{project} on GitHub",  # text that shows on hover
            "url": "https://github.com/raphaelvallat/yasa",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],

    # Options: True (default) | False
    "show_prev_next": False,

    # Defaults to "Search the docs..."
    "search_bar_text": "Search",

    # Options: True | False (default)
    "navigation_with_keys": False,

    # Options: True | False (default)
    "collapse_navigation": False,

    # Defaults to 4
    "navigation_depth": 2,

    # Defaults to 1
    "show_nav_level": 2,

    # Defaults to 1
    "show_toc_level": 2,

    # Options: content (default) | left | right
    "navbar_align": "left",

    # Defaults to 5
    "header_links_before_dropdown": 5,

    # Defaults to "More"
    "header_dropdown_text": "More",

    # Defaults to a11y-high-contrast-light
    "pygments_light_style": "github-light-colorblind",

    # Defaults to a11y-high-contrast-dark
    "pygments_dark_style": "github-dark-colorblind",

    # Defaults to {}
    "logo": {
        "alt_text": f"{project} - Home",  # read first by screen readers
        "text": f"{project} v{release}",  # optional text placed alongside logo image
        # If these are the same, can just set one with sphinx's `html_logo`
        "image_light": "https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/yasa_128x128.png",
        "image_dark": "https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/yasa_128x128.png",
        # "link": "",  # optional URL location to override default of index
    },

    # Defaults to ""
    # "logo_link": "",  # optional URL location to override default of index

    # Options: True (default) | False
    "surface_warnings": True,

    # The Back to Top button is a floating button that appears when you scroll down a page,
    # and allows users to quickly return to the top of the page.
    # Options: True (default) | False
    "back_to_top_button": False,

    # Template placement in theme layouts
    # See list of built-in components that can be inserted in these sections:
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html#built-in-components-to-insert-into-sections
    # Can specify a list of items to include in all sidebars,
    # or a dictionary to specify items to include in sidebars of specific pages.
    # "secondary_sidebar_items": {"**": []},

    # -- Navigation bar / header ---------------

    # Defaults to ["navbar-logo"]
    "navbar_start": ["navbar-logo"],

    # Defaults to ["navbar-nav"]
    "navbar_center": ["navbar-nav"],

    # Defaults to ['theme-switcher', 'navbar-icon-links']
    "navbar_end": ["theme-switcher", "navbar-icon-links"],

    # Defaults to ['search-button-field']
    "navbar_persistent": ["search-button"],

    # -- Article/content -----------------------

    # Defaults to ['breadcrumbs']
    "article_header_start": [],

    # Defaults to []
    "article_header_end": [],

    # Defaults to []
    "article_footer_items": [],

    # Defaults to []
    "content_footer_items": [],

    # -- Primary sidebar (left side) -----------

    # Defaults to ["sidebar-ethical-ads"]
    "primary_sidebar_end": [],

    # -- Footer --------------------------------

    # Defaults to ['copyright', 'sphinx-version']
    "footer_start": ["copyright"],

    # Defaults to []
    "footer_center": [],

    # Defaults to ['theme-version']
    "footer_end": [],

    # Secondary sidebar (right side)

    # Defaults to ['page-toc', 'edit-this-page', 'sourcelink']
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],

    # -- Announcement banner -------------------

    # Options: True | False (default)
    "show_version_warning_banner": True,

    # Defaults to ""
    # "announcement": "&#128680; This is documentation for the <b>unstable development version</b> of YASA. <a href='https://yasa-sleep.org'>Switch to stable version</a> &#128680;",
    # "announcement": "<span style='font-family: Consolas, monospace;'>pip install yasa --upgrade</span> &#127881;",
}

# html_title = f"{project} v{version}"  # defaults to "<project> v<revision> documentation"
# html_short_title = f"{project} v{release}"  # used in the navbar if not using pydata logo

# The base URL which points to the root of the HTML documentation.
# It is used to indicate the location of document using the Canonical Link Relation.
# Defaults to ""
html_baseurl = "https://yasa-sleep.org"

# A dictionary of values to pass into the template engine's context for all pages.
# Defaults to {}
html_context = {
    # "display_github": True,
    # "github_url": "https://github.com",
    "github_user": "raphaelvallat",
    "github_repo": "yasa",
    "github_version": "master",
    "doc_path": "doc",

    # PyData theme uses this to set the initial dark/light theme.
    # Options: "auto" (default) | "light" | "dark"
    "default_mode": "auto",
}

# If given, this must be a filename or URL that points to the logo of the documentation.
# It is placed at the top of the sidebar; its width should therefore not exceed 200 pixels.
# Defaults to ""
# See ``html_theme_options['logo']`` to set this separately for light and dark themes in PyData theme.
html_logo = "https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/yasa_128x128.png"

# If given, this must be a filename or URL that points to the favicon of the documentation.
# Browsers use this as the icon for tabs, windows and bookmarks.
# It should be a 16-by-16 pixel icon in the PNG, SVG, GIF, or ICO file formats.
# Defaults to ""
html_favicon = "https://raw.githubusercontent.com/raphaelvallat/yasa/refs/tags/v0.6.5/docs/pictures/favicon.ico"

# A list of CSS files.
# The entry must be a filename string or a tuple containing the filename string and the attributes dictionary.
# Defaults to []
html_css_files = ["css/custom.css"]

# A list of paths that contain custom static files (such as style sheets or script files).
# Defaults to []
html_static_path = ["_static"]

# Add link anchors for each heading and description environment.
# Options: True (default) | False
html_permalinks = True

# Text for link anchors for each heading and description environment.
# HTML entities and Unicode are allowed.
# Defaults to "¶"
html_permalinks_icon = "#"

# A dictionary defining custom sidebar templates, mapping document names to template names.
# The keys can contain glob-style patterns,
# in which case all matching documents will get the specified sidebars.
# The bundled first-party sidebar templates that can be rendered are:
#   * localtoc.html – a fine-grained table of contents of the current document
#   * globaltoc.html – a coarse-grained table of contents for the whole documentation set, collapsed
#   * relations.html – two links to the previous and next documents
#   * sourcelink.html – a link to the source of the current document,
#                       if enabled in ``html_show_sourcelink``
#   * searchbox.html – the "quick search" box
#   * See other theme-specific templates
# Defaults are defined by the PyData theme: {"**": ["sidebar-nav-bs", "sidebar-ethical-ads"]}
# In PyData theme, this corresponds only to the primary (left) sidebar.
# See also ``html_theme_options['primary_sidebar_end']`` for PyData theme.
# PyData templates:
#   * sidebar-nav-bs.html – a bootstrap-friendly navigation section
#   * sidebar-ethical-ads.html – a placement for ReadTheDocs's Ethical Ads (will only show up on ReadTheDocs)
#   * sidebar-quicklinks.html
html_sidebars = {
    "**": [],  # remove sidebar from all pages
}

# If True, generate domain-specific indices in addition to the general index.
# For e.g. the Python domain, this is the global module index.
# Options: True (default) | False | Sequence of index names
html_domain_indices = True

# Controls if an index is added to the HTML documents.
# Options: True (default) | False
html_use_index = False

# If True, the reStructuredText sources are included in the HTML build as `_sources/docname`.
# Options: True (default) | False
html_copy_source = True

# If True (and ``html_copy_source`` is true as well),
# links to the reStructuredText sources will be added to the sidebar.
# Options: True (default) | False
html_show_sourcelink = True

# The suffix to append to source links (see ``html_show_sourcelink``),
# unless files they have this suffix already.
# Defaults to ".txt"
html_sourcelink_suffix = ".txt"

# The file name suffix (file extension) for generated HTML files.
# Defaults to ".html"
html_file_suffix = ".html"

# If True, "© Copyright …" is shown in the HTML footer, with the value or values from ``copyright``.
# Options: True (default) | False
html_show_copyright = True

# Show a summary of the search result, i.e., the text around the keyword.
# Options: True (default) | False
html_show_search_summary = True

# Add "Created using Sphinx" to the HTML footer.
# Options: True (default) | False
html_show_sphinx = False

# Encoding of HTML output files.
# Defaults to "utf-8"
html_output_encoding = "utf-8"

# The maths renderer to use for HTML output. The bundled renders are mathjax and imgmath.
# You must also load the relevant extension in extensions.
# Defaults to "mathjax"
html_math_renderer = "mathjax"


# -- Built-in extensions -----------------------------------------------------

# -- Built-in extensions -----------------------------------------------------
# -- -> Options for sphinx.ext.autodoc ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# This value selects what content will be inserted into the main body of an autoclass directive.
# The possible values are:
#   * 'class' - Only the class' docstring is inserted.
#               You can still document __init__ as a separate method using automethod
#               or the members option to autoclass.
#   * 'both' - Both the class' and the __init__ method's docstring are concatenated and inserted.
#   * 'init' - Only the __init__ method's docstring is inserted.
# Options: class (default) | init | both
autoclass_content = "class"  # class (default), init, both

# This value selects how the signature will be displayed for the class defined by autoclass directive.
# The possible values are:
#   * 'mixed' - Display the signature with the class name.
#   * 'separated' - Display the signature as a method.
# Defaults to "mixed"
autodoc_class_signature = "mixed"

# Define the order in which automodule and autoclass members are listed.
# Options: alphabetical (default) | groupwise | bysource
autodoc_member_order = "bysource"

# The default options for autodoc directives.
# They are applied to all autodoc directives automatically.
# It must be a dictionary which maps option names to the values.
# For example:
# autodoc_default_options = {
#     'members': 'var1, var2',
#     'member-order': 'bysource',
#     'special-members': '__init__',
#     'undoc-members': True,
#     'exclude-members': '__weakref__'
# }
# Defaults to {}
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

# If True, autodoc will look at the first line of the docstring for functions and methods,
# and if it looks like a signature, use the line as the signature and remove it from
# the docstring content.
# Options: True (default) | False
autodoc_docstring_signature = True

# This value controls how to represent typehints.
# Options: signature (default) | description | none | both
autodoc_typehints = "none"

# If True, the default argument values of functions will be not evaluated on generating document.
# It preserves them as is in the source code.
# Options: True | False (default)
autodoc_preserve_defaults = False

# This value controls the docstrings inheritance.
# If set to True the docstring for classes or methods,
# if not explicitly set, is inherited from parents.
# Options: True (default) | False
autodoc_inherit_docstrings = True

# -- Built-in extensions -----------------------------------------------------
# -- -> Options for sphinx.ext.autosectionlabel ------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html#configuration

# True to prefix each section label with the name of the document it is in,
# followed by a colon. For example, index:Introduction for a section called
# Introduction that appears in document index.rst. Useful for avoiding
# ambiguity when the same section heading appears in different documents.
# Options: True | False (default)
autosectionlabel_prefix_document = False

# If set, autosectionlabel chooses the sections for labeling by its depth.
# For example, when set 1, labels are generated only for top level sections.
# Defaults to None (i.e. all sections are labeled)
autosectionlabel_maxdepth = None

# -- Built-in extensions -----------------------------------------------------
# -- -> Options for sphinx.ext.autosummary -----------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html#generating-stub-pages-automatically

# Boolean indicating whether to scan all found documents
# for autosummary directives, and to generate stub pages for each.
# Can also be a list of documents for which stub pages should be generated.
# Defaults to True
autosummary_generate = True

# If true, autosummary overwrites existing files by generated stub pages.
# Options: True (default) | False
autosummary_generate_overwrite = True

# A boolean flag indicating whether to document classes and functions imported in modules.
# Options: True | False (default)
autosummary_imported_members = False

# If False and a module has the `__all__` attribute set,
# autosummary documents every member listed in `__all__` and no others.
autosummary_ignore_module_all = True

# -- Built-in extensions -----------------------------------------------------
# -- -> Options for sphinx.ext.intersphinx -----------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

# This config value contains the locations and names of other projects
# that should be linked to in this documentation.
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

# The maximum number of days to cache remote inventories.
# Set this to a negative value to cache inventories for unlimited time.
# Defaults to 5
intersphinx_cache_limit = 5

# When a non-external cross-reference is being resolved by intersphinx,
# skip resolution if it matches one of the specifications in this list.
# Defaults to ["std:doc"]
intersphinx_disabled_reftypes = ["std:doc"]


# -- External extensions -----------------------------------------------------
# -- -> Options for sphinx-copybutton ----------------------------------------
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html#use-and-customize

# The prompt text that you'd like removed from copied text in your code blocks.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True  # JavaScript regular expression

# If sphinx-copybutton detects lines that begin with code prompts,
# it will only copy the text in those lines (after stripping the prompts).
# This assumes that the rest of the code block contains outputs that shouldn't be copied.
# Options: True (default) | False
copybutton_only_copy_prompt_lines = True

# Remove the prompt text from lines according to the value of ``copybutton_prompt_text``.
# Options: True (default) | False
copybutton_remove_prompts = True

# Copy/pass through empty lines, determined by `line.trim() === ''`.
# Options: True (default) | False
copybutton_copy_empty_lines = True

# Optionally alter the image used for the copy icon that pops up.
# copybutton_image_svg = ""

# -- External extensions -----------------------------------------------------
# -- -> Options for matplotlib.sphinxext.plot_directive ----------------------
# https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html#configuration-options

# Default value for the include-source option.
# Options: True | False (default)
plot_include_source = True

# Whether to show a link to the source in HTML.
# Options: True (default) | False
plot_html_show_source_link = False

# Code that should be executed before each plot.
# Defaults to None, which is equivalent to a string containing
# `import numpy as np; from matplotlib import pyplot as plt`
plot_pre_code = None

# File formats to generate.
# List of tuples or strings that determine the file format and the DPI.
# [(suffix, dpi), suffix, ...]
# For entries whose DPI was omitted, sensible defaults are chosen.
# Defaults to ['png', 'hires.png', 'pdf']
plot_formats = [("png", 90)]

# Whether to show links to the files in HTML.
# Options: True (default) | False
plot_html_show_formats = False

# A dictionary containing any non-standard rcParams that should be applied before each plot.
# See also ``plot_apply_rcparams``.
# Defaults to {}
plot_rcparams = {}

# -- External extensions -----------------------------------------------------
# -- -> Options for numpydoc -------------------------------------------------
# https://numpydoc.readthedocs.io/en/latest/install.html#configuration

# Whether to produce plot:: directives for Examples sections
# that contain `import matplotlib` or `from matplotlib import`.
# Options: True | False (default)
numpydoc_use_plots = True

# Whether to show all members of a class in the Methods and Attributes sections automatically.
# Options: True (default) | False
numpydoc_show_class_members = False

# Whether to show all inherited members of a class in the Methods and Attributes sections automatically.
# If False, inherited members won't be shown.
# It can also be a dict mapping names of classes to boolean values (missing keys are treated as True).
# Options: True (default) | False
numpydoc_show_inherited_class_members = True

# Whether to create a Sphinx table of contents for the lists of class methods and attributes.
# If a table of contents is made, Sphinx expects each entry to have a separate page.
# Options: True (default) | False
numpydoc_class_members_toctree = True

# Whether to format the Attributes section of a class page in the same way as the Parameter section.
# If it’s False, the Attributes section will be formatted as the Methods section using an autosummary table.
# Options: True (default) | False
numpydoc_attributes_as_param_list = True

# -- External extensions -----------------------------------------------------
# -- -> Options for notfound.extension -------------------------------------------------
# https://sphinx-notfound-page.readthedocs.io/en/latest/configuration.html

# Context passed to the template defined by `notfound_template` or auto-generated.
notfound_context = {
    "title": "Page not found",
    "body": """
        <h1>This page may have moved.</h1>
        <p>The YASA documentation site has recently been upgraded! Some URLs have changed.</p>
        <p>Navigate the menu or search to find the page you were looking for.</p>
        <p>Please update any bookmarks or autofills that point to the YASA documentation site.</p>
    """,
}

# Prefix added to all the URLs generated in the 404 page.
# Defaults to READTHEDOCS env variable, typically "/en/latest/"
# Note special case when using default GitHub pages URL: "/<repo>/"
# https://sphinx-notfound-page.readthedocs.io/en/latest/faq.html#does-this-extension-work-with-github-pages
notfound_urls_prefix = None

# -- External extensions -----------------------------------------------------
# -- -> Options for sphinx_reredirects -------------------------------------------------
# https://documatt.com/sphinx-reredirects/usage.html
# Defaults to {}
redirects = {}

# TEMPORARY: This can be removed after people stop using old links.
# We need to generate a list of redirects for the old documentation
# that was hosted under relative paths behind buid/html.
# Create a sphinx extension that will find all docs and generate a
# redirects dictionary that maps build/html/docpath to just docpath.
def setup(app):
    app.connect("env-updated", generate_redirects)

def generate_redirects(app, env):
    redirects = app.config.redirects or {}
    for docpath in env.found_docs:
        old_path = f"build/html/{docpath}"
        new_path = "../" * old_path.count("/") + f"{docpath}.html"
        redirects[old_path] = new_path
    app.config.redirects = redirects
