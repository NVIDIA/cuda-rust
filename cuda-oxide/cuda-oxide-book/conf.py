# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'cuda-oxide'
copyright = '2025-2026, NVIDIA Corporation'
author = 'Nihal Pasham, NVIDIA Corporation'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',           # Markdown support
    'sphinx_copybutton',     # Copy button on code blocks
    'sphinx_design',         # Cards, grids, tabs
    'sphinx_sitemap',        # Generate sitemap.xml for SEO
]

# Markdown configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "attrs_inline",
]

# Source files can be .rst or .md
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv', 'README.md']

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "logo": {
        "image_light": "_static/nvidia-logo-horiz-rgb-blk-for-screen.svg",
        "image_dark": "_static/nvidia-logo-horiz-rgb-wht-for-screen.svg",
        "text": "cuda-oxide",
    },

    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVlabs/cuda-oxide",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],

    # Navbar
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_end": ["navbar-icon-links", "search-button", "theme-switcher"],
    "navbar_persistent": [],

    # LEFT SIDEBAR - Show global TOC
    "primary_sidebar_end": [],
    "show_nav_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": False,

    # RIGHT SIDEBAR - On this page
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 2,

    # Footer - NVIDIA branding
    "footer_start": [],
    "footer_center": ["footer"],
    "footer_end": [],

    # Misc
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
    "show_prev_next": True,
}

html_sidebars = {
    "**": [
        "globaltoc.html",
    ],
}

html_static_path = ['_static']
html_css_files = [
    'css/nvidia-sphinx-theme.css',
    'css/custom.css',
    'css/lightbox.css',
]
html_js_files = [
    'js/lightbox.js',
    'js/spa-nav.js',
]

pygments_style = 'default'
html_show_sourcelink = False
html_title = "cuda-oxide"
html_baseurl = "https://nvlabs.github.io/cuda-oxide/"
