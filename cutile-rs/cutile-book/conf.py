# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'cuTile Rust'
copyright = '2025-2026, NVIDIA Corporation'
author = 'NVIDIA Corporation'
release = os.environ.get('CUTILE_DOCS_VERSION', '0.1.0')
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',           # Markdown support
    'sphinx_copybutton',     # Copy button on code blocks
    'sphinx_design',         # Cards, grids, tabs
]

# Markdown configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "attrs_inline",
]

# Auto-generate anchors for H1–H3 so inline `#slug` links resolve.
myst_heading_anchors = 3

# Source files can be .rst or .md
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master document
master_doc = 'index'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv', 'book', 'src']

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "logo": {
        "image_light": "_static/nvidia-logo-horiz-rgb-blk-for-screen.svg",
        "image_dark": "_static/nvidia-logo-horiz-rgb-wht-for-screen.svg",
        "text": "cuTile Rust",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVlabs/cutile-rs",
            "icon": "fa-brands fa-github",
        },
    ],
    
    # Navbar - minimal like TileIR
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_end": ["github_links", "navbar-icon-links", "search-button", "theme-switcher"],
    "navbar_persistent": [],
    
    # LEFT SIDEBAR - Show global TOC
    "primary_sidebar_end": [],
    "show_nav_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": False,
    
    # RIGHT SIDEBAR - On this page
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 2,
    
    # Footer - NVIDIA branding like TileIR
    "footer_start": [],
    "footer_center": ["footer"],
    "footer_end": [],
    
    # Misc
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
    "show_prev_next": True,
}

_docs_version = os.environ.get('CUTILE_DOCS_VERSION')
if _docs_version:
    html_theme_options["switcher"] = {
        "json_url": os.environ.get(
            "CUTILE_DOCS_SWITCHER_JSON",
            "/cutile-rs/_static/versions.json",
        ),
        "version_match": _docs_version,
    }
    html_theme_options["check_switcher"] = False
    html_theme_options["navbar_center"] = [
        *html_theme_options.get("navbar_center", []),
        "version-switcher",
    ]

# LEFT SIDEBAR configuration
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
]

# Syntax highlighting
pygments_style = 'default'

# Don't show "View page source"
html_show_sourcelink = False

# Title in sidebar
html_title = "cuTile Rust"
