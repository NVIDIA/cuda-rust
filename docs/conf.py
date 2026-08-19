# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Umbrella docs for the cuda-rust repository. This builds the landing page only;
# the component books are built separately and stitched in by build_all_docs.sh.

import os

project = 'CUDA Rust'
copyright = '2026, NVIDIA Corporation'
author = 'NVIDIA Corporation'
release = os.environ.get('CUDA_RUST_DOCS_VERSION', '0.1.0')
version = release

extensions = [
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
]

myst_enable_extensions = ["colon_fence", "deflist", "tasklist", "attrs_inline"]
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
master_doc = 'index'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv', 'README.md', 'NOTES.md']

html_theme = 'pydata_sphinx_theme'
html_title = "CUDA Rust"
html_baseurl = os.environ.get('CUDA_RUST_DOCS_BASEURL', 'https://nvlabs.github.io/cuda-rust/')

html_theme_options = {
    "logo": {"text": "CUDA Rust"},
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVlabs/cuda-rust",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_end": ["navbar-icon-links", "search-button", "theme-switcher"],
    "show_nav_level": 2,
    "navigation_depth": 2,
    "collapse_navigation": False,
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 2,
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
    "show_prev_next": False,
}

html_show_sourcelink = False
