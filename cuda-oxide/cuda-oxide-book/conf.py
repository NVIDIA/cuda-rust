# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'cuda-oxide'
copyright = '2025-2026, NVIDIA Corporation'
author = 'Nihal Pasham, NVIDIA Corporation'
release = '0.3.0'

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
            "url": "https://github.com/NVIDIA/cuda-rust",
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
html_baseurl = "https://nvidia.github.io/cuda-rust/cuda-oxide/"


def setup(app):
    """Register custom Pygments lexers for CLI commands, Tile C++, and Tile IR."""
    from pygments.lexer import RegexLexer, inherit
    from pygments.lexers.c_cpp import CppLexer
    from pygments.lexers.shell import BashLexer
    from pygments.token import (
        Comment,
        Keyword,
        Literal,
        Name,
        Number,
        Operator,
        Punctuation,
        String,
        Text,
        Whitespace,
    )

    class OxideBashLexer(BashLexer):
        """Bash lexer that highlights CLI tools, cargo-oxide subcommands, flags, and GPU targets."""

        name = "OxideBash"
        aliases = ["bash", "sh"]

        tokens = {
            "basic": [
                (
                    r"\b(cargo|nvcc|tileiras|just|rustup|rustc|git|python3?)\b",
                    Name.Builtin,
                ),
                (
                    r"\b(oxide|tilelib|build|run|test|check|new|doctor|clean|install)\b",
                    Keyword,
                ),
                (r"(--[A-Za-z0-9_-]+|-[A-Za-z0-9]+)\b", Name.Attribute),
                (r"\b(sm_[0-9]+[a-z]?|compute_[0-9]+[a-z]?)\b", Literal.String.Symbol),
                inherit,
            ],
        }

    class OxideCppLexer(CppLexer):
        """C++ lexer that accepts CUDA Tile C++ attributes and foreign-call qualifiers."""

        name = "OxideCpp"
        aliases = ["cpp", "c++", "cuda"]

        tokens = {
            "statements": [
                (r"\[\[[^\]]*\]\]", Name.Attribute),
                (
                    r"\b(__tile_global__|__tile__|__experimental_foreign_decl__|__experimental_foreign_name__)\b",
                    Keyword.Reserved,
                ),
                inherit,
            ],
        }

    class MlirLexer(RegexLexer):
        """Lexer for MLIR and CUDA Tile IR modules."""

        name = "MLIR"
        aliases = ["mlir", "tileir"]

        tokens = {
            "root": [
                (r"\s+", Whitespace),
                (r"//[^\n]*", Comment.Single),
                (r'"[^"]*"', String.Double),
                (
                    r"\b(module|func|extern|implements|synchronization|mem_layout|fragmap|local|in|out|inout|warp|none)\b",
                    Keyword,
                ),
                (r"\b(cuda_tile\.[A-Za-z0-9_$]+)\b", Keyword.Namespace),
                (r"@[A-Za-z_][A-Za-z0-9_$.]*", Name.Function),
                (r"%[A-Za-z0-9_$.]+", Name.Variable),
                (r"![A-Za-z0-9_$.]+", Keyword.Type),
                (r"#[A-Za-z0-9_$.]+", Name.Attribute),
                (r"\b([0-9]+x)?(f16|bf16|f32|f64|i1|i8|i16|i32|i64|index)\b", Keyword.Type),
                (r"\b[0-9]+\b", Number.Integer),
                (r"->|=", Operator),
                (r"[{}()\[\]<>:,]", Punctuation),
                (r"[^\s{}()\[\]<>:,]+", Text),
            ],
        }

    app.add_lexer("bash", OxideBashLexer)
    app.add_lexer("sh", OxideBashLexer)
    app.add_lexer("cpp", OxideCppLexer)
    app.add_lexer("c++", OxideCppLexer)
    app.add_lexer("cuda", OxideCppLexer)
    app.add_lexer("mlir", MlirLexer)
    app.add_lexer("tileir", MlirLexer)

