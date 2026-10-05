# Makefile for cuTile Rust Book

SPHINXBUILD   = sphinx-build
SPHINXAUTOBUILD = sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean livehtml setup

help:
	@echo "cuTile Rust Book Build Commands"
	@echo ""
	@echo "  make setup     - Create venv and install dependencies (run once)"
	@echo "  make html      - Build HTML documentation"
	@echo "  make livehtml  - Start live-reloading server at http://localhost:8000"
	@echo "  make clean     - Remove build directory"
	@echo ""
	@echo "First time? Run: make setup && source .venv/bin/activate && make livehtml"

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "✅ Setup complete! Now run:"
	@echo "   source .venv/bin/activate"
	@echo "   make livehtml"

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)

# Live reload server - auto-refreshes browser on file changes
livehtml:
	$(SPHINXAUTOBUILD) $(SOURCEDIR) $(BUILDDIR)/html \
		--port 8000 \
		--open-browser \
		--write-all \
		--watch $(SOURCEDIR) \
		--ignore "*.pyc" \
		--ignore ".venv/*" \
		--ignore "_build/*" \
		--ignore "book/*"
