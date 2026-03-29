PYTHON ?= python3
.PHONY: install test build check clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test: install
	$(PYTHON) -m pytest tests/ -v --tb=short

build: install
	$(PYTHON) -m pip install build twine
	$(PYTHON) -m build

check: build
	$(PYTHON) -m twine check dist/*

clean:
	rm -rf dist/ build/ .pytest_cache/
	find . -name __pycache__ -o -name '*.egg-info' | xargs rm -rf 2>/dev/null || true
