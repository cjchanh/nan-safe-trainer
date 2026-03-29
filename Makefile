.PHONY: install test build check clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

build:
	python -m build

check: build
	twine check dist/*

clean:
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/
	find . -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
