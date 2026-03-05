#!/usr/bin/env sh
# Development scripts for pyRVT using uv
# Compatible with bash, zsh, and other POSIX shells

set -e

case "$1" in
  "install")
    echo "Installing project in development mode..."
    uv sync --extra test --extra docs --extra style
    ;;
  "test")
    echo "Running tests..."
    uv run --extra test pytest
    ;;
  "test-cov")
    echo "Running tests with coverage..."
    uv run --extra test pytest --cov=src/pyrvt --cov-report=html --cov-report=term
    ;;
  "format")
    echo "Formatting code..."
    uv run --extra style black .
    uv run --extra style ruff check --fix .
    ;;
  "lint")
    echo "Checking code style..."
    uv run --extra style ruff check .
    ;;
  "docs-build")
    echo "Building documentation..."
    uv run --extra docs sphinx-build -b html docs docs/_build/html
    ;;
  "docs-serve")
    echo "Serving documentation..."
    uv run --extra docs sphinx-autobuild docs docs/_build/html --host localhost --port 8000
    ;;
  "docs-clean")
    echo "Cleaning documentation build..."
    rm -rf docs/_build
    ;;
  "clean")
    echo "Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/ docs/_build/
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -delete
    ;;
  *)
    echo "Usage: $0 {install|test|test-cov|format|lint|docs-build|docs-serve|docs-clean|clean|completion}"
    echo ""
    echo "Available commands:"
    echo "  install     - Install project in development mode"
    echo "  test        - Run tests"
    echo "  test-cov    - Run tests with coverage"
    echo "  format      - Format code with black and ruff"
    echo "  lint        - Check code style with ruff"
    echo "  docs-build  - Build documentation"
    echo "  docs-serve  - Serve documentation with auto-reload"
    echo "  docs-clean  - Clean documentation build"
    echo "  clean       - Clean all build artifacts"
    echo "  completion  - Set up tab completion for your shell"
    exit 1
    ;;
esac
