#!/usr/bin/env bash
set -euo pipefail

echo "==> (1/4) Install deps (locked)"
poetry install --with dev --no-root

echo "==> (2/4) Lint: flake8"
poetry run flake8 src tests

echo "==> (3/4) Tests: pytest"
poetry run pytest -q

echo "==> (4/4) Docs build"
poetry run sphinx-build -b html docs/source docs/build/html

echo "✅ CI local checks passed"
