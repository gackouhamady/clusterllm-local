.PHONY: install lint test docs check

install:
	poetry install --with dev

lint:
	poetry run flake8 src tests

test:
	poetry run pytest -q

docs:
	poetry run sphinx-apidoc -o docs/source/api src/clusterllm -f
	poetry run sphinx-build -b html docs/source docs/build/html

check: install lint test docs
