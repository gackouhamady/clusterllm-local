install:
	pip install poetry && poetry install

run_baseline:
	poetry run python src/main.py config=model_gpt

run_local:
	poetry run python src/main.py config=model_local

test:
	poetry run pytest tests/