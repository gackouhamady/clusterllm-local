# Contributing to ClusterLLM-Local

Thank you for considering contributing to **ClusterLLM-Local** 🙏  
This project focuses on **reproducible research for Large Language Models**, combining
scientific rigor, clean engineering, and transparent experimentation.

Contributions are welcome in many forms:
- Code (core methods, experiments, utilities)
- Documentation (tutorials, explanations, diagrams)
- Bug reports & reproducibility issues
- Tests and CI improvements

---

## Project Philosophy

This repository follows a **reproducibility-first** mindset:

- **System reproducibility**: Docker + CUDA
- **Dependency reproducibility**: Poetry + lock file
- **Data reproducibility**: scripted loaders (and DVC when applicable)
- **Scientific reproducibility**: deterministic code, fixed seeds, documented protocols

All contributions should respect these principles.

---

## How to Contribute

### 1. Fork & Clone
```bash
git fork <repo-url>
git clone <your-fork-url>
cd clusterllm-local
````

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Examples:

* `feature/contrastive-loss`
* `fix/docs-toctree`
* `experiment/batch-hard-negatives`

---

### 3. Development Setup

We strongly recommend using the **Docker environment** to avoid system differences.

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run clusterllm
```

Or locally (advanced users only):

```bash
poetry install --with dev
```

---

### 4. Make Your Changes

* Follow the existing project structure (`src/`, `tests/`, `docs/`)
* Keep changes **focused and minimal**
* Prefer **explicit code over clever code**

---

### 5. Add / Update Tests (Required)

All new code should be covered by tests.

```bash
poetry run pytest
```

If external resources are involved (datasets, GPUs, network):

* **Mock them**
* Tests must be fast and deterministic

---

### 6. Run Local CI Checks (Mandatory)

Before committing, run:

```bash
./scripts/ci_local.sh
```

This will run:

* `flake8` (linting)
* `pytest` (tests)
* `sphinx-build` (documentation)

Your contribution **must pass all checks**.

---

### 7. Commit Guidelines

Use clear, descriptive commit messages:

```bash
git commit -m "feat: add contrastive loss with temperature scaling"
git commit -m "fix: correct Bank77 loader path resolution"
git commit -m "docs: add tutorial on embeddings and LLMs"
```

---

### 8. Push & Open a Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request **against `main`**.

---

## Code Style

* Follow **PEP8**
* 4-space indentation
* Clear, explicit variable names
* Linting is enforced via **flake8**
* Avoid unused imports and dead code

Formatting tools:

```bash
flake8 src tests
```

(Black is optional but encouraged for large refactors.)

---

## Testing Guidelines

* Use `pytest`
* Tests should be:

  * isolated (no shared state)
  * deterministic
  * fast (< a few seconds)
* Prefer **unit tests** over heavy integration tests

Example:

```python
def test_something_expected():
    assert result == expected
```

---

## Documentation Contributions

Documentation is **first-class** in this project.

* Tutorials live in `docs/source/tutorials/`
* API docs are auto-generated
* Use **MyST Markdown** (`.md`) or reStructuredText (`.rst`)

To build docs locally:

```bash
poetry run sphinx-build -b html docs/source docs/build/html
```

---

## Reporting Issues

If you encounter a bug or reproducibility problem, please open an issue and include:

* Project version or Git commit hash
* Execution environment (local, Docker CPU/GPU, CI)
* OS, Python version
* Exact commands used
* Full error logs or tracebacks

Use the provided **Bug Report template**.

---

## Pull Request Review Criteria

A PR will be accepted if:

* CI passes (lint, tests, docs)
* Code aligns with reproducibility principles
* Changes are well-scoped and documented
* Scientific assumptions are explicit

---

## Code of Conduct

All contributors are expected to be respectful, constructive, and professional.
This is a research-oriented project — discussions should remain technical and factual.

---

Thank you for helping improve **ClusterLLM-Local** 
Your contribution directly improves the **scientific credibility** of the project.

```

