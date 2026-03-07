FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && python --version
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install "poetry==1.8.3"

WORKDIR /workspace
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-ansi

COPY src ./src
COPY configs ./configs
COPY tests ./tests
COPY dvc.yaml ./
COPY README.md ./

ENV HF_HOME=/root/.cache/huggingface \
    DVC_CACHE_DIR=/root/.cache/dvc

CMD ["/bin/bash"]
