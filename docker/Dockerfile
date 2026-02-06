# =============================================================================
# ClusterLLM-Local Core - Day 2 Dockerfile (CUDA 12.x, Python 3.13, Poetry)
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# --- System deps for building Python + common tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    uuid-dev \
 && rm -rf /var/lib/apt/lists/*

# --- Build Python 3.13 from source
WORKDIR /tmp
RUN wget -q https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz \
 && tar -xzf Python-3.13.0.tgz \
 && cd Python-3.13.0 \
 && ./configure --enable-optimizations \
 && make -j"$(nproc)" \
 && make altinstall

# --- Make python point to 3.13
RUN ln -sf /usr/local/bin/python3.13 /usr/local/bin/python \
 && python --version

# --- Install pip for Python 3.13 (THIS FIXES YOUR ERROR)
RUN /usr/local/bin/python3.13 -m ensurepip --upgrade \
 && /usr/local/bin/python3.13 -m pip install --upgrade pip setuptools wheel

# --- Install Poetry
RUN /usr/local/bin/python3.13 -m pip install --no-cache-dir poetry

# --- App
WORKDIR /workspace

# Copy manifests first for caching
COPY pyproject.toml poetry.lock ./

# Install deps
RUN poetry install --no-root --no-ansi

# Copy code
COPY src ./src
COPY configs ./configs
COPY tests ./tests
COPY dvc.yaml .
COPY README.md .
COPY fix_auth.py .

# Caches (mounted by compose)
ENV HF_HOME=/root/.cache/huggingface \
    DVC_CACHE_DIR=/root/.cache/dvc

CMD ["/bin/bash"]
