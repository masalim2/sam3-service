#!/bin/bash

# Ensure uv
command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Ensure .venv:
[ -d .venv ] || uv venv --python 3.12.13 .venv
source .venv/bin/activate

# Install dependencies:
uv sync --all-groups