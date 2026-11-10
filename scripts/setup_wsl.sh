#!/usr/bin/env bash
# Sets up a local training environment in WSL (or any Linux with an NVIDIA GPU), as an alternative to Colab.
#   bash scripts/setup_wsl.sh
# The venv lives in ~/ft rather than on the Windows drive, which is much faster from WSL.
set -euo pipefail
# Triton compiles its CUDA driver on first use and needs the Python headers
[ -f /usr/include/python3.12/Python.h ] || sudo apt-get install -y python3.12-dev build-essential
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/ft && cd ~/ft
[ -d .venv ] || uv venv -p 3.12 .venv
. .venv/bin/activate
uv pip install unsloth matplotlib
python -c "import torch, unsloth; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'unsloth', unsloth.__version__)"
