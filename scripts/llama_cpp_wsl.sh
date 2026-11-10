#!/usr/bin/env bash
# Fetches llama.cpp's converter (Python only, nothing to compile) into ~/ft/llama.cpp
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
. ~/ft/.venv/bin/activate
[ -d ~/ft/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/ft/llama.cpp
uv pip install -q ~/ft/llama.cpp/gguf-py sentencepiece
grep -c "Qwen3_5" ~/ft/llama.cpp/convert_hf_to_gguf.py
