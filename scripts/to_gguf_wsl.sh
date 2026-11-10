#!/usr/bin/env bash
# Converts the merged model from train_local.py to GGUF for Ollama: the language model in q8_0 and the vision
# projector (mmproj) in f16, both written next to the Modelfile in ollama/.
#   bash scripts/llama_cpp_wsl.sh   # once, fetches the converter
#   bash scripts/to_gguf_wsl.sh
set -euo pipefail
cd "$(dirname "$0")/.."
. ~/ft/.venv/bin/activate
mkdir -p ollama
python ~/ft/llama.cpp/convert_hf_to_gguf.py ~/ft/out/merged --outtype q8_0 --outfile ollama/doc-extraction-2b-q8_0.gguf
python ~/ft/llama.cpp/convert_hf_to_gguf.py ~/ft/out/merged --mmproj --outtype f16 --outfile ollama/doc-extraction-2b-mmproj-f16.gguf
ls -lh ollama/*.gguf
