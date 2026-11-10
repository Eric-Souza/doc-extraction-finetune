"""Uploads the local run (scripts/train_local.py and scripts/to_gguf_wsl.sh) to the Hugging Face Hub, under the names
the notebook pushes to: the GGUF with its vision projector, which ollama/Modelfile pulls, and the LoRA adapter.
--merged also uploads the merged 16-bit model (4.3 GB) that notebooks/serve_vllm.ipynb serves.

    python scripts/upload_model.py [your-hf-username] [--out ~/ft/out] [--merged]

HF_USER and HF_TOKEN (with write access) come from the environment or from .env.
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
NAME = "qwen3.5-2b-doc-extraction-br"

if (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().partition("=")
        if key and not key.startswith("#") and value:
            os.environ.setdefault(key, value.strip().strip('"'))

parser = argparse.ArgumentParser()
parser.add_argument("user", nargs="?", default=os.environ.get("HF_USER"))
parser.add_argument("--out", type=Path, default=Path.home() / "ft/out", help="where train_local.py saved the models")
parser.add_argument("--merged", action="store_true")
args = parser.parse_args()
if not args.user or not os.environ.get("HF_TOKEN"):
    parser.error("set HF_USER and HF_TOKEN in .env")

summary = json.loads((ROOT / "runs/train_summary.json").read_text(encoding="utf-8"))
CARD = f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
language: [pt]
tags: [document-extraction, brazil, vision, qlora, unsloth]
---
# Qwen3.5-2B fine-tuned for Brazilian financial documents

Reads a boleto, a DANFE (the printout of an NF-e) or an informe de rendimentos into the JSON that
[doc-extraction-copilot](https://github.com/Eric-Souza/doc-extraction-copilot) expects. A QLoRA on the language layers,
{summary["steps"]} steps in {summary["train_minutes"]:.0f} minutes on an {summary["gpu"]}; how it was trained is in
[doc-extraction-finetune](https://github.com/Eric-Souza/doc-extraction-finetune).

On the app's 36 eval documents it reads 99.7% of the fields right (the base model: 7.3%), and the app's check digits
flag both fields it gets wrong. It was trained on documents from the same generator and templates as the eval set, so
expect less on layouts it has not seen.

```bash
ollama pull hf.co/{args.user}/{NAME}-GGUF:Q8_0
```
"""

api = HfApi()


def upload(repo: str, **files):
    """Uploads a folder (folder=...) or single files (path_in_repo=local path), with the model card."""
    api.create_repo(repo, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=repo)
    folder = files.pop("folder", None)
    if folder:
        api.upload_folder(folder_path=folder, repo_id=repo, ignore_patterns=["README.md"], commit_message="Local run")
    for name, path in files.items():
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo, commit_message=f"Add {name}")
    print(f"https://huggingface.co/{repo}")


# Ollama picks the file by the tag in its name, and the vision projector by "mmproj"
upload(
    f"{args.user}/{NAME}-GGUF",
    **{
        f"{NAME}-Q8_0.gguf": ROOT / "ollama/doc-extraction-2b-q8_0.gguf",
        f"mmproj-{NAME}-f16.gguf": ROOT / "ollama/doc-extraction-2b-mmproj-f16.gguf",
    },
)
upload(f"{args.user}/{NAME}-lora", folder=args.out / "lora")
if args.merged:
    upload(f"{args.user}/{NAME}", folder=args.out / "merged")
