"""Uploads the training set (data/, rendered by doc-extraction-copilot's scripts/make-train-set.ts) to the Hugging
Face Hub as a dataset, where the notebook downloads it.

    HF_TOKEN=... python scripts/upload_dataset.py your-hf-username
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
license: mit
task_categories: [image-text-to-text]
language: [pt]
tags: [synthetic, document-extraction, brazil]
---
# Synthetic Brazilian financial documents

Boletos, DANFEs of NF-e and informes de rendimentos, rendered by
[doc-extraction-copilot](https://github.com/Eric-Souza/doc-extraction-copilot) from fictional data with valid check
digits. Each line of `train.jsonl` and `val.jsonl` is one chat: the app's system prompt and user prompt, the page
images, and the known answer as JSON. Every name, CPF and CNPJ is made up.
"""

user = sys.argv[1]
repo = f"{user}/doc-extraction-br-synthetic"
data = Path(__file__).parent.parent / "data"
(data / "README.md").write_text(CARD, encoding="utf-8")
api = HfApi()
api.create_repo(repo, repo_type="dataset", exist_ok=True)
api.upload_folder(folder_path=data, repo_id=repo, repo_type="dataset", commit_message="Training set")
print(f"https://huggingface.co/datasets/{repo}")
