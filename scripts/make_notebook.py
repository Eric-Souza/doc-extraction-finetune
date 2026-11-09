"""Writes notebooks/finetune.ipynb and notebooks/serve_vllm.ipynb from the cells below, so the notebooks stay
reviewable as code.

    python scripts/make_notebook.py
"""

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Fine-tune Qwen3.5-2B to read Brazilian financial documents

QLoRA with [Unsloth](https://github.com/unslothai/unsloth) on a free Colab T4 (or a Kaggle P100/T4), tracked in
[Weights & Biases](https://wandb.ai). The model learns to turn a boleto, a DANFE or an informe de rendimentos into
the JSON that [doc-extraction-copilot](https://github.com/Eric-Souza/doc-extraction-copilot) expects, from the
exact prompts that app sends. The result goes to the Hugging Face Hub as a LoRA adapter, a merged 16-bit model (for
vLLM) and a GGUF with its vision projector (for Ollama).

Before running, add two secrets (the key icon on the left in Colab, *Add-ons > Secrets* on Kaggle):

- `HF_TOKEN`: a Hugging Face token with write access
- `WANDB_API_KEY`: from https://wandb.ai/authorize

Then set `HF_USER` below and run everything. Training takes about 40 minutes on a T4.""",
    ),
    (
        "code",
        """%%capture
# Unsloth's own install cell for Qwen3.5 (from its Qwen3_5_(2B)_Vision notebook), plus wandb
import os, importlib.util
!pip install --upgrade -qqq uv
if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):
    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
    except: _numpy = "numpy"; _pil = "pillow"
    !uv pip install -qqq \\
        "torch==2.8.0" "triton>=3.3.0" {_numpy} {_pil} torchvision "torchaudio==2.8.0" bitsandbytes xformers==0.0.32.post2 \\
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\
        "unsloth[base] @ git+https://github.com/unslothai/unsloth"
    !uv pip install -qqq --no-deps "torchcodec==0.7.0"
elif importlib.util.find_spec("unsloth") is None:
    !uv pip install -qqq unsloth
!uv pip install --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo
!uv pip install transformers==5.2.0
!unsloth install-kernels
!uv pip install --no-deps --upgrade "torchao>=0.16.0"
!uv pip install -qqq wandb""",
    ),
    (
        "code",
        """import os

HF_USER = "your-hf-username"  # <- yours
BASE_MODEL = "unsloth/Qwen3.5-2B"
DATASET = f"{HF_USER}/doc-extraction-br-synthetic"  # uploaded by scripts/upload_dataset.py
OUT = f"{HF_USER}/qwen3.5-2b-doc-extraction-br"
RUN_NAME = "qlora-r16-lang-only"


def secret(name: str) -> str:
    \"\"\"A secret from Colab, then Kaggle, then the environment\"\"\"
    try:
        from google.colab import userdata

        return userdata.get(name)
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ[name]


os.environ["HF_TOKEN"] = secret("HF_TOKEN")
os.environ["WANDB_API_KEY"] = secret("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "doc-extraction-finetune\"""",
    ),
    (
        "markdown",
        """## The data

About 900 synthetic documents rendered by doc-extraction-copilot (`scripts/make-train-set.ts`), from other seeds
than its eval set, and none with an identifier the eval set uses. Each line is one chat: the app's system prompt,
its user prompt, the page images, and the known answer. A third of the documents also teach the classification
call the app makes first. PDFs were drawn page by page with the same code the app uses before sending one to a
local model, so training and inference see the same pixels.""",
    ),
    (
        "code",
        """import json
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image

root = Path(snapshot_download(DATASET, repo_type="dataset"))
rows = {s: [json.loads(l) for l in (root / f"{s}.jsonl").read_text().splitlines()] for s in ("train", "val")}


def to_chat(row: dict, with_answer: bool = True) -> list[dict]:
    images = [Image.open(root / p).convert("RGB") for p in row["images"]]
    # The order the app's OpenAI-compatible provider sends: the text, then the images
    messages = [
        {"role": "system", "content": [{"type": "text", "text": row["system"]}]},
        {"role": "user", "content": [{"type": "text", "text": row["prompt"]}, *({"type": "image", "image": i} for i in images)]},
    ]
    if with_answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]})
    return messages


class Chats:
    \"\"\"Opens the images only when a batch needs them: 1,200 page images don't fit in Colab's RAM\"\"\"

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {"messages": to_chat(self.rows[i])}


train, val = Chats(rows["train"]), Chats(rows["val"])
print(len(train), "training chats,", len(val), "held out")
print(rows["train"][0]["prompt"], "\\n", rows["train"][0]["answer"][:300])""",
    ),
    (
        "code",
        """from unsloth import FastVisionModel
import torch

model, processor = FastVisionModel.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,  # QLoRA: the base stays in 4 bits, only the adapter trains
    use_gradient_checkpointing="unsloth",
)
model = FastVisionModel.get_peft_model(
    model,
    # The base model already reads the digits (83% of fields right with the type given); what it gets wrong is
    # which value goes in which field and in what format. So only the language layers train
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)""",
    ),
    (
        "markdown",
        """## The prompt the model trains on must be the one it is served with

Ollama and vLLM are asked with thinking off, so the chat template opens the reply with an empty `<think></think>`
block. The training chats are rendered the same way, and this cell checks it.""",
    ),
    (
        "code",
        """sample = to_chat(rows["train"][0])
served = processor.apply_chat_template(sample[:2], add_generation_prompt=True, enable_thinking=False, tokenize=False)
trained = processor.apply_chat_template(sample, enable_thinking=False, tokenize=False)
assert trained.startswith(served), "The training chat does not start with the served prompt"
print(trained[len(served) - 120 : len(served) + 200])""",
    ),
    (
        "code",
        """from trl import SFTConfig, SFTTrainer
from unsloth.trainer import UnslothVisionDataCollator

FastVisionModel.for_training(model)
collator = UnslothVisionDataCollator(
    model,
    processor,
    resize="max",  # no resizing: Ollama and vLLM see the page at full size, so training must too
    train_on_responses_only=True,  # the loss is on the JSON only, not on the prompt and the image
    instruction_part="<|im_start|>user\\n",
    response_part="<|im_start|>assistant\\n",
    chat_template_kwargs={"enable_thinking": False},
)
trainer = SFTTrainer(
    model=model,
    tokenizer=processor,
    data_collator=collator,
    train_dataset=train,
    eval_dataset=val,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=0.001,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        seed=3407,
        output_dir="outputs",
        report_to="wandb",
        run_name=RUN_NAME,
        # Required for vision fine-tuning with Unsloth's collator
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=4096,
    ),
)
stats = trainer.train()
print(f"{stats.metrics['train_runtime'] / 60:.1f} min, peak {torch.cuda.max_memory_reserved() / 2**30:.1f} GB")""",
    ),
    (
        "markdown",
        """## A first look on the held-out chats

Not the eval: that runs in doc-extraction-copilot against the served model, with its check-digit review. This only
tells you, before exporting, whether the replies parse and how many fields match. The table goes to W&B.""",
    ),
    (
        "code",
        """import wandb


def leaves(value, prefix=""):
    if isinstance(value, list):
        return {k: v for i, x in enumerate(value) for k, v in leaves(x, f"{prefix}[{i}]").items()}
    if isinstance(value, dict):
        return {k: v for key, x in value.items() if key != "uncertain" for k, v in leaves(x, f"{prefix}.{key}" if prefix else key).items()}
    return {prefix: value}


FastVisionModel.for_inference(model)
table = wandb.Table(columns=["id", "task", "parsed", "fields right", "fields", "reply"])
right = total = parsed = 0
for row in rows["val"]:
    messages = to_chat(row, with_answer=False)
    images = [part["image"] for part in messages[1]["content"] if part["type"] == "image"]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    inputs = processor(images=images, text=[text], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=1500, do_sample=False, use_cache=True)
    reply = processor.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    truth = leaves(json.loads(row["answer"]))
    try:
        got, ok = leaves(json.loads(reply)), True
    except json.JSONDecodeError:
        got, ok = {}, False
    n = sum(got.get(k) == v for k, v in truth.items())
    right, total, parsed = right + n, total + len(truth), parsed + ok
    table.add_data(row["id"], row["task"], ok, n, len(truth), reply)
print(f"{parsed}/{len(rows['val'])} replies parsed, {right}/{total} fields match exactly")
wandb.log({"val/replies": table, "val/field_match": right / total, "val/parsed": parsed / len(rows["val"])})""",
    ),
    (
        "markdown",
        """## Export

Three copies on the Hub:
- **The LoRA adapter**, about 40 MB.
- **A merged 16-bit model**, which vLLM loads (`serve_vllm.ipynb`).
- **GGUFs with the vision projector**, which Ollama pulls straight from the Hub:

```
ollama pull hf.co/HF_USER/qwen3.5-2b-doc-extraction-br-GGUF:Q8_0
```""",
    ),
    (
        "code",
        """model.push_to_hub(f"{OUT}-lora", token=os.environ["HF_TOKEN"])
processor.push_to_hub(f"{OUT}-lora", token=os.environ["HF_TOKEN"])
model.push_to_hub_merged(OUT, processor, save_method="merged_16bit", token=os.environ["HF_TOKEN"])
model.push_to_hub_gguf(f"{OUT}-GGUF", processor, quantization_method=["q8_0", "q4_k_m"], token=os.environ["HF_TOKEN"])
wandb.finish()""",
    ),
]


VLLM_CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Serve the fine-tune with vLLM on Colab

The merged 16-bit model from `finetune.ipynb`, served by [vLLM](https://docs.vllm.ai) behind an OpenAI-compatible
API on a free T4, and opened to your machine with a Cloudflare quick tunnel, so doc-extraction-copilot's eval can
score it like the Ollama build:

```bash
AI_PROVIDER=local AI_BASE_URL=https://<tunnel>.trycloudflare.com/v1 AI_MODEL=doc-extraction-2b npx tsx scripts/eval.ts --delay 0
```

Set `HF_USER` below and run everything. Add `HF_TOKEN` as a secret if the model repo is private.""",
    ),
    ("code", """!pip install -qqq vllm
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared"""),
    (
        "code",
        """import os, subprocess, time, urllib.request

HF_USER = "your-hf-username"  # <- yours
MODEL = f"{HF_USER}/qwen3.5-2b-doc-extraction-br"

try:
    from google.colab import userdata

    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN") or ""
except Exception:
    pass

# The T4 has no bfloat16, so the weights load as float16. 8k tokens is what a two-page DANFE needs, as on Ollama
server = subprocess.Popen(
    ["vllm", "serve", MODEL, "--served-model-name", "doc-extraction-2b", "--dtype", "half",
     "--max-model-len", "8192", "--limit-mm-per-prompt", '{"image": 4}', "--gpu-memory-utilization", "0.9"],
    stdout=open("vllm.log", "w"), stderr=subprocess.STDOUT,
)
for _ in range(120):
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models")
        break
    except Exception:
        time.sleep(5)
else:
    raise RuntimeError(open("vllm.log").read()[-3000:])
print("vLLM is up")""",
    ),
    (
        "code",
        """import re

# A quick tunnel needs no Cloudflare account; the URL changes on each run
tunnel = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8000"], stderr=subprocess.PIPE, text=True)
for line in tunnel.stderr:
    if m := re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line):
        print("AI_BASE_URL=" + m.group(0) + "/v1")
        break""",
    ),
    (
        "markdown",
        """Keep this tab open while the eval runs: the server stops with the runtime. The request body from the app
turns thinking off with `chat_template_kwargs`, which vLLM passes to the chat template.""",
    ),
]


def cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    return base if kind == "markdown" else {**base, "execution_count": None, "outputs": []}


def write(cells: list[tuple[str, str]], name: str) -> None:
    notebook = {
        "cells": [cell(k, s) for k, s in cells],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = Path(__file__).parent.parent / "notebooks" / name
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {out}")


write(CELLS, "finetune.ipynb")
write(VLLM_CELLS, "serve_vllm.ipynb")
