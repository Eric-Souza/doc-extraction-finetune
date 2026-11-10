"""Trains the same QLoRA as notebooks/finetune.ipynb on a local 8 GB GPU, from data/ instead of the Hub, with the
loss written to runs/ instead of W&B. Run it in the venv from scripts/setup_wsl.sh:

    python scripts/train_local.py --out ~/ft/out

It saves the LoRA adapter and a merged 16-bit model under --out, and the loss history, a loss chart and the
held-out replies under runs/.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from unsloth import FastVisionModel  # noqa: I001, Unsloth has to be imported before transformers and trl
import torch
from datasets import Dataset
from PIL import Image
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth.trainer import UnslothVisionDataCollator

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
BASE_MODEL = "unsloth/Qwen3.5-2B"

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True, help="where the adapter and the merged model go")
parser.add_argument("--max-steps", type=int, default=-1, help="stop early, for a smoke test")
args = parser.parse_args()
args.out = args.out.expanduser()
RUNS.mkdir(exist_ok=True)

rows = {s: [json.loads(line) for line in (DATA / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in ("train", "val")}


def to_chat(row: dict, with_answer: bool = True) -> list[dict]:
    images = [Image.open(DATA / p).convert("RGB") for p in row["images"]]
    # The order the app's OpenAI-compatible provider sends: the text, then the images
    messages = [
        {"role": "system", "content": [{"type": "text", "text": row["system"]}]},
        {"role": "user", "content": [{"type": "text", "text": row["prompt"]}, *({"type": "image", "image": i} for i in images)]},
    ]
    if with_answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]})
    return messages


def chats(rows: list[dict]) -> Dataset:
    """A Dataset that opens the images only when a batch needs them"""
    # json.dumps keeps the rows flat for Arrow; to_chat gets them back whole
    table = Dataset.from_list([{"row": json.dumps(r, ensure_ascii=False)} for r in rows])
    return table.with_transform(lambda batch: {"messages": [to_chat(json.loads(r)) for r in batch["row"]]})


class LossLog(TrainerCallback):
    """Writes every logged loss to runs/loss.csv as training goes"""

    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["step", "loss", "eval_loss", "learning_rate"])

    def on_log(self, _args, state, _control, logs=None, **_kwargs):
        logs = logs or {}
        if "loss" in logs or "eval_loss" in logs:
            self.writer.writerow([state.global_step, logs.get("loss", ""), logs.get("eval_loss", ""), logs.get("learning_rate", "")])
            self.file.flush()


model, processor = FastVisionModel.from_pretrained(BASE_MODEL, load_in_4bit=True, use_gradient_checkpointing="unsloth")
model = FastVisionModel.get_peft_model(
    model,
    # As in the notebook: the base model already reads the digits, so only the language layers train
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# The served prompt must be where the training chat starts (Ollama and vLLM are asked with thinking off)
sample = to_chat(rows["train"][0])
served = processor.apply_chat_template(sample[:2], add_generation_prompt=True, enable_thinking=False, tokenize=False)
trained = processor.apply_chat_template(sample, enable_thinking=False, tokenize=False)
assert trained.startswith(served), "The training chat does not start with the served prompt"

FastVisionModel.for_training(model)
collator = UnslothVisionDataCollator(
    model,
    processor,
    resize="max",  # no resizing: the served model sees the page at full size, so training must too
    train_on_responses_only=True,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
    chat_template_kwargs={"enable_thinking": False},
)
trainer = SFTTrainer(
    model=model,
    tokenizer=processor,
    data_collator=collator,
    train_dataset=chats(rows["train"]),
    eval_dataset=chats(rows["val"]),
    callbacks=[LossLog(RUNS / "loss.csv")],
    args=SFTConfig(
        # The notebook's batch of 8 (2 x 4) as 1 x 8, to fit in 8 GB
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        warmup_steps=7,  # 5% of the 145 steps
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=0.001,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        seed=3407,
        output_dir=str(args.out / "checkpoints"),
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=4096,
    ),
)
start = time.time()
stats = trainer.train()
summary = {
    "base_model": BASE_MODEL,
    "steps": stats.global_step,
    "train_minutes": round(stats.metrics["train_runtime"] / 60, 1),
    "train_loss": round(stats.metrics["train_loss"], 4),
    "peak_gpu_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    "gpu": torch.cuda.get_device_name(),
}
print(summary)

model.save_pretrained(args.out / "lora")
processor.save_pretrained(args.out / "lora")


# A first look on the held-out chats, as in the notebook. The real eval runs in doc-extraction-copilot
def leaves(value, prefix=""):
    if isinstance(value, list):
        return {k: v for i, x in enumerate(value) for k, v in leaves(x, f"{prefix}[{i}]").items()}
    if isinstance(value, dict):
        return {k: v for key, x in value.items() if key != "uncertain" for k, v in leaves(x, f"{prefix}.{key}" if prefix else key).items()}
    return {prefix: value}


FastVisionModel.for_inference(model)
right = total = parsed = 0
checked = rows["val"] if args.max_steps < 0 else rows["val"][:4]
with (RUNS / "val_replies.jsonl").open("w", encoding="utf-8") as out:
    for row in checked:
        messages = to_chat(row, with_answer=False)
        images = [part["image"] for part in messages[1]["content"] if part["type"] == "image"]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        inputs = processor(images=images, text=[text], return_tensors="pt").to("cuda")
        generated = model.generate(**inputs, max_new_tokens=1500, do_sample=False, use_cache=True)
        reply = processor.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        truth = leaves(json.loads(row["answer"]))
        try:
            got, ok = leaves(json.loads(reply)), True
        except json.JSONDecodeError:
            got, ok = {}, False
        n = sum(got.get(k) == v for k, v in truth.items())
        right, total, parsed = right + n, total + len(truth), parsed + ok
        out.write(json.dumps({"id": row["id"], "task": row["task"], "parsed": ok, "fields_right": n, "fields": len(truth), "reply": reply}, ensure_ascii=False) + "\n")
summary |= {"val_parsed": f"{parsed}/{len(checked)}", "val_field_match": round(right / total, 4)}
print(summary)
(RUNS / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

# The merged 16-bit model, which llama.cpp converts to GGUF for Ollama
model.save_pretrained_merged(str(args.out / "merged"), processor, save_method="merged_16bit")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = list(csv.DictReader((RUNS / "loss.csv").open(encoding="utf-8")))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([int(r["step"]) for r in log if r["loss"]], [float(r["loss"]) for r in log if r["loss"]], label="train loss")
ax.plot([int(r["step"]) for r in log if r["eval_loss"]], [float(r["eval_loss"]) for r in log if r["eval_loss"]], "o-", label="held-out loss")
ax.set_xlabel("step (8 chats each)")
ax.set_ylabel("loss")
ax.set_title(f"QLoRA on {BASE_MODEL}, {summary['gpu']}")
ax.legend()
fig.tight_layout()
fig.savefig(RUNS / "loss.png", dpi=150)
print("done in", round((time.time() - start) / 60, 1), "min")
