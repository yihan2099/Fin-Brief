#!/usr/bin/env python
"""
Standalone evaluation of Gemma-3-12B-IT fine-tuned on AG-News via 4-bit QLoRA.
The script keeps **all runtime arguments identical** to training but adds a
lightweight evaluation loop that avoids GPU OOM by
  • slicing logits to the **last token only**; and
  • off-loading them to CPU before they are accumulated.

Launch (2 GPUs):
    export CUDA_VISIBLE_DEVICES=2,3
    accelerate launch --num_processes 2 --multi_gpu eval_gemma3_agnews.py
"""

import math
import os
import random

from datasets import load_dataset
import evaluate
import numpy as np
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ── Distributed init ────────────────────────────────────────────────────
local_rank = int(os.getenv("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
DEVICE_STR = f"cuda:{local_rank}"

# ── Reproducibility ─────────────────────────────────────────────────────
RAND_SEED = 42
random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)

# ── Constants ───────────────────────────────────────────────────────────
BASE_ID = "google/gemma-3-12b-it"
ADAPTER_DIR = "gemma3-agnews-lora/adapter"
OUT_DIR = "eval_gemma3_agnews_out"
SEQ_LEN = 512
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# ── Metrics ─────────────────────────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_preds):
    """Compute accuracy / precision / recall / F1 on the decoded label."""
    logits, labels = eval_preds

    # now move both to CPU / numpy for metric computation
    preds = logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    pred_ids = np.argmax(preds, axis=-1)

    preds = [LABELS[p] for p in pred_ids]
    refs = [LABELS[r] for r in labels]
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=refs)["accuracy"],
        "precision": precision_metric.compute(predictions=preds, references=refs, average="macro")[
            "precision"
        ],
        "recall": recall_metric.compute(predictions=preds, references=refs, average="macro")[
            "recall"
        ],
        "f1": f1_metric.compute(predictions=preds, references=refs, average="macro")["f1"],
    }


# ── Slice logits & move to CPU before accumulation ─────────────────────


def preprocess_logits_for_metrics(logits, labels):
    # slice out the last-token logits, but keep them on the GPU
    return logits[:, -1, :]


# ── AG-News → ChatML helper ─────────────────────────────────────────────


def to_chatml(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant. Answer with exactly one label from [World, Sports, Business, Sci/Tech].",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Classify the following news article:\n\n{sample['text']}",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": LABELS[int(sample["label"])]}],
            },
        ]
    }


# ── Validation dataset ─────────────────────────────────────────────────
raw_valid = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=RAND_SEED)
valid_ds = raw_valid.map(to_chatml, remove_columns=raw_valid.column_names)

# ── Base model (identical quant settings) ───────────────────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=torch.float32,
    attn_implementation="eager",
    device_map={"": DEVICE_STR},
    quantization_config=bnb_cfg,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

# ── Tokeniser ───────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
tokenizer.pad_token = tokenizer.eos_token

# ── SFTConfig (eval-only) ───────────────────────────────────────────────
sft_cfg = SFTConfig(
    output_dir=OUT_DIR,
    max_seq_length=SEQ_LEN,
    per_device_train_batch_size=1,  # dummy for SFTTrainer
    per_device_eval_batch_size=8,
    bf16=False,
    fp16=True,
    report_to="all",
    eval_accumulation_steps=1,  # flush every step to keep memory low
    logging_steps=50,
    seed=RAND_SEED,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": True,
    },
)

# ── Build trainer ───────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=valid_ds,  # harmless stub
    eval_dataset=valid_ds,
    args=sft_cfg,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# ── Evaluate ───────────────────────────────────────────────────────────
print("Running evaluation …")
results = trainer.evaluate()
results["eval_perplexity"] = math.exp(results["eval_loss"])
print({k: v for k, v in results.items() if k.startswith("eval_")})
