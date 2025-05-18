#!/usr/bin/env python
"""
Fine‑tune Gemma‑3‑12B‑IT on AG‑News via 4‑bit QLoRA **with periodic evaluation**
--------------------------------------------------------------------------
This script is the **exact** training configuration used in
`finetune_gemma3_agnews_multi‑gpu.py`, plus the extra logic required to:
  • run validation at the end of every epoch;
  • report accuracy, precision, recall, F1; and
  • compute perplexity from the validation loss.

Launch (two GPUs):
    export CUDA_VISIBLE_DEVICES=2,3
    accelerate launch --num_processes 2 --multi_gpu train_eval_gemma3_agnews.py
"""

import math
import os
import random

from datasets import load_dataset
import evaluate
import numpy as np
from peft import LoraConfig, prepare_model_for_kbit_training
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ─── Distributed initialisation ───────────────────────────────────────────
local_rank = int(os.getenv("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
DEVICE_STR = f"cuda:{local_rank}"

# ─── Reproducibility ─────────────────────────────────────────────────────
RAND_SEED = 42
random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)

# ─── Constants ───────────────────────────────────────────────────────────
BASE_ID = "google/gemma-3-12b-it"
OUT_DIR = "gemma3-agnews-lora"
SEQ_LEN = 512
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# ─── Evaluation metrics ─────────────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_preds):
    """Decode the very last token of each sample and compute metrics."""
    logits, labels = eval_preds
    pred_ids = np.argmax(logits, axis=-1)
    preds = [
        tokenizer.decode(pred_ids[i, -1:], skip_special_tokens=True).strip()
        for i in range(pred_ids.shape[0])
    ]
    refs = [
        tokenizer.decode(labels[i, -1:], skip_special_tokens=True).strip()
        for i in range(labels.shape[0])
    ]
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


# ─── Dataset helpers ────────────────────────────────────────────────────


def to_chatml(sample):
    """AG‑News row → ChatML conversation suitable for Gemma."""

    system = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a helpful assistant. Answer with exactly one label from [World, Sports, Business, Sci/Tech].",
            }
        ],
    }
    user = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Classify the following news article:\n\n{sample['text']}",
            }
        ],
    }
    assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": LABELS[int(sample["label"])]}],
    }
    return {"messages": [system, user, assistant]}


# ─── Load & split datasets (same as training‑only script) ───────────────
raw_train = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=RAND_SEED)
raw_valid = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=RAND_SEED)
train_ds = raw_train.map(to_chatml, remove_columns=raw_train.column_names)
valid_ds = raw_valid.map(to_chatml, remove_columns=raw_valid.column_names)

# ─── 4‑bit base model (identical hyper‑params) ───────────────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32,  # ‼ matches training‑only script
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=torch.float32,  # ‼ matches training‑only script
    attn_implementation="eager",  # ‼ matches training‑only script
    device_map={"": DEVICE_STR},  # ‼ matches training‑only script
    quantization_config=bnb_cfg,
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# ─── Tokeniser ──────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
tokenizer.pad_token = tokenizer.eos_token  # silence warnings

# ─── LoRA config (unchanged) ────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# ─── SFT training arguments — EXACT copy + evaluation hooks ─────────────
sft_cfg = SFTConfig(
    output_dir=OUT_DIR,
    max_seq_length=SEQ_LEN,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,  # ‼ same as training‑only script
    learning_rate=1e-4,
    bf16=False,
    fp16=True,
    max_grad_norm=1.0,
    optim="paged_adamw_8bit",
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=25,
    save_strategy="epoch",  # ‼ same strategy
    evaluation_strategy="epoch",  # ← added
    compute_metrics=compute_metrics,  # ← added
    report_to="all",
    seed=RAND_SEED,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=True,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": True,
    },
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# ─── Build trainer ──────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=valid_ds,  # ← added
    peft_config=lora_cfg,
    args=sft_cfg,
    processing_class=tokenizer,
)

# ─── Train & evaluate ───────────────────────────────────────────────────
trainer.train()
print("\n✅ Training complete. Evaluating on validation set …")
results = trainer.evaluate()
results["eval_perplexity"] = math.exp(results["eval_loss"])
print({k: v for k, v in results.items() if k.startswith("eval_")})

# ─── Save adapter & tokenizer ───────────────────────────────────────────
trainer.model.save_pretrained(f"{OUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUT_DIR}/adapter")
print(f"\n✓ LoRA adapter & tokenizer saved to {OUT_DIR}/adapter")
