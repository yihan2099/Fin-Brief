#!/usr/bin/env python
"""
Evaluate Gemma-3-12B-IT + LoRA adapter on AG-News test set.
Assumes `gemma3-agnews-lora/adapter` was produced by your training job.
"""

import os, torch, json, re
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, GenerationConfig)
from peft import PeftModel
from evaluate import load as load_metric

# ─── constants ────────────────────────────────────────────────────
BASE_ID   = "google/gemma-3-12b-it"
ADAPTER   = "gemma3-agnews-lora/adapter"
LABELS    = ["World", "Sports", "Business", "Sci/Tech"]
MAX_NEW   = 4                       # only need one word but be safe
BATCH_SZ  = 4                       # tune for VRAM

# ─── load 4-bit base + adapter ───────────────────────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
base      = AutoModelForCausalLM.from_pretrained(
    BASE_ID, quantization_config=bnb_cfg, device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER).eval()

# ─── helper: build prompt identical to training ───────────────────
def build_prompt(text: str) -> str:
    msgs = [
        {
            "role": "system",
            "content": [{"type": "text",
                         "text": ("You are a helpful assistant. "
                                  "Answer with exactly one label from "
                                  "[World, Sports, Business, Sci/Tech].")}],
        },
        {
            "role": "user",
            "content": [{"type": "text",
                         "text": f"Classify the following news article:\n\n{text}"}],
        },
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False)

pattern = re.compile(r"^\s*([A-Za-z/]+)")

# ─── load & preprocess test set ───────────────────────────────────
ds = load_dataset("fancyzhx/ag_news", split="test")
prompts = [build_prompt(t) for t in ds["text"]]
targets = [LABELS[l] for l in ds["label"]]

# ─── batched generation ───────────────────────────────────────────
gen_cfg = GenerationConfig(max_new_tokens=MAX_NEW,
                           do_sample=False,      # greedy
                           pad_token_id=tokenizer.eos_token_id)

preds = []
for i in range(0, len(prompts), BATCH_SZ):
    batch = prompts[i:i+BATCH_SZ]
    inputs = tokenizer(batch, return_tensors="pt",
                       padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    decoded = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
    for s in decoded:
        m = pattern.match(s)
        preds.append(m.group(1) if m else "INVALID")

# ─── compute metrics ──────────────────────────────────────────────
accuracy = load_metric("accuracy")
f1       = load_metric("f1")

acc = accuracy.compute(predictions=preds, references=targets)["accuracy"]
f1m = f1.compute(predictions=preds, references=targets,
                 average="macro")["f1"]

print(f"Accuracy : {acc:.4f}")
print(f"Macro-F1 : {f1m:.4f}")