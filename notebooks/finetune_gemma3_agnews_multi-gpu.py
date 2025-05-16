#!/usr/bin/env python
"""
Fine-tune Gemma-3-12B-IT on AG-News via 4-bit QLoRA.

Launch:

export CUDA_VISIBLE_DEVICES=1,2,3 
    accelerate launch \
    --num_processes 2 \
    --multi_gpu \
    finetune_gemma3_agnews_multi-gpu.py 
"""

import os
import random

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ─── distributed init ─────────────────────────────────────────────
local_rank = int(os.getenv("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device_str = f"cuda:{local_rank}"

# ─── constants ────────────────────────────────────────────────────
BASE_ID = "google/gemma-3-12b-it"
OUT_DIR = "gemma3-agnews-lora"
RAND_SEED = 42
SEQ_LEN = 512
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)

# ─── 4-bit base model ─────────────────────────────────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=torch.float32,
    attn_implementation="eager",
    device_map={"": device_str},
    quantization_config=bnb_cfg,
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
tokenizer.pad_token = tokenizer.eos_token  # silence warning

# ─── LoRA config ─────────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


# ─── dataset → messages ──────────────────────────────────────────
def to_chatml(sample):
    system_msg = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a helpful assistant. "
                "Answer with exactly one label from "
                "[World, Sports, Business, Sci/Tech].",
            }
        ],
    }

    user_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Classify the following news article:\n\n{sample['text']}"}
        ],
    }

    assistant_msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": LABELS[int(sample["label"])]}],
    }

    return {"messages": [system_msg, user_msg, assistant_msg]}


raw = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=RAND_SEED)
# raw = raw.select(range(500))  # demo size
train_ds = raw.map(to_chatml, remove_columns=raw.column_names)

# ─── SFT hyper-params ────────────────────────────────────────────
sft_cfg = SFTConfig(
    output_dir=OUT_DIR,
    max_seq_length=SEQ_LEN,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=1e-4,
    bf16=False,
    fp16=True,
    max_grad_norm=1.0,
    optim="paged_adamw_8bit",
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=25,
    save_strategy="epoch",
    report_to="all",
    seed=RAND_SEED,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=True,
    dataset_kwargs={
        "add_special_tokens": False,  # tokenizer already has template
        "append_concat_token": True,
    },
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# ─── build & train ───────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    peft_config=lora_cfg,
    args=sft_cfg,
    processing_class=tokenizer,
)
trainer.train()

# ─── save adapter & tokenizer ────────────────────────────────────
trainer.model.save_pretrained(f"{OUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUT_DIR}/adapter")
print(f"\n✓ LoRA adapter written to {OUT_DIR}/adapter")
