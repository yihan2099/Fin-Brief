from dbm import gnu
import os, torch, random, gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

'''
accelerate launch \
  --num_processes 4 \
  --multi_gpu \
  finetune_gemma3_agnews_multi-gnu.py 
'''

# ─── Grab the local_rank that Accelerate / torch.distributed set ─────────
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device_str = f"cuda:{local_rank}"

# ─── rest of your constants ───────────────────────────────────────────────
MODEL_ID  = "google/gemma-3-12b-it"
OUT_DIR   = "gemma3-agnews-lora"
RAND_SEED = 42
MAX_LEN   = 512
LABELS    = ["World", "Sports", "Business", "Sci/Tech"]
torch.manual_seed(RAND_SEED)

# ─── 4-bit quant config ───────────────────────────────────────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
)

# ─── load *one* 4-bit copy on the local GPU ───────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    attn_implementation="eager",
    device_map          = {"": device_str},
    quantization_config = bnb_cfg,
    trust_remote_code   = True,
)

# ─── now make it grad-ready + add LoRA ──────────────────
model = prepare_model_for_kbit_training(model)
# model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

lora_cfg = LoraConfig(
    r              = 16,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = "CAUSAL_LM",
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_cfg)

# ─── dataset + SFTConfig + SFTTrainer ────────────────────────────────────
def to_chatml(sample):
    user = (f"Classify the following news article into one of "
            f"[World, Sports, Business, Sci/Tech]:\n{sample['text']}")
    return {"messages": [
        {"role":"user",      "content":user},
        {"role":"assistant", "content":LABELS[int(sample["label"])]},
    ]}

raw     = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=RAND_SEED)
# …select or not…
train_ds= raw.map(to_chatml, remove_columns=raw.column_names)

sft_cfg = SFTConfig(
    output_dir                  = OUT_DIR,
    max_length                  = MAX_LEN,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 16,
    num_train_epochs            = 2,
    learning_rate               = 2e-4,
    bf16                        = True,
    optim                       = "paged_adamw_8bit",
    warmup_ratio                = 0.05,
    lr_scheduler_type           = "cosine",
    logging_steps               = 50,
    save_total_limit            = 2,
    report_to                   = "tensorboard",
    packing                     = True,
    seed                        = RAND_SEED,
    # ---------- critical for multi-GPU + LoRA ----------
    ddp_find_unused_parameters  = False,  # <-- stops the duplicate-hook error 
    gradient_checkpointing      = False,
    # (optional) make checkpointing non-re-entrant if you still get re-entrant warnings
    # gradient_checkpointing_kwargs = {"use_reentrant": False},
)

trainer = SFTTrainer(model=model,
                     train_dataset=train_ds,
                     args=sft_cfg)
trainer.train()

model.save_pretrained(f"{OUT_DIR}-adapter")
tokenizer.save_pretrained(f"{OUT_DIR}-adapter")
print(f"\n✓ Adapter written to → {OUT_DIR}-adapter")