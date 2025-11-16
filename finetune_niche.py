import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# =============================
# CONFIG
# =============================
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA = "dataset_chat.jsonl"
OUT_DIR = "tiny_finetuned_niche"  # çıkış klasörü

# ========================AA=====
# 1️⃣ Dataset yükle ve hazırla
# =============================
ds = load_dataset("json", data_files=DATA)["train"]

# Eğer dataset zaten "text" alanına sahipse, formatlama gerekmez
if "text" not in ds.column_names:
    def format_sample(example):
        return {
            "text": f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"
        }
    ds = ds.map(format_sample)

print(f"✅ Dataset loaded with {len(ds)} examples. Fields: {ds.column_names}")

# =============================
# 2️⃣ Tokenizer ve Model
# =============================
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("🚀 Loading model in FP16 mode...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.float16,  # FP16 mod, Windows için en kararlı
)

# =============================
# 3️⃣ LoRA yapılandırması
# =============================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print("✅ LoRA adapters attached")

# =============================
# 4️⃣ Tokenizasyon
# =============================
def tokenize_fn(ex):
    return tokenizer(ex["text"], truncation=True, max_length=512)

tok_ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# =============================
# 5️⃣ Training ayarları
# =============================
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,  # FP16 eğitim
    save_total_limit=1,
    logging_steps=5,
    report_to="none",
    overwrite_output_dir=True
)

# =============================
# 6️⃣ Trainer
# =============================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds,
    data_collator=collator,
)

print("💪 Training started...\n")
trainer.train()

# =============================
# 7️⃣ Modeli kaydet
# =============================
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"✅ Fine-tuning complete! Model saved to: {OUT_DIR}")
