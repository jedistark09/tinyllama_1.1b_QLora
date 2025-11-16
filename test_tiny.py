import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model adı
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Downloading and loading TinyLlama (this can take a few minutes on first run)...")

# Tokenizer ve model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  # FP16 mod, 4bit kapalı
)

print("✅ Model loaded successfully!\n")

# Test prompt
prompt = "What is the capital of England?"

# Girdi tensoru oluştur
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Modelden çıktı al
output = model.generate(**inputs, max_new_tokens=50)

# Sonucu çöz ve yazdır
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("🧠 Model Output:\n", response)
