import json

# Girdi dosyan (eski dataset)
INPUT_FILE = "dataset_niche.jsonl"
# Çıktı dosyan (TinyLlama Chat format)
OUTPUT_FILE = "dataset_chat.jsonl"

# Sistem rolü (modelin genel davranışını tanımlar)
SYSTEM_PROMPT = "You are a helpful assistant that answers questions about embedded systems, electronics, and signal processing clearly and accurately."

converted = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        instruction = ex.get("instruction", "").strip()
        output = ex.get("output", "").strip()

        # TinyLlama chat formatına dönüştür
        text = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{instruction}\n"
            f"<|assistant|>\n{output}"
        )
        converted.append({"text": text})

# Yeni dosyayı kaydet
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ex in converted:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"✅ Converted {len(converted)} examples to TinyLlama Chat format")
print(f"💾 Saved as: {OUTPUT_FILE}")
