import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FINE_PATH  = "./tiny_finetuned_niche"  # LoRA adapter klasörün veya tam model klasörü

PROMPTS = [
    "Describe in detail how DMA improves data transfer efficiency in embedded systems. Include a comparison between DMA-based and CPU-based transfer methods, and provide a practical example.",
    "Explain the role of Inter-Processor Communication (IPC) in multicore microcontrollers. Describe how data synchronization and message queues are managed between cores, and why this mechanism is essential for real-time performance.",
    "Walk through how a PID controller regulates a motor's speed. Explain each component (Proportional, Integral, Derivative) and how it affects system stability, response time, and steady-state error.",
    "Compare the Cortex-M4 and C28x cores commonly used in TI microcontrollers. Explain their architectural differences, typical application domains, and how they collaborate in mixed-control systems.",
    "Provide a step-by-step explanation of how FreeRTOS schedules multiple tasks. Discuss how priority-based preemption and time slicing affect responsiveness, and give a concrete example with three concurrent tasks.",
    "Explain in depth how a bootloader operates in a microcontroller. Include the startup sequence, firmware verification, and how OTA (Over-the-Air) firmware updates are handled safely.",
    "Discuss how QLoRA fine-tuning allows large language models to run on low-VRAM GPUs. Include an explanation of 4-bit quantization, low-rank adapters, and why this approach preserves model quality.",
    "Explain the purpose and operation of a Watchdog Timer in embedded systems. Describe how it detects system failures, how the timer is reset ('kicked'), and what happens if the watchdog expires.",
    "Describe how flash memory handles erase-before-write operations. Explain why flash cannot directly overwrite data, and how the Flash API ensures data integrity and timing control.",
    "Discuss why ADC calibration is important in precision applications. Explain how offset and gain errors arise, how calibration is performed, and how temperature or voltage drift can affect accuracy over time."
]

SYSTEM_PROMPT = "You are a precise embedded-systems assistant. Answer with correct facts, step-by-step reasoning, and a concrete example where relevant."

def load_tokenizer(model_or_dir):
    tok = AutoTokenizer.from_pretrained(model_or_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok

def load_base(model_name):
    tok = load_tokenizer(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16
    )
    return tok, mdl

def load_finetuned(base_name, fine_path):
    # Eğer PEFT adapter’ı ise base + adapter olarak yükle
    if os.path.exists(os.path.join(fine_path, "adapter_config.json")) and PEFT_AVAILABLE:
        tok = load_tokenizer(base_name)  # tokenizer base’ten
        base = AutoModelForCausalLM.from_pretrained(base_name, device_map="auto", dtype=torch.float16)
        mdl  = PeftModel.from_pretrained(base, fine_path)  # adapter’ı tak
        return tok, mdl
    else:
        # Tam model klasörü olarak kaydedildiyse direkt yükle
        tok = load_tokenizer(fine_path)
        mdl = AutoModelForCausalLM.from_pretrained(fine_path, device_map="auto", dtype=torch.float16)
        return tok, mdl

def generate_answer(tok, mdl, user_prompt, max_new_tokens=500, temperature=0.35, top_p=0.9):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    inp = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
    attn = (inp != tok.pad_token_id)

    gen = mdl.generate(
        inp,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    new_tokens = gen[0, inp.shape[-1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

# --- modelleri bir kez yükle ---
base_tok, base_mdl = load_base(BASE_MODEL)
fine_tok, fine_mdl = load_finetuned(BASE_MODEL, FINE_PATH)

for p in PROMPTS:
    print("="*90)
    print("Question:", p)
    print("\n--- Base Model ---")
    print(generate_answer(base_tok, base_mdl, p))
    print("\n--- Fine-tuned Model ---")
    print(generate_answer(fine_tok, fine_mdl, p))
