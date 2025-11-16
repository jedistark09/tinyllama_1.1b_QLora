import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("🚀 Loading TinyLlama model (first time download may take a few minutes)...")

# Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.float16,   # FP16 mode (4bit kapalı)
)
model.eval()

# Başlangıç mesajı (system rolü)
system_prompt = "You are a helpful and friendly AI assistant. Answer concisely and clearly."
messages = [{"role": "system", "content": system_prompt}]

def generate_reply(messages, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """Chat template ile modelden yanıt üretir"""
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        inputs=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.1,
        streamer=streamer
    )

    import threading
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    reply_text = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        reply_text += new_text
    print()
    thread.join()
    return reply_text.strip()

print("\n✅ TinyLlama is ready to chat! Type '/exit' to quit.\n")

# Sohbet döngüsü
while True:
    try:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in {"/exit", "exit", "quit", "q"}:
            print("👋 Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        print("🤖 TinyLlama: ", end="", flush=True)
        reply = generate_reply(messages)
        messages.append({"role": "assistant", "content": reply})

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Exiting...")
        break
