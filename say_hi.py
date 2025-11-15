#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------- ustawienia --------
BASE_MODEL    = "deepseek-ai/deepseek-coder-1.3b-instruct"
ADAPTER_PATH  = "./lora-deepseek-1.3b"
DEVICE        = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE         = torch.float16 if DEVICE.type == "mps" else torch.float32
MAX_TOKENS    = 100

# Wymuś CPU, jeśli chcesz – odkomentuj poniższe dwie linie:
# DEVICE = torch.device("cpu")
# DTYPE = torch.float32

# -------- tokenizer --------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# -------- model + LoRA adapter --------
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(DEVICE)
model.eval()
CATS = [
    "BD","Zasilacz","Tłumienie","Linia światłowodowa","Router","Radiówka",
    "Core","Sprzet","Po stronie klienta","Kabel","SGT","Aktywacja",
    "Konfiguracja","Awaria globalna","EPIX","WiFi","Zapytanie",
    "Sieć energetyczna","Zmiana hasła","tSEC","BOK","Dubel","Odwołanie"
]
CAT_LIST = ", ".join(CATS)

# -------- prompt --------
prompt = (
    f"Wybierz JEDNO słowo z poniższej listy kategorii, które najlepiej opisuje przyczynę.\n"
    "Napisz tylko to słowo – bez dodatkowego tekstu.\n\n"
    f"Kategorie: {CAT_LIST}\n"
    "Temat: Callcenter - internet\n"
    "Rozmowa:\n"
    "Klient: Klientka zgłasza, że kabel Internetowy został przegryziony przez królika i prosi o serwis.\n"
    "Technik: Klient otrzymał nowy patchcord.\n"
    "\n"
    "Przyczyna:"
)

# -------- generacja --------
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# -------- wynik --------
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = generated_text[len(prompt):].strip()

print("\n🧠 Odpowiedź modelu:\n")
print(response or "(brak odpowiedzi)")
