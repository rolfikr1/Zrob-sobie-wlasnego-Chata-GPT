from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Ścieżki
base_model = "deepseek-ai/deepseek-coder-1.3b-instruct"
adapter_path = "./lora-deepseek-1.3b"

# Wczytaj tokenizer i model z adapterem LoRA
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)

# === PROMPT DO TESTU ===
prompt = """Jesteś koordynatorem dzialu technicznego lokalnego operatora telekomunikacyjnego. Twoim zadaniem jest rozpoznanie przyczyny problemu zgloszonego przez klienta i odpowiedź jednym słowem, które najlepiej opisuje przyczynę.
W odpowiedzi podaj tylko jedną z listy możliwych przyczyn.

    "id_zgloszenia": "999999",
    "id_klienta": "999999",
    "temat": "brak usług",
    "rozmowa": [
      "Klient: klient zgłasza brak mozliwości korzystania z INT, proszę o weryfikację",
      "Technik: NAT nie był włączony na WANie. Usługa już działa poprawnie."
    ]

Możliwe przyczyny: BD, Zasilacz, Tłumienie, Linia światłowodowa, Router, Radiówka, Core, Sprzet, Po stronie klienta, Kabel, SGT, Aktywacja, Konfiguracja, Awaria globalna, EPIX, WiFi, Zapytanie, Sieć energetyczna, Zmiana hasła, tSEC, BOK, Dubel, Odwołanie

Przyczyna (podaj jedną z listy możliwych przyczyn):"""

# Tokenizacja i generowanie
inputs = tokenizer(prompt, return_tensors="pt").to("mps")
outputs = model.generate(**inputs, max_new_tokens=200)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🔍 Parsowanie wyniku
print("\n===== WYNIK MODELU =====")
print(result)
