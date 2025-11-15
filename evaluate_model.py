#!/usr/bin/env python3
import json, re, torch, textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------- ustawienia --------
BASE_MODEL   = "deepseek-ai/deepseek-coder-1.3b-instruct"
ADAPTER_PATH = "./lora-deepseek-1.3b"
DATA_PATH    = "dane_prompt.jsonl"
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_TOKENS   = 50          # tylko jedno słowo
STOP_AFTER   = 10      # np. 5 → przetnie po pięciu próbkach

CATS = [
    "BD","Zasilacz","Tłumienie","Linia światłowodowa","Router","Radiówka",
    "Core","Sprzet","Po stronie klienta","Kabel","SGT","Aktywacja",
    "Konfiguracja","Awaria globalna","EPIX","WiFi","Zapytanie",
    "Sieć energetyczna","Zmiana hasła","tSEC","BOK","Dubel","Odwołanie"
]
CAT_LIST = ", ".join(CATS)

# -------- model --------
tok   = AutoTokenizer.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.float16, device_map="auto"),
            ADAPTER_PATH)
print("[debug] Załadowane adaptery:", model.peft_config, "\n")

# -------- test --------
total = correct = 0
print("\n=== TESTY Z INSTRUKCJĄ „JEDNO SŁOWO” ===\n")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if STOP_AFTER and idx > STOP_AFTER:
            break
        if not line.strip():
            continue
        row       = json.loads(line)
        base_prompt = row["prompt"].rstrip()
        expected    = row["completion"].strip()

        # usuwamy ewentualny stary “Przyczyna:” (jeśli był w prompt-ach)
        base_prompt = re.sub(r"Przyczyna:\s*$", "", base_prompt, flags=re.I)

        # nowa instrukcja
        prompt = (
            f"{base_prompt}\n\n"
            "Wybierz JEDNO słowo z poniższej listy kategorii, które najlepiej opisuje przyczynę.\n"
            "Napisz tylko to słowo – bez dodatkowego tekstu.\n\n"
            f"Kategorie: {CAT_LIST}\n\nPrzyczyna:"
        )

        # --- generacja ---
        inp     = tok(prompt, return_tensors="pt").to(DEVICE)
        out_ids = model.generate(**inp,
                                 max_new_tokens=MAX_TOKENS,
                                 do_sample=False, temperature=0.0,
                                 pad_token_id=tok.eos_token_id)

        full_out  = tok.decode(out_ids[0], skip_special_tokens=True)
        added_ids = out_ids[0][inp["input_ids"].shape[-1]:]
        added     = tok.decode(added_ids, skip_special_tokens=True).strip()

        # pierwsze słowo wygenerowane przez model
        word = added.split()[0] if added else ""
        # mapowanie dokładne (model powinien trafić 1:1)
        prediction = word if word in CATS else "(brak)"

        ok = prediction.lower() == expected.lower()
        total   += 1
        correct += ok

        # -------- DEBUG --------
        print(f"\n=== PRÓBKA {idx} ===============================")
        print("PROMPT ↓")
        print(textwrap.indent(prompt, "  "))
        print("\nFULL MODEL OUTPUT ↓")
        print(textwrap.indent(full_out, "  "))
        print("\nADDED (po prompt-cie) ↓")
        print(textwrap.indent(added or "(pusto)", "  "))
        print(f"\nprediction = '{prediction}'   |   expected = '{expected}'   →   {'✅' if ok else '❌'}")
        print("===============================================\n")

print(f"\n🎯 Trafność: {correct}/{total} = {correct/total*100:.2f}%")
