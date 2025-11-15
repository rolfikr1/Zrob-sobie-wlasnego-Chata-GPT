## local-llm — fine-tuning LoRA (deepseek-coder-1.3b-instruct, macOS/MPS/CPU)

Ten projekt pokazuje prosty pipeline trenowania adapterów LoRA na modelu `deepseek-ai/deepseek-coder-1.3b-instruct` do klasyfikacji „jednym słowem” na podstawie rozmowy. Skrypty są gotowe do uruchomienia lokalnie (Apple Silicon MPS lub CPU).

- **Trenowanie**: `train_lora.py` (Transformers + Datasets + PEFT)
- **Konwersja danych**: `convert_to_prompt.py` → `dane_prompt.jsonl`
- **Ewaluacja**: `evaluate_model.py` (trafność top-1 na pliku wejściowym)
- **Demo inferencji**: `say_hi.py` oraz `test_model.py`
- **Wyjście treningu**: katalog `lora-deepseek-1.3b/` z adapterami i checkpointami

---

### Wymagania
- Python 3.11–3.13
- macOS 14+ (MPS) lub Linux/CPU
- Pakiety: `torch`, `transformers`, `datasets`, `peft`, `safetensors` (opcjonalnie `accelerate`)

Przykładowa instalacja (zalecany wirtualny environment):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers datasets peft safetensors
```

Na Apple Silicon (MPS) najnowsze stabilne `torch` z PyPI zwykle wystarcza. Jeśli MPS nie jest dostępny, skrypty automatycznie przełączą się na CPU.

#### Jak wejść / wyjść z wirtualnego środowiska (venv)
- Zsh/Bash (macOS/Linux) – aktywacja:
  ```bash
  source .venv/bin/activate
  ```
- Jeśli w repo masz już istniejący `venv-ollama/`, możesz go użyć:
  ```bash
  source venv-ollama/bin/activate
  ```
- Fish:
  ```fish
  source .venv/bin/activate.fish
  ```
- PowerShell (Windows):
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- Wyjście z venv:
  ```bash
  deactivate
  ```
- Ponowne wejście (po restarcie terminala): przejdź do katalogu projektu i ponownie wykonaj komendę aktywacji odpowiednią dla Twojej powłoki.

---

### Dane wejściowe
Do treningu oczekujemy pliku `dane_prompt.jsonl` z wierszami JSON o polach:
- `prompt`: treść instrukcji/wejścia
- `completion`: oczekiwane JEDNO słowo-kategoria (z poprzedzającą spacją, np. `" Zasilacz"`)

Generator `dane_prompt.jsonl` bazuje na źródle `dane.json` o strukturze:

```json
{
  "id_zgloszenia": 999999,
  "id_klienta": 123456,
  "temat": "Problem z Internetem",
  "rozmowa": [
    "Klient: Od rana nie działa internet, router miga na czerwono",
    "Technik: Wymiana zasilacza"
  ],
  "przyczyna": "Zasilacz"
}
```

Konwersja do formatu treningowego:

```bash
python convert_to_prompt.py
```

Wynik: `dane_prompt.jsonl` z rekordami typu:

```json
{"prompt": "ID zgłoszenia: ...\nID klienta: ...\nTemat: ...\nRozmowa:\n...\n\nMożliwe przyczyny: BD, Zasilacz, ...\n\nWybierz jedną przyczynę ...", "completion": " Zasilacz"}
```

---

### Trenowanie (LoRA)
Uruchom:

```bash
python train_lora.py
```

Domyślne ustawienia:
- Model bazowy: `deepseek-ai/deepseek-coder-1.3b-instruct`
- Urządzenie: MPS (jeśli dostępne) → CPU fallback
- Tokenizacja: `max_length=512` (możesz dostosować)
- LoRA: `r=8`, `lora_alpha=32`, `lora_dropout=0.1`
- Trening: batch size 1, gradient accumulation 2, epoki 3
- Zapisy: `./lora-deepseek-1.3b/` oraz `./lora-deepseek-1.3b/checkpoint-*`

Zmiana modelu bazowego:
- Edytuj zmienną `model_name` w `train_lora.py`
- Dla ewaluacji/inferencji zmień `BASE_MODEL` w pozostałych skryptach

Wskazówki wydajnościowe (MPS/CPU):
- Jeśli pamięć jest ograniczona, zwiększ `gradient_accumulation_steps` zamiast `per_device_train_batch_size`
- Dostosuj `max_length` w tokenizacji do długości kontekstu danych

##### Jak uczyć „odpowiedz jednym słowem”
- Dane: `completion` powinno być dokładnie jednym słowem (np. `" Zasilacz"`).
- Maskowanie promptu: uczymy tylko na odpowiedzi (prompt ma `labels = -100`), więc model skupia się na tym, co ma wypisać.
- Szybkie stopowanie: dodaj `eos_token` zaraz po etykiecie, by model kończył generację po jednym słowie.
- Podczas generacji: ustaw małe `max_new_tokens` (np. 5), `do_sample=False` dla deterministycznej odpowiedzi i opcjonalnie bierz pierwsze słowo z wyjścia.

#### Co oznaczają logi treningu (prosto i po ludzku)
- **loss (strata)**: mówi, jak bardzo model się myli w danym momencie. Myśl o tym jak o „temperaturze błędu” — im niższa, tym lepiej. Jeśli loss spada z czasem, idziemy w dobrą stronę. Gdy pojawia się `nan` lub `inf`, zwykle tempo nauki jest zbyt duże albo są problemy z danymi/typami liczb.
- **grad_norm (siła kroku nauki)**: jak duży „krok” model robi podczas uczenia w danym momencie. Pojedyncze, bardzo duże skoki mogą oznaczać, że kroki są za agresywne („exploding gradients”). Co wtedy? Spróbuj:
  - obniżyć `learning_rate`,
  - ustawić niższe `max_grad_norm` (ogranicza wielkość kroku),
  - zwiększyć `gradient_accumulation_steps` (mniejsze batch’e, stabilniejsze kroki).
- **learning_rate (tempo nauki)**: jak szybko model uczy się na błędach. Za duże → szybciej, ale ryzyko rozjechania (loss rośnie, `nan`). Za małe → stabilnie, ale wolniej. W logach ta wartość zwykle stopniowo maleje, bo działa harmonogram (scheduler).
- **epoch (przejście przez dane)**: jedno „okrążenie” po całym zbiorze danych. Liczba może być ułamkowa, bo logujemy co krok; `1.0` znaczy, że model przeszedł całość jeden raz.

---

### Ewaluacja
Szybki test trafności na danych z `dane_prompt.jsonl`:

```bash
python evaluate_model.py
```

Parametry w pliku:
- `BASE_MODEL` – model bazowy
- `ADAPTER_PATH` – ścieżka do adapterów (np. `./lora-deepseek-1.3b`)
- `STOP_AFTER` – ograniczenie liczby przykładów (0 = wszystkie)

Skrypt:
- Ładuje bazowy model + adapter PEFT,
- Generuje odpowiedź (bez sampling’u; `temperature=0.0`),
- Porównuje pierwsze wygenerowane słowo z etykietą.

---

### Demo inferencji
Krótka generacja „jednym słowem” z gotowym promptem:

```bash
python say_hi.py
```

Alternatywny, prosty test:

```bash
python test_model.py
```

Uwaga: `test_model.py` wskazuje na `./lora-deepseek` — dostosuj do faktycznej ścieżki adaptera (np. `./lora-deepseek-1.3b`).

---

### Jak ręcznie załadować adapter LoRA w swoim kodzie
Przykład (Python):

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL   = "deepseek-ai/deepseek-coder-1.3b-instruct"
ADAPTER_PATH = "./lora-deepseek-1.3b"
DEVICE       = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE        = torch.float16 if DEVICE.type == "mps" else torch.float32

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE)
model = PeftModel.from_pretrained(base, ADAPTER_PATH).to(DEVICE).eval()
```

---

### Co znajduje się w repo
- `convert_to_prompt.py` – konwersja `dane.json` → `dane_prompt.jsonl`
- `train_lora.py` – trening adapterów LoRA
- `evaluate_model.py` – ewaluacja dokładności (pierwsze słowo vs etykieta)
- `say_hi.py` – szybkie demo generacji
- `test_model.py` – prosty test promptu

---

### Najczęstsze problemy
- Brak MPS: skrypty przełączą się na CPU (wolniej). Upewnij się, że `torch.backends.mps.is_available()` zwraca `True`, jeśli chcesz używać GPU Apple.
- Za długi kontekst: dopasuj `max_length` w tokenizacji (w `train_lora.py` → funkcja `tokenize`).
- Ścieżki adapterów: trzymaj spójne nazwy/ścieżki w `evaluate_model.py`, `say_hi.py`, `test_model.py`.
- Loss 0.0 i `grad_norm = NaN`: zwykle wszystkie etykiety są zamaskowane (`-100`) po przycięciu sekwencji. Rozwiązanie: w tokenizacji najpierw rezerwuj miejsce na odpowiedź (target) i dopiero potem przycinaj prompt (to już jest zaimplementowane w `train_lora.py`).

---

### Licencje i modele
Model `deepseek-ai/deepseek-coder-1.3b-instruct` pochodzi z Hugging Face. Upewnij się, że akceptujesz warunki licencji modelu i przestrzegasz ich w użyciu produkcyjnym.


