from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Wybór modelu
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"

# Automatyczny wybór urządzenia (MPS → CPU fallback)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float16 if device.type == "mps" else torch.float32

# Wczytaj dane (upewnij się że dane_prompt.jsonl istnieje)
dataset = load_dataset("json", data_files="dane_prompt.jsonl", split="train")

# Tokenizer i model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map={"": device.type},
    trust_remote_code=True
)

# Tokenizacja
def tokenize(example):
    # Budujemy wyraźny podział: PROMPT (bez etykiet) + ODPOWIEDŹ (z etykietami)
    source_text = f"{example['prompt'].rstrip()}\n\nPrzyczyna:"
    # Upewniamy się, że odpowiedź to jedno słowo + EOS (uczy model kończyć po etykiecie)
    target_word = example["completion"].strip()
    target_text = f" {target_word}{tokenizer.eos_token}"

    source_ids = tokenizer(
        source_text,
        add_special_tokens=False
    )["input_ids"]
    target_ids = tokenizer(
        target_text,
        add_special_tokens=False
    )["input_ids"]

    max_length = 512
    # Rezerwujemy miejsce na target: najpierw przytnij source tak, by target zawsze się zmieścił
    space_for_source = max_length - len(target_ids)
    if space_for_source < 0:
        # Skrajny przypadek: target dłuższy niż max_length → przytnij target
        target_ids = target_ids[:max_length]
        source_ids = []
    else:
        source_ids = source_ids[:space_for_source]
    input_ids = source_ids + target_ids

    # Labels: -100 dla promptu (nie uczymy na prompt-cie), target jako etykiety
    labels = [-100] * len(source_ids) + target_ids
    # Jeśli trzeba, dopełnij paddingiem
    attention_mask = [1] * len(input_ids)
    if len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + ([-100] * pad_len)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized = dataset.map(tokenize)

# Konfiguracja LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)

# Argumenty treningu
training_args = TrainingArguments(
    output_dir="./lora-deepseek-1.3b",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    remove_unused_columns=False,
    fp16=False
)

# Trener
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

# Trening
trainer.train()

# Zapis adaptera LoRA
model.save_pretrained("lora-deepseek-1.3b")
