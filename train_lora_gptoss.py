from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Model 20B (open-source) – nazwa odpowiadająca HF
# Uwaga: "gpt-oss:20b" to konwencja z ekosystemu uruchamiania (np. Ollama).
# Do trenowania LoRA w Transformers używamy ID z Hugging Face:
MODEL_NAME = "EleutherAI/gpt-neox-20b"

# Urządzenie (spróbuje CUDA → MPS → CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

# Dane
DATA_FILE = "dane_prompt.jsonl"
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# Tokenizer i model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Dla dużych modeli użyj device_map="auto" (offload wg zasobów)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True
)

# Tokenizacja z rezerwacją miejsca na target i maskowaniem promptu
def tokenize(example):
    source_text = f"{example['prompt'].rstrip()}\n\nPrzyczyna:"
    target_word = example["completion"].strip()
    target_text = f" {target_word}{tokenizer.eos_token}"

    source_ids = tokenizer(source_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

    max_length = 512
    space_for_source = max_length - len(target_ids)
    if space_for_source < 0:
        target_ids = target_ids[:max_length]
        source_ids = []
    else:
        source_ids = source_ids[:space_for_source]

    input_ids = source_ids + target_ids
    labels = [-100] * len(source_ids) + target_ids
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

# LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# Argumenty treningu – dla 20B ustaw bardziej zachowawcze parametry
training_args = TrainingArguments(
    output_dir="./lora-gptoss-20b",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    remove_unused_columns=False,
    fp16=(dtype == torch.float16),
    gradient_checkpointing=True
)

# Trener
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

# Trening
trainer.train()

# Zapis adaptera
model.save_pretrained("lora-gptoss-20b")


