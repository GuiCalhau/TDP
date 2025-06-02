import torch
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# 1) Config quantization 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# 2) Chargement du modèle
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/mistral-7b-instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prépare pour k-bit training
model = prepare_model_for_kbit_training(model)

# 3) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/mistral-7b-instruct-v0.3",
    trust_remote_code=True
)
# Assure un pad_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 4) Dataset
raw_datasets = load_dataset(
    "json",
    data_files={"train": "train.jsonl", "validation": "valid.jsonl"},
)

def build_prompt(messages):
    """
    messages: liste de dicts, p. ex. [ {"role":"user","content":"…"}, {"role":"assistant","content":"…"} ]
    """
    chat = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        tag = "<|user|>:" if role == "user" else "<|assistant|>:"
        chat += f"{tag} {content}\n"
    return chat

def preprocess(examples):
    # examples["messages"] est une liste de conversations
    prompts = [ build_prompt(conv) for conv in examples["messages"] ]
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_datasets = raw_datasets.map(
    preprocess,
    remove_columns=["messages"],
    batched=True
)

# 5) Config LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# 6) Arguments d’entraînement
training_args = TrainingArguments(
    output_dir="mistral7b-qlora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    eval_strategy="steps",    
    eval_steps=200,          
    do_eval=True,             
    save_total_limit=3,
    load_best_model_at_end=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 7) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("mistral7b-qlora")