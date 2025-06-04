import torch  # Pour la manipulation des tenseurs et le support GPU

from transformers import (
    BitsAndBytesConfig,                # Pour la configuration de quantification 4-bit
    AutoModelForCausalLM,              # Pour charger un modèle de langage causale (auto-régressif)
    AutoTokenizer,                     # Pour charger le tokenizer associé au modèle
    TrainingArguments,                 # Pour configurer les paramètres d'entraînement
    Trainer,                           # Classe haute-niveau pour l'entraînement des modèles
    DataCollatorForLanguageModeling,  # Pour coller dynamiquement les données pendant l'entraînement
)

from peft import (
    LoraConfig,             # Configuration de LoRA (Low-Rank Adaptation)
    get_peft_model,         # Pour appliquer LoRA sur un modèle existant
    prepare_model_for_kbit_training,  # Prépare un modèle pour l'entraînement quantifié (k-bit)
)

from datasets import load_dataset  # Pour charger des datasets via Hugging Face

# ============================================================
# 1) Configuration de la quantisation 4 bits (NF4)
# ------------------------------------------------------------
# • load_in_4bit=True              : charge tous les poids du modèle en 4 bits (sauve mémoire GPU ×8).
# • bnb_4bit_compute_dtype=float16 : les produits matriciels sont faits en FP16 (meilleur compromis V‑RAM / précision).
# • bnb_4bit_use_double_quant=True : applique une quantisation hiérarchique (8 bits + 4 bits) pour plus de fidélité.
# • bnb_4bit_quant_type="nf4"      : NormalFloat4 → distribution plus proche d'un float16 qu'un int4 standard.
#   Résultat : un modèle ~13 Go → ~3‑4 Go, assez petit pour un seul GPU 24 Go.
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ============================================================
# 2) Chargement du modèle pré‑entraîné
# ------------------------------------------------------------
# • "mistralai/mistral-7b-instruct-v0.3" : checkpoint Instruct (SFT) de Mistral‑AI.
# • quantization_config=bnb_config       : charge directement en 4 bits.
# • device_map="auto"                   : répartit automatiquement les poids sur les GPU disponibles.
#   Après le chargement, on adapte le modèle pour un entraînement QLoRA.
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/mistral-7b-instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prépare pour k-bit training
model = prepare_model_for_kbit_training(model)

# ============================================================
# 3) Initialisation du tokenizer
# ------------------------------------------------------------
# • On réutilise exactement le même tokenizer que le modèle (fidèle aux tokens spéciaux).
# • trust_remote_code=True car Mistral fournit une implémentation custom des tokens spéciaux.
# • Si pad_token_id est absent (cas courant pour modèles de chat), on le masque sur eos_token.
#   ‑ Ça évite des erreurs de padding dans le DataCollator.
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/mistral-7b-instruct-v0.3",
    trust_remote_code=True
)
# Assure un pad_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ============================================================
# 4) Chargement et préparation du jeu de données
# ------------------------------------------------------------
# • Attendu : deux fichiers JSONL (train.jsonl, valid.jsonl) où chaque ligne contient :
#     {"messages": [ {"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}, … ]}
# • build_prompt()        : reconstruit le chat en texte plat avec marqueurs <|user|> / <|assistant|>.
# • preprocess()          : tokenise chaque conversation, tronque/pad à max_length, copie input_ids → labels.
#   ‑ labels masqués par DataCollatorForLanguageModeling (ignore_index) lors du MLM=False.
# ============================================================
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

# ============================================================
# 5) Configuration LoRA (PEFT)
# ------------------------------------------------------------
# • r=8, lora_alpha=16       : largeur du rang réduit et facteur d'échelle (klassique).
# • target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
#   ‑ On n'ajoute des matrices LoRA que sur les projections d'attention, suffisant pour bien apprendre.
# • lora_dropout=0.05        : régularisation légèrement supérieure pour petit dataset.
# • bias="none"             : on ne LoRA‑ifie pas les biais.
# • task_type="CAUSAL_LM"   : type de tâche pour peft.
# ============================================================
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# ============================================================
# 6) Hyper‑paramètres d'entraînement (TrainingArguments)
# ------------------------------------------------------------
# • per_device_train_batch_size=1 & gradient_accumulation_steps=4
#   ‑ Simule un batch effectif de 4 pour ne pas dépasser la VRAM.
# • fp16=True permet un forward mixte FP16/INT8/INT4.
# • logging_steps, save_steps et eval_steps synchronisés à 200 pour voir l'évolution en temps réel.
# • save_total_limit=3  : garde seulement les 3 derniers checkpoints (économie de disque).
# • load_best_model_at_end=True : recharge le meilleur checkpoint basé sur la métrique d'éval par défaut (loss).
# ============================================================
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

# ============================================================
# 7) Création du DataCollator et du Trainer
# ------------------------------------------------------------
# • DataCollatorForLanguageModeling avec mlm=False conserve la perte causal LM (Shift Left) et masque juste le padding.
# • Trainer orchestre : boucle d'entraînement, évaluation, checkpointing, logs.
#   Toutes les optimisations LoRA/4‑bit sont gérées sous le capot par Hugging Face + peft.
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Entraînement et sauvegarde du modèle
if __name__ == "__main__":
    trainer.train()
    trainer.save_model("mistral7b-qlora")