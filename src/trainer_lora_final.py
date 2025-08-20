# src/trainer_lora_final.py

import os
import json
import torch
from PIL import Image
from transformers import (
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from peft import LoraConfig, get_peft_model

# --- CONFIGURATION ---
LOCAL_DATA_ROOT = "/content/local_data"
DATASET_PATH = f"{LOCAL_DATA_ROOT}/train_dataset.json" # On s'entraîne sur le jeu d'entraînement
BASE_MODEL_ID = "google/pix2struct-docvqa-base"
OUTPUT_MODEL_DIR = "simplifeo-lora-adapter-v1" # Nom de notre adaptateur LoRA

# --- 1. Préparation du Jeu de Données ---
class BankStatementDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, processor):
        self.processor = processor
        self.dataset = json.load(open(dataset_path))
        print(f"Chargement de {len(self.dataset)} exemples depuis '{dataset_path}'.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = os.path.join(LOCAL_DATA_ROOT, os.path.basename(item['image_path']))
        
        image = Image.open(image_path)
        question = item['question']
        answer = item['answer']

        # Formatage simple et direct
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        # On utilise le tokenizer pour les labels
        labels = self.processor.tokenizer(
            text=answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).input_ids

        inputs['labels'] = labels
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        
        return inputs

# --- 2. Fonction Principale d'Entraînement ---
def train():
    print("Début du fine-tuning LoRA final...")

    processor = Pix2StructProcessor.from_pretrained(BASE_MODEL_ID)
    dataset = BankStatementDataset(dataset_path=DATASET_PATH, processor=processor)

    # On charge le modèle de base, le Trainer s'occupera de la conversion fp16
    model = Pix2StructForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
    )
    
    # Configuration LoRA stable
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Arguments d'entraînement optimisés et stables
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=50,
        save_strategy="epoch", # On sauvegarde à chaque époque
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        remove_unused_columns=False,
        fp16=True, # Indispensable pour l'entraînement sur GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DefaultDataCollator()
    )

    print("\n--- Lancement de l'entraînement LoRA ---")
    trainer.train()
    print("--- Entraînement terminé ---")

    # On sauvegarde l'adaptateur LoRA final
    final_adapter_dir = os.path.join(OUTPUT_MODEL_DIR, "final_adapter")
    trainer.save_model(final_adapter_dir)
    print(f"✔ Adaptateur LoRA final sauvegardé dans '{final_adapter_dir}'")


if __name__ == "__main__":
    train()