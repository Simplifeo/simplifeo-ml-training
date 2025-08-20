# src/trainer_pix2struct.py

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
import argparse

# --- CONFIGURATION ---
# Chemins relatifs qui fonctionneront en local
DATASET_PATH = "data/train_dataset.json" # On utilise le jeu d'entraînement
BASE_MODEL_ID = "google/pix2struct-docvqa-base"
OUTPUT_MODEL_DIR = "simplifeo-lora-adapter-final" # Nom de notre adaptateur LoRA final

# --- 1. Préparation du Jeu de Données ---
class BankStatementDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, processor, data_root="."):
        self.processor = processor
        self.data_root = data_root # Le dossier racine où se trouvent les images
        
        print(f"Chargement du jeu de données depuis '{dataset_path}'...")
        self.dataset = json.load(open(dataset_path))
        print(f"Chargement de {len(self.dataset)} exemples terminé.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # On reconstruit le chemin complet de l'image
        image_path = os.path.join(self.data_root, os.path.basename(item['image_path']))
        
        try:
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
        except FileNotFoundError:
            print(f"\nERREUR : Image non trouvée au chemin '{image_path}'. Cet exemple sera ignoré.")
            return None
        except Exception as e:
            print(f"\nERREUR inattendue sur l'exemple #{idx} ({image_path}): {e}. Cet exemple sera ignoré.")
            return None

# --- 2. Fonction Principale d'Entraînement ---
def train(smoke_test=False):
    print("Début du processus de fine-tuning...")

    # Définir les chemins en fonction de l'environnement
    if smoke_test:
        # En local, on travaille depuis la racine du projet
        data_root = "data/synthetic_bank_statements"
        dataset_path = "data/train_dataset.json"
    else:
        # Sur Colab, les données sont dans /content/local_data
        data_root = "/content/local_data"
        dataset_path = os.path.join(data_root, os.path.basename(DATASET_PATH))

    processor = Pix2StructProcessor.from_pretrained(BASE_MODEL_ID)
    dataset = BankStatementDataset(dataset_path=dataset_path, processor=processor, data_root=data_root)

    if smoke_test:
        print("\n--- LANCEMENT DU SMOKE TEST LOCAL ---")
        print("Vérification de la préparation de 5 exemples...")
        if len(dataset) < 5:
            print(f"ERREUR: Pas assez de données ({len(dataset)} trouvées).")
            return
        
        valid_samples = 0
        i = 0
        while valid_samples < 5 and i < len(dataset):
            sample = dataset[i]
            if sample is not None:
                valid_samples += 1
            i += 1
            print(f"\rExemples valides traités : {valid_samples}", end="")

        print("\n\n✔ Smoke test terminé avec succès ! Le code de préparation des données est valide.")
        return

    # --- Le code ci-dessous ne s'exécute que sur Colab ---
    model = Pix2StructForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=50,
        save_strategy="epoch",
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        remove_unused_columns=False,
        fp16=True,
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

    final_adapter_dir = os.path.join(OUTPUT_MODEL_DIR, "final_adapter")
    trainer.save_model(final_adapter_dir)
    print(f"✔ Adaptateur LoRA final sauvegardé dans '{final_adapter_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Lance une vérification rapide sans entraînement'
    )
    args = parser.parse_args()

    train(smoke_test=args.smoke_test)