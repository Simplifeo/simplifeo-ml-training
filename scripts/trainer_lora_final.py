# src/trainer_lora_final.py

import json
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, dataloader
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor, Trainer, TrainingArguments, DefaultDataCollator
from peft import LoraConfig, get_peft_model
import argparse
from sklearn.model_selection import train_test_split # <-- Nouvel import

MAX_LENGTH = 1024

class BankStatementDataset(Dataset):
    # ... (inchangé)
    def __init__(self, data, processor, project_root):
        self.dataset = data
        self.processor = processor
        self.project_root = project_root
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        try:
            absolute_image_path = os.path.join(self.project_root, item['image_path'])
            image = Image.open(absolute_image_path)
        except FileNotFoundError:
            return None
        question = item['question']
        answer = item['answer']
        inputs = self.processor(images=image, text=question, return_tensors="pt", max_length=MAX_LENGTH, padding="max_length", truncation=True)
        labels = self.processor.tokenizer(text=answer, return_tensors="pt", max_length=MAX_LENGTH, padding="max_length", truncation=True).input_ids
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        inputs['labels'] = labels.squeeze()
        return inputs

def collate_fn_filter_none(batch):
    # ... (inchangé)
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    return DefaultDataCollator()(batch)

def train(args):
    print("Début du fine-tuning LoRA final...")
    model_name = "google/pix2struct-docvqa-base"
    print(f"Chargement du modèle spécialisé : {model_name}")
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
    processor = Pix2StructProcessor.from_pretrained(model_name)

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "key", "value"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- NOUVELLE LOGIQUE DE GESTION DES DONNÉES ---
    print(f"Chargement et division des données depuis {args.dataset_path}")
    with open(args.dataset_path, 'r') as f:
        full_dataset_data = json.load(f)
    
    # On divise en 90% pour l'entraînement+validation et 10% pour le test final (non utilisé ici)
    train_val_data, _ = train_test_split(full_dataset_data, test_size=0.1, random_state=42)
    # On divise le reste en 90% pour l'entraînement et 10% pour la validation
    train_data, eval_data = train_test_split(train_val_data, test_size=0.1, random_state=42)
    
    project_root = os.path.dirname(os.path.dirname(args.dataset_path))
    train_dataset = BankStatementDataset(train_data, processor, project_root)
    eval_dataset = BankStatementDataset(eval_data, processor, project_root)
    print(f"{len(train_dataset)} exemples d'entraînement, {len(eval_dataset)} exemples de validation.")

    # --- MISE À JOUR DES ARGUMENTS D'ENTRAÎNEMENT ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch", # <-- On évalue à chaque époque
        load_best_model_at_end=True, # <-- On recharge le meilleur modèle à la fin
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # <-- On fournit le jeu de validation
        data_collator=collate_fn_filter_none,
    )
    trainer.train()

    print("Entraînement terminé. Sauvegarde du meilleur modèle...")
    model.save_pretrained(args.output_dir)
    print(f"Meilleur modèle sauvegardé dans {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuner un modèle Pix2Struct avec LoRA.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=15) # On met plus d'époques, l'arrêt précoce choisira
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)