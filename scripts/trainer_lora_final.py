# scripts/bank_statement/trainer_lora_final.py (Version Finale Simplifiée)

import json
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, dataloader
from transformers import (
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
    Trainer,
    TrainingArguments,
    DefaultDataCollator
)
from peft import LoraConfig, get_peft_model
import argparse
from sklearn.model_selection import train_test_split

MAX_LENGTH = 1024

class BankStatementDataset(Dataset):
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

    print(f"Chargement des données depuis {args.dataset_path}")
    with open(args.dataset_path, 'r') as f:
        train_dataset_data = json.load(f) # On utilise tout le dataset pour l'entraînement
    
    project_root = os.path.dirname(os.path.dirname(args.dataset_path))
    train_dataset = BankStatementDataset(train_dataset_data, processor, project_root)
    print(f"{len(train_dataset)} exemples d'entraînement.")

    # --- CONFIGURATION D'ENTRAÎNEMENT SIMPLIFIÉE ET ROBUSTE ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_strategy="epoch", # On sauvegarde à chaque époque
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn_filter_none,
    )
    trainer.train()

    print("Entraînement terminé. Sauvegarde du dernier modèle...")
    model.save_pretrained(args.output_dir)
    print(f"Dernier modèle sauvegardé dans {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuner un modèle Pix2Struct avec LoRA.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)