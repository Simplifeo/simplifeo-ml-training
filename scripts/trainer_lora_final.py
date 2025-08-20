import json
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, dataloader
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import argparse

# --- HYPERPARAMÈTRE CRUCIAL ---
# Définit la longueur maximale des séquences. Doit être assez grand pour contenir le JSON.
MAX_LENGTH = 1024

class BankStatementDataset(Dataset):
    def __init__(self, dataset_path, processor):
        self.dataset = json.load(open(dataset_path))
        self.processor = processor
        self.project_root = os.path.dirname(os.path.dirname(dataset_path))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        try:
            absolute_image_path = os.path.join(self.project_root, item['image_path'])
            image = Image.open(absolute_image_path)
        except FileNotFoundError:
            print(f"AVERTISSEMENT : Fichier image non trouvé à {absolute_image_path}. Cet exemple sera ignoré.")
            return None
        
        question = item['question']
        answer = item['answer']
        
        # --- CORRECTION DE LA TOKENISATION ---
        # On force la troncature et le padding à MAX_LENGTH pour les entrées ET les étiquettes.
        inputs = self.processor(
            images=image, 
            text=question, 
            return_tensors="pt",
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True
        )
        labels = self.processor(
            text=answer, 
            return_tensors="pt",
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True
        ).input_ids
        
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        inputs['labels'] = labels.squeeze()
        
        return inputs

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    return dataloader.default_collate(batch)

def train(args):
    print("Début du fine-tuning LoRA final...")
    model_name = "google/pix2struct-base"
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
    processor = Pix2StructProcessor.from_pretrained(model_name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = BankStatementDataset(dataset_path=args.dataset_path, processor=processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    trainer.train()

    print("Entraînement terminé. Sauvegarde du modèle...")
    model.save_pretrained(args.output_dir)
    print(f"Modèle sauvegardé dans {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuner un modèle Pix2Struct avec LoRA.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Chemin vers le fichier dataset JSON.")
    parser.add_argument("--output_dir", type=str, required=True, help="Dossier où sauvegarder l'adaptateur LoRA.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Nombre d'époques d'entraînement.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Taille du batch par appareil.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Taux d'apprentissage.")
    
    args = parser.parse_args()
    train(args)