import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import argparse # <-- Ajout important

# --- Classe Dataset (inchangée) ---
class BankStatementDataset(Dataset):
    def __init__(self, dataset_path, processor):
        self.dataset = json.load(open(dataset_path))
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item['image_path'])
        question = item['question']
        answer = item['answer']
        
        # Le processeur gère la tokenisation pour le modèle
        inputs = self.processor(images=image, text=question, return_tensors="pt")
        labels = self.processor(text=answer, return_tensors="pt").input_ids
        
        # Aplatir les tenseurs pour qu'ils soient au bon format
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        inputs['labels'] = labels.squeeze()
        
        return inputs

# --- Fonction d'entraînement (maintenant avec des arguments) ---
def train(args):
    print("Début du fine-tuning LoRA final...")

    # Charger le modèle et le processeur
    model_name = "google/pix2struct-base"
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
    processor = Pix2StructProcessor.from_pretrained(model_name)

    # Configurer LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "k", "v"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Charger le jeu de données en utilisant le chemin fourni en argument
    dataset = BankStatementDataset(dataset_path=args.dataset_path, processor=processor)

    # Définir les arguments d'entraînement
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
        fp16=True, # Utiliser la précision mixte pour accélérer l'entraînement sur les GPU compatibles
    )

    # Créer et lancer l'entraîneur
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    print("Entraînement terminé. Sauvegarde du modèle...")
    model.save_pretrained(args.output_dir)
    print(f"Modèle sauvegardé dans {args.output_dir}")

# --- Point d'entrée du script ---
if __name__ == '__main__':
    # Créer le parser pour les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Fine-tuner un modèle Pix2Struct avec LoRA.")
    
    # Définir les arguments attendus
    parser.add_argument("--dataset_path", type=str, required=True, help="Chemin vers le fichier dataset JSON.")
    parser.add_argument("--output_dir", type=str, required=True, help="Dossier où sauvegarder l'adaptateur LoRA.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Nombre d'époques d'entraînement.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Taille du batch par appareil.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Taux d'apprentissage.")
    
    # Parser les arguments fournis
    args = parser.parse_args()
    
    # Lancer la fonction d'entraînement avec les arguments
    train(args)