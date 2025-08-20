# src/create_datasets.py
import json
import random

INPUT_FILE = "data/questions_dataset.json"
TRAIN_FILE = "data/train_dataset.json"
TEST_FILE = "data/test_dataset.json"
TRAIN_SPLIT_RATIO = 0.8

print(f"Chargement du jeu de données complet depuis '{INPUT_FILE}'...")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    full_dataset = json.load(f)

print(f"Total d'exemples : {len(full_dataset)}")

# On mélange le jeu de données pour garantir une répartition aléatoire
random.shuffle(full_dataset)

# On calcule le point de division
split_index = int(len(full_dataset) * TRAIN_SPLIT_RATIO)

# On divise les données
train_data = full_dataset[:split_index]
test_data = full_dataset[split_index:]

print(f"Taille du jeu d'entraînement : {len(train_data)} exemples ({TRAIN_SPLIT_RATIO * 100}%)")
print(f"Taille du jeu de test : {len(test_data)} exemples")

# On sauvegarde les nouveaux fichiers
with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
print(f"✔ Jeu d'entraînement sauvegardé dans '{TRAIN_FILE}'")

with open(TEST_FILE, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)
print(f"✔ Jeu de test sauvegardé dans '{TEST_FILE}'")