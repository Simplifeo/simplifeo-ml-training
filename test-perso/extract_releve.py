import torch
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

# --- 1. Chargement du Modèle et du Processeur ---
# On utilise le modèle de base de Pix2Struct pour le DocVQA.
# Il est un bon compromis entre performance et ressources nécessaires.
model_name = "google/pix2struct-docvqa-base"
model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
processor = Pix2StructProcessor.from_pretrained(model_name)

# Déplacer le modèle sur le GPU si disponible (fortement recommandé)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Modèle chargé sur le périphérique : {device}")

# --- 2. Chargement de l'image ---
try:
    image = Image.open("releve.png")
    print("Image 'releve.png' chargée avec succès.")
except FileNotFoundError:
    print("Erreur : Le fichier 'releve.png' n'a pas été trouvé. Assurez-vous qu'il est dans le même dossier que le script.")
    exit()

# --- 3. Création du Prompt d'Extraction ---
# C'est la partie la plus importante. On donne une instruction claire au modèle
# pour qu'il extraie toutes les informations dans un format JSON structuré.
prompt = """Extrais les informations suivantes de ce relevé de compte. Réponds uniquement avec le JSON.
Les champs à extraire sont :
- "banque": Le nom de la banque.
- "titulaire_compte": Le nom complet du titulaire du compte.
- "iban": L'IBAN du compte.
- "numero_compte": Le numéro de compte.
- "periode_debut": La date de début du relevé.
- "periode_fin": La date de fin du relevé.
- "solde_precedent": Le montant du solde précédent.
- "nouveau_solde": Le montant du nouveau solde.
- "transactions": Une liste d'objets, où chaque objet contient "date", "libelle", "debit" et "credit". Pour les débits ou crédits absents, utilise la valeur null."""

print("\n--- Prompt envoyé au modèle ---")
print(prompt)

# --- 4. Traitement et Génération ---
# is_document=True est important pour optimiser le traitement pour les documents.
inputs = processor(images=image, text=prompt, return_tensors="pt", is_document=True).to(device)

# Génération de la réponse
predictions = model.generate(**inputs, max_new_tokens=2048)

# Décodage de la réponse pour la rendre lisible
decoded_prediction = processor.decode(predictions[0], skip_special_tokens=True)

# --- 5. Affichage du Résultat ---
print("\n--- Réponse JSON extraite par Pix2Struct ---")
print(decoded_prediction)

# Optionnel : Essayer de convertir la chaîne JSON en dictionnaire Python
import json

try:
    data = json.loads(decoded_prediction)
    print("\n--- Le JSON est valide et a été converti en dictionnaire Python ---")
    # print(data) # Décommentez pour voir le dictionnaire
except json.JSONDecodeError:
    print("\n--- Avertissement : La sortie du modèle n'est pas un JSON parfaitement valide. ---")