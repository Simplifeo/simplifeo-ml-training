import pandas as pd
import json
import os
from tqdm import tqdm
import ast
from datetime import datetime

# --- CONFIGURATION ---
GROUND_TRUTH_PATH = 'data/ground_truth.csv' 
IMAGES_BASE_PATH = 'data/synthetic_bank_statements/'
OUTPUT_DATASET_PATH = 'data/dataset_json.json'

# --- MISE À JOUR DU SCHÉMA JSON ---
# On ne l'inclut plus dans le prompt, mais on le garde ici pour référence
JSON_SCHEMA = {
    "banque": "string",
    "numero_compte": "string", # Nouveau champ
    "iban": "string",
    "titulaire": "string",
    "adresse": { # Nouveau champ structuré
        "rue": "string",
        "code_postal": "string",
        "ville": "string"
    },
    "periode": { "debut": "YYYY-MM-DD", "fin": "YYYY-MM-DD" },
    "solde_initial": "float",
    "solde_final": "float",
    "transactions": [
      { "date": "YYYY-MM-DD", "description": "string", "debit": "float", "credit": "float" }
    ]
}

INSTRUCTION_PROMPT = "Extrais les informations de ce document et retourne-les au format JSON."

# --- SCRIPT ---

def format_date(date_str: str) -> str:
    if not date_str or pd.isna(date_str): return None
    try: return datetime.strptime(date_str, '%d/%m/%Y').strftime('%Y-%m-%d')
    except (ValueError, TypeError): return None

def parse_period(period_str: str) -> tuple[str, str]:
    if not period_str or pd.isna(period_str): return None, None
    try:
        parts = period_str.replace('du ', '').split(' au ')
        return format_date(parts[0]), format_date(parts[1])
    except Exception: return None, None

def create_json_object(row: pd.Series) -> dict:
    """Construit l'objet JSON complet pour une ligne du DataFrame."""
    start_date, end_date = parse_period(row.get('period'))

    transactions_list = []
    try:
        transactions_raw = ast.literal_eval(row.get('transactions_json', '[]'))
        for trans in transactions_raw:
            transactions_list.append({
                "date": format_date(trans.get('date')),
                "description": trans.get('description'),
                "debit": float(trans['debit']) if trans.get('debit') else None,
                "credit": float(trans['credit']) if trans.get('credit') else None
            })
    except (ValueError, SyntaxError): pass

    # --- MISE À JOUR DE LA CRÉATION DU JSON ---
    json_output = {
        "banque": row.get('bank_name'),
        "numero_compte": row.get('account_number'), # Ajout du numéro de compte
        "iban": row.get('iban'),
        "titulaire": row.get('account_holder'),
        "adresse": { # Ajout de l'adresse structurée
            "rue": row.get('address_street'),
            "code_postal": row.get('address_postcode'),
            "ville": row.get('address_city')
        },
        "periode": { "debut": start_date, "fin": end_date },
        "solde_initial": float(row.get('start_balance')) if pd.notna(row.get('start_balance')) else None,
        "solde_final": float(row.get('end_balance')) if pd.notna(row.get('end_balance')) else None,
        "transactions": transactions_list
    }
    return json_output

def generate_json_dataset():
    """Fonction principale qui lit les données, les transforme et sauvegarde le nouveau jeu de données."""
    print(f"Chargement des données depuis : {GROUND_TRUTH_PATH}")
    try:
        df = pd.read_csv(GROUND_TRUTH_PATH).fillna(value=pd.NA)
    except FileNotFoundError:
        print(f"ERREUR : Le fichier {GROUND_TRUTH_PATH} n'a pas été trouvé.")
        return

    print(f"{len(df)} enregistrements trouvés.")
    new_dataset = []
    
    print("Génération du nouveau jeu de données au format (Instruction, JSON_Complet)...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        json_obj = create_json_object(row)
        json_string = json.dumps(json_obj, ensure_ascii=False, separators=(',', ':'))
        image_path = os.path.join(IMAGES_BASE_PATH, row['file_name'])
        
        new_dataset.append({
            "image_path": image_path,
            "question": INSTRUCTION_PROMPT,
            "answer": json_string
        })
        
    print(f"Sauvegarde du jeu de données final dans : {OUTPUT_DATASET_PATH}")
    with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, indent=4, ensure_ascii=False)
        
    print("Opération terminée avec succès !")
    print(f"{len(new_dataset)} exemples ont été générés.")

if __name__ == '__main__':
    generate_json_dataset()