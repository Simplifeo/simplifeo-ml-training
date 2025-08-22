# scripts/bank_statement/annotator_json.py (Version Finale Corrigée)

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
INSTRUCTION_PROMPT = "Extrais les informations de ce document et retourne-les au format JSON."

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
    """Construit l'objet JSON complet en s'assurant que les valeurs numériques sont des chaînes."""
    start_date, end_date = parse_period(row.get('period'))
    transactions_list = []
    try:
        transactions_raw = ast.literal_eval(row.get('transactions_json', '[]'))
        for trans in transactions_raw:
            transactions_list.append({
                "date": format_date(trans.get('date')),
                "description": trans.get('description'),
                # --- CORRECTION : Forcer les nombres en chaînes ---
                "debit": str(trans['debit']) if trans.get('debit') and str(trans['debit']).strip() else None,
                "credit": str(trans['credit']) if trans.get('credit') and str(trans['credit']).strip() else None
            })
    except (ValueError, SyntaxError): pass

    json_output = {
        "banque": row.get('bank_name'),
        "numero_compte": row.get('account_number'),
        "iban": row.get('iban'),
        "titulaire": row.get('account_holder'),
        "adresse": {
            "rue": row.get('address_street'),
            "code_postal": str(row.get('address_postcode')) if pd.notna(row.get('address_postcode')) else None,
            "ville": row.get('address_city')
        },
        "periode": { "debut": start_date, "fin": end_date },
        # --- CORRECTION : Forcer les nombres en chaînes ---
        "solde_initial": str(row.get('start_balance')) if pd.notna(row.get('start_balance')) else None,
        "solde_final": str(row.get('end_balance')) if pd.notna(row.get('end_balance')) else None,
        "transactions": transactions_list
    }
    return json_output

def generate_json_dataset():
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
        new_dataset.append({"image_path": image_path, "question": INSTRUCTION_PROMPT, "answer": json_string})
    
    print(f"Sauvegarde du jeu de données final dans : {OUTPUT_DATASET_PATH}")
    with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, indent=4, ensure_ascii=False)
    print("Opération terminée avec succès !")

if __name__ == '__main__':
    generate_json_dataset()