import pandas as pd
import json
import os
from tqdm import tqdm
import ast  # Pour parser les chaînes qui ressemblent à des littéraux Python (comme le JSON avec des apostrophes)
from datetime import datetime

# --- CONFIGURATION ---
GROUND_TRUTH_PATH = 'data/ground_truth.csv' 
IMAGES_BASE_PATH = 'data/synthetic_bank_statements/'
OUTPUT_DATASET_PATH = 'data/dataset_json.json'

JSON_SCHEMA = {
    "banque": "string",
    "iban": "string",
    "titulaire": "string",
    "periode": { "debut": "YYYY-MM-DD", "fin": "YYYY-MM-DD" },
    "solde_initial": "float",
    "solde_final": "float",
    "transactions": [
      { "date": "YYYY-MM-DD", "description": "string", "debit": "float", "credit": "float" }
    ]
}

INSTRUCTION_PROMPT = f"Extrais les informations de ce document et retourne-les au format JSON en respectant scrupuleusement le schéma suivant: {json.dumps(JSON_SCHEMA)}"

# --- SCRIPT FINAL ---

def format_date(date_str: str) -> str:
    """Convertit une date du format DD/MM/YYYY au format YYYY-MM-DD."""
    if not date_str or pd.isna(date_str):
        return None
    try:
        return datetime.strptime(date_str, '%d/%m/%Y').strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None

def parse_period(period_str: str) -> tuple[str, str]:
    """Extrait les dates de début et de fin de la chaîne 'du DD/MM/YYYY au DD/MM/YYYY'."""
    if not period_str or pd.isna(period_str):
        return None, None
    try:
        parts = period_str.replace('du ', '').split(' au ')
        start_date_str = parts[0]
        end_date_str = parts[1]
        return format_date(start_date_str), format_date(end_date_str)
    except Exception:
        return None, None

def create_json_object(row: pd.Series) -> dict:
    """
    Construit l'objet JSON complet pour une ligne du DataFrame en se basant sur la structure du CSV fourni.
    """
    # Parse la période
    start_date, end_date = parse_period(row.get('period'))

    # Parse les transactions depuis la chaîne JSON
    transactions_list = []
    try:
        # ast.literal_eval est plus sûr que eval() et gère les apostrophes
        transactions_raw = ast.literal_eval(row.get('transactions_json', '[]'))
        for trans in transactions_raw:
            transactions_list.append({
                "date": format_date(trans.get('date')),
                "description": trans.get('description'),
                "debit": float(trans['debit']) if trans.get('debit') else None,
                "credit": float(trans['credit']) if trans.get('credit') else None
            })
    except (ValueError, SyntaxError):
        # En cas d'erreur de parsing, on laisse la liste des transactions vide
        pass

    # Construit l'objet final
    json_output = {
        "banque": row.get('bank_name'),
        "iban": row.get('iban'),
        "titulaire": row.get('account_holder'),
        "periode": {
            "debut": start_date,
            "fin": end_date
        },
        "solde_initial": float(row.get('start_balance')) if pd.notna(row.get('start_balance')) else None,
        "solde_final": float(row.get('end_balance')) if pd.notna(row.get('end_balance')) else None,
        "transactions": transactions_list
    }
    return json_output

def generate_json_dataset():
    """
    Fonction principale qui lit les données, les transforme et sauvegarde le nouveau jeu de données.
    """
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
        
        # Utilise le bon nom de colonne 'file_name'
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