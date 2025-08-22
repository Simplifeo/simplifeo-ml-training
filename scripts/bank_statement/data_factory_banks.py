# src/data_factory_banks.py

import os
import csv
import random
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---
TOTAL_STATEMENTS = 100
OUTPUT_DIR = "data/synthetic_bank_statements"
DATA_DIR = "data"

fake = Faker('fr_FR')

# ... La fonction generate_statement_data() reste inchangée ...
def generate_statement_data():
    random_date_in_month = fake.date_object()
    start_date = random_date_in_month.replace(day=1)
    end_date = random_date_in_month.replace(day=28)
    account_holder = fake.name()
    address_street = fake.street_address()
    address_postcode = fake.postcode()
    address_city = fake.city()
    full_address_display = f"{address_street}, {address_postcode} {address_city}"
    bank_name = random.choice(["BNP Paribas", "Société Générale", "Crédit Agricole", "La Banque Postale", "Crédit Mutuel"])
    iban = fake.iban()
    account_number = fake.numerify(text='N° %%%%%%%%%%%')
    start_balance = round(random.uniform(500.0, 5000.0), 2)
    transactions = []
    current_balance = start_balance
    for _ in range(random.randint(5, 15)):
        is_debit = random.choice([True, True, False])
        amount = round(random.uniform(10.0, 450.0), 2)
        if is_debit:
            description = random.choice(["PAIEMENT CB", "PRELEVEMENT", "RETRAIT DAB", "VIREMENT A"]) + " " + fake.company().upper()
            debit, credit = f"{amount:.2f}", ""
            current_balance -= amount
        else:
            description = random.choice(["VIREMENT DE", "REMISE DE CHEQUE"]) + " " + fake.company().upper()
            debit, credit = "", f"{amount:.2f}"
            current_balance += amount
        transactions.append({"date": fake.date_between(start_date=start_date, end_date=end_date).strftime('%d/%m/%Y'), "description": description, "debit": debit, "credit": credit})
    end_balance = round(current_balance, 2)
    return {
        "account_holder": account_holder, "full_address_display": full_address_display,
        "address_street": address_street, "address_postcode": address_postcode, "address_city": address_city,
        "bank_name": bank_name, "iban": iban, "account_number": account_number,
        "period": f"du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}",
        "start_balance": f"{start_balance:.2f}", "end_balance": f"{end_balance:.2f}",
        "transactions": transactions
    }

# --- FONCTION DE CRÉATION D'IMAGE ENTIÈREMENT REVUE ---
def create_dynamic_statement_image(data, output_path):
    """Crée une image PNG avec une mise en page et des libellés très variés."""
    width, height = 800, 1100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    
    try:
        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
        font_regular = ImageFont.truetype(font_path, random.randint(13, 15))
        font_bold = ImageFont.truetype(font_path.replace("Arial", "Arial Bold"), random.randint(15, 17))
        font_title = ImageFont.truetype(font_path.replace("Arial", "Arial Bold"), random.randint(22, 26))
    except IOError:
        font_regular, font_bold, font_title = ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default()

    margin = 40
    y_pos = margin + random.randint(0, 20)
    
    # --- 1. Titre et Nom de la Banque (positions variables) ---
    title_text = random.choice(["RELEVÉ DE COMPTE", "RELEVE BANCAIRE", "SITUATION DE COMPTE"])
    draw.text((random.randint(margin, width // 2), y_pos), title_text, fill="black", font=font_title)
    y_pos += random.randint(40, 60)
    draw.text((random.randint(margin, width // 2), y_pos), data['bank_name'], fill="black", font=font_bold)
    
    # --- 2. Création de 5 blocs d'information indépendants ---
    
    # Variation des libellés
    holder_label = random.choice(["Titulaire :", "Client :", "Nom :", ""])
    account_label = random.choice(["N° de Compte :", "Compte :", "Compte N° :", ""])
    iban_label = random.choice(["IBAN :", "IBAN N° :", ""])
    period_label = random.choice(["Période :", "Du", ""])
    
    # On ajoute le libellé à la valeur, ou pas si le libellé est vide
    holder_text = f"{holder_label} {data['account_holder']}" if holder_label else data['account_holder']
    account_text = f"{account_label} {data['account_number']}" if account_label else data['account_number']
    iban_text = f"{iban_label} {data['iban']}" if iban_label else data['iban']
    period_text = f"{period_label} {data['period']}" if period_label else data['period']
    
    info_blocks = [
        holder_text,
        data['full_address_display'], # L'adresse n'a généralement pas de libellé
        account_text,
        iban_text,
        period_text
    ]
    
    # On mélange l'ordre des blocs pour une variété maximale
    random.shuffle(info_blocks)
    
    # --- 3. Placement aléatoire des blocs d'information ---
    
    # On définit des "zones" de placement possibles sur la page
    available_zones = [
        (margin, 150), (width // 2, 150), # Ligne 1
        (margin, 250), (width // 2, 250), # Ligne 2
        (margin, 350)                     # Ligne 3
    ]
    random.shuffle(available_zones)
    
    max_y_after_blocks = 0
    for i, block_text in enumerate(info_blocks):
        x, y = available_zones[i]
        draw.text((x, y), block_text, fill="black", font=font_regular)
        text_height = 20 # Approximation
        max_y_after_blocks = max(max_y_after_blocks, y + text_height)

    y_pos = max_y_after_blocks + random.randint(40, 80)
    
    # --- 4. Tableau des transactions (inchangé) ---
    draw.line([(margin, y_pos), (width - margin, y_pos)], fill="black", width=2)
    y_pos += 10
    draw.text((margin + 10, y_pos), "Date", fill="black", font=font_bold)
    draw.text((margin + 110, y_pos), "Libellé de l'opération", fill="black", font=font_bold)
    draw.text((width - margin - 250, y_pos), "Débit (€)", fill="black", font=font_bold)
    draw.text((width - margin - 120, y_pos), "Crédit (€)", fill="black", font=font_bold)
    y_pos += 10
    draw.line([(margin, y_pos), (width - margin, y_pos)], fill="black", width=2)
    y_pos += 15

    for trans in data['transactions']:
        draw.text((margin + 10, y_pos), trans['date'], fill="black", font=font_regular)
        draw.text((margin + 110, y_pos), trans['description'], fill="black", font=font_regular)
        draw.text((width - margin - 250, y_pos), trans['debit'], fill="black", font=font_regular)
        draw.text((width - margin - 120, y_pos), trans['credit'], fill="black", font=font_regular)
        y_pos += random.randint(22, 28)

    # --- 5. Bloc des soldes (position aléatoire) ---
    y_pos += random.randint(30, 60)
    balance_x = random.choice([margin, width - margin - 300])
    draw.text((balance_x, y_pos), f"Solde précédent : {data['start_balance']} €", fill="black", font=font_regular)
    y_pos += 25
    draw.text((balance_x, y_pos), f"Nouveau solde : {data['end_balance']} €", fill="black", font=font_bold)

    image.save(output_path)

# ... La fonction main() reste inchangée, elle appelle maintenant la nouvelle fonction de dessin ...
def main():
    print(f"Début de la génération de {TOTAL_STATEMENTS} relevés bancaires synthétiques...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_file_path = os.path.join(DATA_DIR, "ground_truth.csv")
    csv_headers = [
        "file_name", "account_holder", "address_street", "address_postcode", "address_city",
        "bank_name", "account_number", "iban", "period", "start_balance", "end_balance", "transactions_json"
    ]
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        for i in range(TOTAL_STATEMENTS):
            statement_data = generate_statement_data()
            file_name = f"statement_{i:04d}.png"
            output_path = os.path.join(OUTPUT_DIR, file_name)
            create_dynamic_statement_image(statement_data, output_path) # Appel de la nouvelle fonction
            row_to_write = {
                "file_name": file_name, "account_holder": statement_data['account_holder'],
                "address_street": statement_data['address_street'], "address_postcode": statement_data['address_postcode'],
                "address_city": statement_data['address_city'], "bank_name": statement_data['bank_name'],
                "account_number": statement_data['account_number'], "iban": statement_data['iban'],
                "period": statement_data['period'], "start_balance": statement_data['start_balance'],
                "end_balance": statement_data['end_balance'], "transactions_json": str(statement_data['transactions'])
            }
            writer.writerow(row_to_write)
            print(f"  ({i + 1}/{TOTAL_STATEMENTS}) Image {file_name} générée.", end='\r')
    print(f"\n\nTerminé ! {TOTAL_STATEMENTS} images et leur ground_truth.csv sont prêts.")

if __name__ == "__main__":
    main()