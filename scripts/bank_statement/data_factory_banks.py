# scripts/bank_statement/data_factory_banks.py (Version "Monde Réel")

import os
import csv
import random
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

# --- CONFIGURATION ---
TOTAL_STATEMENTS = 1000
OUTPUT_DIR = "data/synthetic_bank_statements"
DATA_DIR = "data"

fake = Faker('fr_FR')

def generate_statement_data():
    """Génère les données textuelles pour un relevé bancaire complet."""
    
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
            
        transactions.append({
            "date": fake.date_between(start_date=start_date, end_date=end_date).strftime('%d/%m/%Y'),
            "description": description, "debit": debit, "credit": credit
        })
        
    end_balance = round(current_balance, 2)
    
    return {
        "account_holder": account_holder, "full_address_display": full_address_display,
        "address_street": address_street, "address_postcode": str(address_postcode),
        "address_city": address_city, "bank_name": bank_name, "iban": iban,
        "account_number": account_number,
        "period": f"du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}",
        "start_balance": f"{start_balance:.2f}", "end_balance": f"{end_balance:.2f}",
        "transactions": transactions
    }

def create_hyper_realistic_image(data, output_path):
    """Crée une image avec une mise en page, des polices, et des formats très variés."""
    
    is_landscape = random.random() < 0.1 # 10% chance d'être en paysage
    width, height = (1100, 800) if is_landscape else (800, 1100)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    
    font_families = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Times New Roman.ttf", "/System/Library/Fonts/Supplemental/Verdana.ttf"]
    base_font_path = random.choice(font_families)
    try:
        font_regular = ImageFont.truetype(base_font_path, random.randint(12, 14))
        font_bold = ImageFont.truetype(base_font_path.replace(".ttf", " Bold.ttf"), random.randint(14, 16))
        font_title = ImageFont.truetype(base_font_path.replace(".ttf", " Bold.ttf"), random.randint(20, 24))
    except IOError:
        font_regular, font_bold, font_title = ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default()

    margin = 40
    y_pos = margin + random.randint(0, 20)
    
    title_text = random.choice(["RELEVÉ DE COMPTE", "RELEVE BANCAIRE", "SITUATION DE COMPTE"])
    if random.choice([True, False]):
        draw.text((margin, y_pos), title_text, fill="black", font=font_title)
    else:
        text_width = draw.textlength(title_text, font=font_title)
        draw.text(((width - text_width) / 2, y_pos), title_text, fill="black", font=font_title)
    y_pos += random.randint(50, 70)

    holder_label = random.choice(["Titulaire :", "Client :", "Nom :", ""])
    account_label = random.choice(["N° de Compte :", "Compte :", "Compte N° :", ""])
    iban_label = random.choice(["IBAN :", "IBAN N° :", ""])
    period_label = random.choice(["Période :", "Du", ""])
    
    holder_text = f"{holder_label} {data['account_holder']}" if holder_label else data['account_holder']
    account_text = f"{account_label} {data['account_number']}" if account_label else data['account_number']
    iban_text = f"{iban_label} {data['iban']}" if iban_label else data['iban']
    period_text = f"{period_label} {data['period']}" if period_label else data['period']
    
    info_blocks = [holder_text, data['full_address_display'], account_text, iban_text, period_text]
    random.shuffle(info_blocks)
    
    available_zones = [(margin, 150), (width // 2, 150), (margin, 250), (width // 2, 250), (margin, 350)]
    random.shuffle(available_zones)
    
    max_y_after_blocks = 0
    for i, block_text in enumerate(info_blocks):
        x, y = available_zones[i]
        draw.text((x, y), block_text, fill="black", font=font_regular)
        max_y_after_blocks = max(max_y_after_blocks, y + 20)

    y_pos = max_y_after_blocks + random.randint(40, 80)
    
    layout_type = random.choice(['2_colonnes', '1_colonne_signee', 'grille', 'decale'])
    decimal_sep = random.choice([',', '.'])
    date_format_out = random.choice(['%d/%m/%Y', '%Y-%m-%d'])

    header_y = y_pos
    draw.text((margin, header_y), "Date", font=font_bold, fill="black")
    draw.text((margin + 100, header_y), "Libellé", font=font_bold, fill="black")
    if layout_type in ['2_colonnes', 'grille', 'decale']:
        draw.text((width - margin - 250, header_y), "Débit", font=font_bold, fill="black")
        draw.text((width - margin - 120, header_y), "Crédit", font=font_bold, fill="black")
    else: # 1_colonne_signee
        draw.text((width - margin - 150, header_y), "Montant", font=font_bold, fill="black")

    y_pos += 30
    if layout_type == 'grille':
        draw.line([(margin, y_pos - 20), (width - margin, y_pos - 20)], fill="gray", width=1)

    for trans in data['transactions']:
        trans_date = datetime.strptime(trans['date'], '%d/%m/%Y').strftime(date_format_out)
        debit_val = trans['debit'].replace('.', decimal_sep) if trans['debit'] else ""
        credit_val = trans['credit'].replace('.', decimal_sep) if trans['credit'] else ""
        draw.text((margin, y_pos), trans_date, font=font_regular, fill="black")
        draw.text((margin + 100, y_pos), trans['description'], font=font_regular, fill="black")
        if layout_type in ['2_colonnes', 'grille']:
            draw.text((width - margin - 250, y_pos), debit_val, font=font_regular, fill="black")
            draw.text((width - margin - 120, y_pos), credit_val, font=font_regular, fill="black")
        elif layout_type == '1_colonne_signee':
            montant = f"-{debit_val}" if debit_val else f"+{credit_val}"
            draw.text((width - margin - 150, y_pos), montant, font=font_regular, fill="black")
        elif layout_type == 'decale':
            if debit_val: draw.text((width - margin - 250, y_pos), debit_val, font=font_regular, fill="black")
            if credit_val: draw.text((width - margin - 120, y_pos + 5), credit_val, font=font_regular, fill="gray")
        y_pos += 25
        if layout_type == 'grille':
            draw.line([(margin, y_pos - 5), (width - margin, y_pos - 5)], fill="lightgray", width=1)

    y_pos += random.randint(30, 60)
    balance_x = random.choice([margin, width - margin - 300])
    draw.text((balance_x, y_pos), f"Solde précédent : {data['start_balance'].replace('.', decimal_sep)} €", fill="black", font=font_regular)
    y_pos += 25
    draw.text((balance_x, y_pos), f"Nouveau solde : {data['end_balance'].replace('.', decimal_sep)} €", fill="black", font=font_bold)

    if random.random() > 0.3:
        angle = random.uniform(-1.5, 1.5)
        image = image.rotate(angle, expand=True, fillcolor='white')
        draw = ImageDraw.Draw(image)
        for _ in range(int(width * height * 0.001)):
            x, y = random.randint(0, image.width - 1), random.randint(0, image.height - 1)
            draw.point((x, y), fill="lightgray")

    image.save(output_path)

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
            create_hyper_realistic_image(statement_data, output_path)
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