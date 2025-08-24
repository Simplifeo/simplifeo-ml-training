# ocr_releve.py  (version corrigée : montants ., IBAN/compte robustes, soldes, débit/crédit)
import re, json
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import dateparser

# Décommente si Tesseract n'est pas trouvé (Mac Apple Silicon/Homebrew) :
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# -------- REGEX plus souples --------
AMOUNT_RE   = re.compile(r"([+-]?\d{1,3}(?:[ \u00A0]\d{3})*[.,]\d{2})$")  # accepte , ou .
DATE_RE     = re.compile(r"\b(\d{2}/\d{2}(?:/\d{2,4})?)\b")
IBAN_RE     = re.compile(r"\bFR\d{2}[0-9A-Z ]{21,25}\b")  # IBAN FR sans hypothèses trop fortes
# tolère Compte/Compie/Comp..e + fallback N° 123456...
ACCT_MAIN   = re.compile(r"(Comp\w*e|N[°o]\s*compte|Account)\s*(?:n[°o]\s*)?:?\s*([A-Z0-9\- ]{6,})", re.IGNORECASE)
ACCT_FALLBK = re.compile(r"\bN[°o]\s*([0-9]{6,})\b")
PERIOD_RE   = re.compile(r"\bdu\s*(\d{2}/\d{2}/\d{2,4})\s*(?:au|-|→)\s*(\d{2}/\d{2}/\d{2,4})\b", re.IGNORECASE)
BAL_PREV_RE = re.compile(r"Solde\s+pr[ée]c[ée]dent\s*:\s*([0-9\u00A0\s]+[.,]\d{2})", re.IGNORECASE)
BAL_NEW_RE  = re.compile(r"Nouveau\s+solde\s*:\s*([0-9\u00A0\s]+[.,]\d{2})", re.IGNORECASE)

# mots-clés pour classer le sens
CREDIT_PATS = [
    r"\bREMISE\b", r"\bVIREMENT\s+(DE|RECU)\b", r"\bVERSEMENT\b",
    r"\bAVOIR\b", r"\bREMBOURSEMENT\b", r"\bINTER[ÊE]TS?\b"
]
DEBIT_PATS = [
    r"\bPAIEMENT\s+CB\b", r"\bRETRAIT\s+DAB\b", r"\bPR[ÉE]L[ÈE]VEMENTS?\b",
    r"\bFRAIS\b", r"\bVIREMENT\s+EMIS\b"
]

def normalize_amount(s: str) -> float:
    s = s.replace("\u00A0", " ").replace(" ", "").replace(",", ".")
    return float(s)

def preprocess(img_path: Path):
    img = Image.open(img_path).convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    img = img.point(lambda p: 255 if p > 175 else 0)  # binarisation légère
    return img

def ocr_text(img) -> str:
    return pytesseract.image_to_string(img, lang="fra+eng", config=r'--oem 1 --psm 6')

def classify_sign(label: str) -> str:
    L = label.upper()
    for pat in CREDIT_PATS:
        if re.search(pat, L):
            return "credit"
    for pat in DEBIT_PATS:
        if re.search(pat, L):
            return "debit"
    return "unknown"

def parse_header_and_balances(text: str) -> Dict[str, Optional[str]]:
    iban = IBAN_RE.search(text)
    acct_main = ACCT_MAIN.search(text)
    acct = acct_main.group(2).strip() if acct_main else None
    if not acct:
        fb = ACCT_FALLBK.search(text)
        acct = fb.group(1).strip() if fb else None

    period_match = PERIOD_RE.search(text)
    period = None
    if period_match:
        d1 = dateparser.parse(period_match.group(1), languages=["fr"])
        d2 = dateparser.parse(period_match.group(2), languages=["fr"])
        if d1 and d2:
            period = {"from": d1.date().isoformat(), "to": d2.date().isoformat()}

    prev = BAL_PREV_RE.search(text)
    new  = BAL_NEW_RE.search(text)

    return {
        "iban": iban.group(0) if iban else None,
        "account_number": acct,
        "period": period,
        "balance_previous": normalize_amount(prev.group(1)) if prev else None,
        "balance_new": normalize_amount(new.group(1)) if new else None,
    }

def parse_operations(text: str) -> List[Dict]:
    ops = []
    for ln in [l.strip() for l in text.splitlines() if l.strip()]:
        mdate = DATE_RE.search(ln)
        mamt  = AMOUNT_RE.search(ln)
        if not (mdate and mamt):
            continue
        raw_date = mdate.group(1)
        d = dateparser.parse(raw_date, languages=["fr"])
        date_iso = d.date().isoformat() if d else None
        label = ln[mdate.end():mamt.start()].strip(" -–:\t")
        try:
            amount = normalize_amount(mamt.group(1))
        except:
            continue
        sense = classify_sign(label)
        ops.append({
            "date": date_iso,          # peut être None si OCR a fait 40/04/2020
            "raw_date": raw_date,
            "label": label,
            "amount": amount,
            "sense": sense,            # "credit" | "debit" | "unknown"
            "signed": None
        })
    return ops

def reconcile_signs(ops: List[Dict], bal_prev: Optional[float], bal_new: Optional[float]) -> Dict:
    total_known = 0.0
    sum_unknown = 0.0
    for o in ops:
        if o["sense"] == "credit":
            o["signed"] =  +o["amount"]; total_known += o["signed"]
        elif o["sense"] == "debit":
            o["signed"] =  -o["amount"]; total_known += o["signed"]
        else:
            sum_unknown += o["amount"]

    inferred = None
    if bal_prev is not None and bal_new is not None and sum_unknown > 0:
        expected_net = bal_new - bal_prev
        delta = round(expected_net - total_known, 2)
        if abs(delta - sum_unknown) < 0.02:
            for o in ops:
                if o["sense"] == "unknown":
                    o["sense"] = "credit"; o["signed"] = +o["amount"]
            inferred = "unknown->credit"
        elif abs(delta + sum_unknown) < 0.02:
            for o in ops:
                if o["sense"] == "unknown":
                    o["sense"] = "debit"; o["signed"] = -o["amount"]
            inferred = "unknown->debit"

    debit_total  = round(sum(-o["signed"] for o in ops if o["signed"] is not None and o["signed"] < 0), 2)
    credit_total = round(sum( o["signed"] for o in ops if o["signed"] is not None and o["signed"] > 0), 2)
    net = round(credit_total - debit_total, 2)

    check = None
    if bal_prev is not None and bal_new is not None:
        check = abs((bal_prev + net) - bal_new) < 0.02

    return {
        "operations": ops,
        "totals": {"debit": debit_total, "credit": credit_total, "net": net},
        "reconciliation": {"applied": inferred, "balanced": check}
    }

def extract_bank_statement(image_path: str) -> Dict:
    img_path = Path(image_path)
    img = preprocess(img_path)
    text = ocr_text(img)
    header = parse_header_and_balances(text)
    ops = parse_operations(text)
    reco = reconcile_signs(ops, header["balance_previous"], header["balance_new"])
    return {"source_image": str(img_path), "header": header, **reco, "raw_text_preview": "\n".join(text.splitlines()[:30])}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Chemin de l'image PNG/JPG du relevé")
    args = p.parse_args()

    result = extract_bank_statement(args.image)
    out = Path("releve_extraction.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] JSON écrit dans {out.resolve()}")
    print(f"  - Opérations: {len(result['operations'])}")
    print(f"  - Totaux: +{result['totals']['credit']} / -{result['totals']['debit']} / net {result['totals']['net']}")
    bp = result['header'].get('balance_previous'); bn = result['header'].get('balance_new')
    if bp is not None and bn is not None:
        print(f"  - Solde précédent: {bp}  → Nouveau: {bn}  (OK={result['reconciliation']['balanced']})")
    if result['header'].get('iban'):
        print(f"  - IBAN détecté: {result['header']['iban']}")
    if result['header'].get('account_number'):
        print(f"  - Compte détecté: {result['header']['account_number']}")
