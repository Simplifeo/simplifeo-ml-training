import argparse, json, sys
from pathlib import Path
from PIL import Image

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

# --------- CLI ---------
parser = argparse.ArgumentParser(description="Test Donut sur une facture (image -> JSON)")
parser.add_argument("--image", type=str, required=True, help="Chemin de l'image (PNG/JPG)")
parser.add_argument("--model", type=str, default="naver-clova-ix/donut-base-finetuned-cord-v2",
                    help="Modèle Donut à utiliser")
args = parser.parse_args()

img_path = Path(args.image)
if not img_path.exists():
    print(f"Image introuvable: {img_path}")
    sys.exit(1)

# --------- Device (MPS > CUDA > CPU) ---------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"[info] Device: {device}")

# --------- Chargement modèle + processor ---------
print("[info] Téléchargement/chargement du modèle (peut prendre 1-2 minutes la première fois)...")
processor = DonutProcessor.from_pretrained(args.model)
model = VisionEncoderDecoderModel.from_pretrained(args.model)
model.to(device)
model.eval()

# --------- Ouvrir l'image ---------
image = Image.open(img_path).convert("RGB")

# --------- Préparer le prompt (spécifique au modèle CORD-v2) ---------
task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(
    task_prompt,
    add_special_tokens=False,
    return_tensors="pt"
).input_ids.to(device)

# --------- Encodage image ---------
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

# --------- Génération ---------
with torch.no_grad():
    outputs = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.config.max_length,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
    )

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# --------- Conversion en JSON Donut ---------
try:
    result = processor.token2json(sequence)
except Exception as e:
    result = {"raw": sequence, "note": f"Conversion JSON échouée: {e}"}

# --------- Affichage + sauvegarde ---------
print(json.dumps(result, indent=2, ensure_ascii=False))
out_path = Path("sortie_donut.json")
out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n[ok] Résultat enregistré dans: {out_path.resolve()}")
