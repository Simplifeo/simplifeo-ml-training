import torch

# --- 1. Vérification ---
print(f"Version de PyTorch installée : {torch.__version__}")

if torch.backends.mps.is_available():
    # Définit le device sur MPS s'il est disponible
    device = torch.device("mps")
    print(f"Le backend MPS est disponible. Utilisation du device : '{device}'")
    
    # Une vérification supplémentaire : est-ce que PyTorch a été compilé avec le support MPS ?
    # C'est généralement inclus dans is_available(), mais c'est bon à savoir.
    if not torch.backends.mps.is_built():
        print("AVERTISSEMENT : PyTorch n'a pas été compilé avec le support MPS.")

else:
    # Si MPS n'est pas disponible, on se rabat sur le CPU
    device = torch.device("cpu")
    print(f"Le backend MPS n'est pas disponible. Utilisation du device : '{device}'")


# --- 2. Utilisation pratique ---
print("\n--- Démonstration d'utilisation ---")

# Crée un tenseur (par défaut sur le CPU)
x_cpu = torch.randn(3, 3)
print(f"Tenseur créé initialement sur : {x_cpu.device}")

# Déplace le tenseur vers le device choisi (MPS ou CPU)
x_device = x_cpu.to(device)
print(f"Tenseur déplacé sur : {x_device.device}")

# Effectue une opération sur le device
y = x_device @ x_device.T  # Multiplication matricielle
print("Une opération a été effectuée sur le device.")
print("Résultat :")
print(y)