import json
import os
import requests

# Vos identifiants (récupérés de vos logs)
CLIENT_ID = "949184945625-132slr8b0sbjinaecmdaf7s84j3l6mk8.apps.googleusercontent.com"
CLIENT_SECRET = "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"

# Le chemin exact où DVC attend le fichier (Basé sur vos logs)
Target_Dir = f"/home/hamadygackou777/.cache/pydrive2fs/{CLIENT_ID}"
Target_File = os.path.join(Target_Dir, "default.json")

print("--- GÉNÉRATEUR DE CLÉ DVC ---")
code = input("Collez votre code Google (commençant par 4/0AS...) ici : ").strip()

print("Echange du code contre un token...")
# On demande manuellement le token à Google
response = requests.post(
    "https://oauth2.googleapis.com/token",
    data={
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": "http://localhost:8090/",  # C'est ce que DVC utilise par défaut
        "grant_type": "authorization_code",
    },
)

if response.status_code != 200:
    print(f"ERREUR : {response.text}")
    exit(1)

tokens = response.json()

# On construit le fichier JSON exactement comme DVC le veut
dvc_credentials = {
    "access_token": tokens["access_token"],
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "refresh_token": tokens.get("refresh_token", ""),
    "token_expiry": "2030-01-01T00:00:00Z", # Fake expiry pour forcer le refresh si besoin
    "user_agent": None,
    "invalid": False,
}

# On sauvegarde le fichier
os.makedirs(Target_Dir, exist_ok=True)
with open(Target_File, "w") as f:
    json.dump(dvc_credentials, f)

print(f"\n✅ SUCCÈS ! Fichier créé ici : {Target_File}")
print("Vous pouvez maintenant lancer 'dvc push' sans problème.")
