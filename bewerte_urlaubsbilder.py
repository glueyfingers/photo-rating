import os
import csv
import json
import base64
from pathlib import Path

import requests  # pip install requests

# === KONFIGURATION ============================================================
BILDER_ORDNER = Path(r"D:\daten\NextCloud\share\Lichtblick\Stadtbild")  # <-- anpassen
CSV_OUTPUT    = Path("bewertung_urlaub.csv")
OLLAMA_URL    = "http://eva:11434/api/generate"
MODELL        = "llama3.2-vision"  # oder z.B. "llava"
UNTERSTUETZTE_ENDUNGEN = {".jpg", ".jpeg"}

PROMPT = (
    "Du bist ein erfahrener Reise-Fotograf.\n"
    "Bewerte dieses Urlaubsfoto nach Schärfe, Belichtung, Bildgestaltung, Motiv und Stimmung.\n"
    "Gib NUR gültiges JSON im Format "
    "{\"score\": Zahl 0-10, \"kommentar\": \"kurzer Satz\", \"tags\": [\"tag1\", \"tag2\"]} aus.\n"
    "Score 0 = unbrauchbar, 10 = herausragendes Foto.\n"
    "Berücksichtige: offene Augen, keine starke Verwacklung, angenehme Farben, klar erkennbares Motiv.\n"
)

# === HILFSFUNKTIONEN ==========================================================

def sammle_bilder(ordner: Path):
    """Gibt eine Liste aller Bilddateien im Ordner zurück."""
    return [
        p for p in ordner.iterdir()
        if p.is_file() and p.suffix.lower() in UNTERSTUETZTE_ENDUNGEN
    ]

def bild_zu_base64(pfad: Path) -> str:
    """Liest ein Bild von der Platte und gibt eine Base64-kodierte Zeichenkette zurück."""
    with open(pfad, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def bewerte_bild(pfad: Path) -> dict:
    """Schickt ein Bild an Ollama und gibt das geparste JSON mit Score/Kommentar/Tags zurück."""
    image_b64 = bild_zu_base64(pfad)

    payload = {
        "model": MODELL,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    roh_text = response.json().get("response", "").strip()

    # Versuche, direkt JSON zu parsen; bei Fehler evtl. einfache „Reinigung“
    try:
        data = json.loads(roh_text)
    except json.JSONDecodeError:
        # Primitive Notfall-Reinigung: JSON-Block extrahieren (falls Modell etwas drumherum schreibt)
        start = roh_text.find("{")
        end = roh_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            kandidat = roh_text[start:end+1]
            data = json.loads(kandidat)
        else:
            raise

    return {
        "score": data.get("score"),
        "kommentar": data.get("kommentar", ""),
        "tags": data.get("tags", []),
    }

def schreibe_csv(ergebnisse: list, ziel_pfad: Path):
    """Schreibt die Bewertungsergebnisse in eine CSV-Datei."""
    with open(ziel_pfad, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["datei", "score", "kommentar", "tags"])
        for eintrag in ergebnisse:
            writer.writerow([
                eintrag["datei"],
                eintrag["score"],
                eintrag["kommentar"],
                ",".join(eintrag["tags"])
            ])

# === HAUPTLAUF ===============================================================

def main():
    if not BILDER_ORDNER.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {BILDER_ORDNER}")

    bilder = sammle_bilder(BILDER_ORDNER)
    if not bilder:
        print("Keine Bilder im Ordner gefunden.")
        return

    print(f"{len(bilder)} Bilder gefunden. Starte Bewertung mit Modell '{MODELL}' ...")

#    ergebnisse = []

#    for i, pfad in enumerate(bilder, start=1):
#        print(f"[{i}/{len(bilder)}] Verarbeite: {pfad.name}")
#        try:
#            result = bewerte_bild(pfad)
#            ergebnisse.append({
#                "datei": pfad.name,
#                **result
#            })
#            print(f" -> Score: {result['score']} | Tags: {', '.join(result['tags'])}")
#        except Exception as e:
#            print(f" !! Fehler bei {pfad.name}: {e}")

#    schreibe_csv(ergebnisse, CSV_OUTPUT)
#    print(f"Fertig. Ergebnisse in '{CSV_OUTPUT}' gespeichert.")
#    print("Öffne die CSV, sortiere nach 'score' absteigend und schau dir die Top-Bilder an.")

if __name__ == "__main__":
    main()