import os
import csv
import json
import base64
import time
import shutil
from pathlib import Path

import requests  # pip install requests

# === KONFIGURATION ============================================================
BILDER_ORDNER = Path(r"D:\daten\NextCloud\share\Lichtblick\Stadtbild")  # <-- anpassen
AUSWAHL_ORDNER = BILDER_ORDNER / "Auswahl"
CSV_OUTPUT    = Path("bewertung_urlaub.csv")
OLLAMA_URL    = "http://eva:11434/api/generate"
MODELL        = "qwen3-vl:4b"  # oder z.B. "llava"
UNTERSTUETZTE_ENDUNGEN = {".jpg", ".jpeg"}
TOP_N = 2  # wie viele Bilder übernommen werden sollen

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
        writer.writerow(["datei", "zeit_s", "score", "kommentar", "tags"])
        for eintrag in ergebnisse:
            writer.writerow([
                eintrag["datei"],
                eintrag["zeit_s"],
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

    ergebnisse = []
    gesamt_start = time.perf_counter()

    for i, pfad in enumerate(bilder, start=1):
        bild_start = time.perf_counter()
        print(f"[{i}/{len(bilder)}] Verarbeite: {pfad.name}")
        try:
            result = bewerte_bild(pfad)
            bild_zeit = time.perf_counter() - bild_start
            ergebnisse.append({
                "datei": pfad.name,
                "zeit_s": round(bild_zeit, 2),
                **result
            })
            print(f" -> Score: {result['score']} | Zeit: {bild_zeit:.2f}s | Tags: {', '.join(result['tags'])}")
        except Exception as e:
            print(f" !! Fehler bei {pfad.name}: {e}")

    gesamt_zeit = time.perf_counter() - gesamt_start
    schreibe_csv(ergebnisse, CSV_OUTPUT)
    print(f"Ergebnisse in '{CSV_OUTPUT}' gespeichert.")

    # Nur Einträge mit gültigem Score (nicht None) berücksichtigen
    gueltige = [e for e in ergebnisse if isinstance(e.get("score"), (int, float))]
    if not gueltige:
        print("Keine gültigen Scores, es werden keine Dateien kopiert.")
    else:
        # Nach Score absteigend sortieren
        gueltige.sort(key=lambda x: x["score"], reverse=True)
        top_bilder = gueltige[:TOP_N]

        # Auswahl-Ordner anlegen (falls nicht vorhanden)
        AUSWAHL_ORDNER.mkdir(exist_ok=True)

        print(f"\nKopiere die Top {len(top_bilder)} Bilder nach: {AUSWAHL_ORDNER}")
        for eintrag in top_bilder:
            src = BILDER_ORDNER / eintrag["datei"]
            dst = AUSWAHL_ORDNER / eintrag["datei"]
            try:
                shutil.copy2(src, dst)
                print(f" -> {eintrag['datei']} (Score {eintrag['score']}) kopiert")
            except Exception as e:
                print(f" !! Fehler beim Kopieren von {eintrag['datei']}: {e}")

    # === Zusammenfassung ===
    print(f"\n=== FERTIG ===")
    print(f"Gesamtzeit: {gesamt_zeit:.1f}s | Durchschnitt pro Bild: {gesamt_zeit/len(bilder):.2f}s")
    print(f"Ergebnisse in '{CSV_OUTPUT}' gespeichert.")
    print(f"Top-{TOP_N}-Bilder liegen in: '{AUSWAHL_ORDNER}'")

if __name__ == "__main__":
    main()