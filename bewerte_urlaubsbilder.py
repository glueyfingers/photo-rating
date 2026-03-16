import os
import csv
import json
import base64
import time
import shutil
import re
from pathlib import Path

import requests  # pip install requests

# === KONFIGURATION ============================================================
BILDER_ORDNER = Path(r"E:\photos\2026\Venedig\Kamera")  # <-- anpassen
AUSWAHL_ORDNER = BILDER_ORDNER / "Auswahl"
CSV_OUTPUT    = Path(BILDER_ORDNER / "bewertung_urlaub.csv")
OLLAMA_URL    = "http://tom:11434/api/generate"
#MODELL        = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"   # oder z.B. "llava"
MODELL        = "qwen3-vl:4b"  # oder z.B. "llava"
#MODELL        = "hf.co/mradermacher/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M"  # Modell tut nicht -> Internal Server Error
UNTERSTUETZTE_ENDUNGEN = {".jpg", ".jpeg"}
TOP_N = 4  # wie viele Bilder übernommen werden sollen

PROMPT = (
    "Du bist ein extrem kritischer Profi-Fotograf. Bewerte dieses Urlaubsfoto nach diesen 20 Kriterien:\n\n"
    "📸 TECHNISCHE QUALITÄT (40%):\n"
    "1. Schärfe: Perfekter Fokus, keine Verwacklung?\n"
    "2. Belichtung: Korrekte Helligkeit?\n"
    "3. Weißabgleich: Natürliche Farben?\n"
    "4. Kontrast: Ausgewogene Hell-Dunkel?\n"
    "5. Rauschen: Saubere Details?\n"
    "6. Sättigung: Lebendig aber natürlich?\n"
    "7. Dynamikumfang: Details in Licht+Schatten?\n"
    "8. Horizont: Gerade Linien?\n\n"
    "🎨 KOMPOSITION (30%):\n"
    "9. Drittelregel: Hauptmotiv richtig platziert?\n"
    "10. Führende Linien: Blickführung?\n"
    "11. Tiefenschärfe: Vorder-/Hintergrund?\n"
    "12. Bildfüllung: Keine leeren Bereiche?\n"
    "13. Formatwahl: Quer/Hoch passend?\n"
    "14. Symmetrie: Bei Landschaften?\n\n"
    "✨ BILDWIRKUNG (30%):\n"
    "15. Motivwahl: Originell/emotional?\n"
    "16. Moment: Perfekter Augenblick?\n"
    "17. Emotion: Berührt es?\n"
    "18. Storytelling: Klare Geschichte?\n"
    "19. Wow-Faktor: Einzigartig?\n"
    "20. Ausdruckskraft: Klare Botschaft?\n\n"
    "Score-Rechnung: (Technik×0.4)+(Komposition×0.3)+(Wirkung×0.3)\n\n"
    "ANTWORTEN IMMER ALS REINES JSON-OBJEKT nach folgendem Muster: {\"score\": 0-100, \"kommentar\": \"1-2 Sätze\", \"tags\": [\"tag1\",\"tag2\"]}\n"
    "'score' entspricht der Bewertung als Zahl. Wobei 0 einem sehr schlechtem Bild entspricht und 100 einem perfekten Bild entspricht\n"
    "'kommentar' sind Verbesserungsvorschläge\n"
    "'tags' soll keine Kritik enthalten! Ein Tag ist ein Wort, dass das Bild beschreibt. Es sollen pro Bild ca. 5 Tags aufgelistet werden\n"
    "Ignoriere die Auflösung des Bildes, konzentriere dich auf die Komposition des Bildes.\n\n"
    "=== WICHTIG ===\n"
    "GIB NUR EIN JSON-OBJEKT ZURÜCK!\n"
    "KEINE Markdown-Codeblocks (```)\n"
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

    JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "minimum": 0,
                "maximum": 100
            },
            "kommentar": {
                "type": "string",
                "maxLength": 500
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 50
                },
                "minItems": 3,
                "maxItems": 8
            }
        },
        "required": ["score", "kommentar", "tags"],
        "additionalProperties": False
    }

    payload = {
        "model": MODELL,
        "prompt": PROMPT,
        "images": [image_b64],
        "format": JSON_SCHEMA,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=800)
    response.raise_for_status()

    # ROH-ANTWORT direkt ausgeben für Debug
    #roh_response = response.json().get("thinking", "").strip()
    #roh_response = response.json().get("thinking", "").strip()
    full_response = response.json()
    roh_response = full_response.get("thinking", full_response.get("response", "")).strip()
    #print(f"DEBUG Roh: {repr(roh_response)[:200]}...")  # Zeigt escaped Zeichen

    # 1. Markdown entfernen
    cleaned = re.sub(r'```(?:json)?\s*', '', roh_response, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'^\s*', '', cleaned)  # Leading Whitespace

    try:
        # DIREKT JSON parsen (meistens klappt das!)
        data = json.loads(cleaned)
        if "score" in data:
            return {
                "score": float(data["score"]),
                "kommentar": data.get("kommentar", ""),
                "tags": data.get("tags", [])
            }
    except json.JSONDecodeError as e:
        print(f"JSON-Fehler: {e}")

    # 2. Fallback: Brute-Force JSON-Suche
    start = cleaned.find('{')
    if start != -1:
        end = cleaned.rfind('}')
        if end > start:
            try:
                candidate = cleaned[start:end+1]
                data = json.loads(candidate)
                if "score" in data:
                    return {
                        "score": float(data["score"]),
                        "kommentar": data.get("kommentar", ""),
                        "tags": data.get("tags", [])
                    }
            except json.JSONDecodeError:
                pass

    # 3. Letzter Fallback
    score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', cleaned)
    if score_match:
        return {
            "score": float(score_match.group(1)),
            "kommentar": "Score extrahiert",
            "tags": []
        }

    # 4: DEBUG + Fail
    print(f"  DEBUG Roh-Antwort: {response.json()}")
    raise ValueError("Kein gültiges JSON/Score gefunden")


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