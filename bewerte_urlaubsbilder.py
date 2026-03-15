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
CSV_OUTPUT    = Path("bewertung_urlaub.csv")
OLLAMA_URL    = "http://tom:11434/api/generate"
#MODELL        = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"   # oder z.B. "llava"
MODELL        = "qwen3-vl:4b"  # oder z.B. "llava"
#MODELL        = "hf.co/mradermacher/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M"  # Modell tut nicht -> Internal Server Error
UNTERSTUETZTE_ENDUNGEN = {".jpg", ".jpeg"}
TOP_N = 20  # wie viele Bilder übernommen werden sollen

PROMPT = (
    "Du bist ein extrem kritischer Profi-Fotograf für Urlaubsbilder.\n"
    "Bewerte nach diesen Kriterien:\n"
    "1. Perfekte Schärfe (keine Verwacklung, nur gewollte Bewegungsunschärfe)\n"
    "2. Korrekte Belichtung (keine Über-/Unterbelichtung)\n"
    "3. Bei Personen im Hauptmotiv sollen die Augen geöffnet sein und die Gesichter sind zu erkennen\n"
    "4. Starker Bildaufbau (Drittelregel, führende Linien oder Natürlichkeit)\n"
    "5. Komplementärfarben\n"
    "ANTWORTEN SIE IMMER ALS REINES JSON-OBJEKT nach folgendem Muster: {\"score\": 0-100, \"kommentar\": \"1-2 Sätze\", \"tags\": [\"tag1\",\"tag2\"]}\n"
    "'score' entspricht der Bewertung als Zahl. Wobei 0 einem sehr schlechtem Bild entspricht und 100 einem perfekten Bild entspricht\n"
    "'kommentar' sind Verbesserungsvorschläge\n"
    "'tags' soll keine Kritik enthalten! Ein Tag ist ein Wort, dass das Bild beschreibt. Es sollen pro Bild ca. 5 Tags aufgelistet werden\n"
    "Sei brutal ehrlich - 80% deiner Urlaubs-Schnappschüsse sind Durchschnitt! Ignoriere die Auflösung des Bildes, konzentriere dich auf die Komposition des Bildes. Keine Nettigkeiten, nur knallharte Kritik und ehrliche Bewertung!\n\n"
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

    payload = {
        "model": MODELL,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    roh_text = response.json().get("response", "").strip()

    # === NEUE STRATEGIE: Alle möglichen JSONs extrahieren ===

    # 1. Markdown-Codeblocks entfernen (```json ... ```)
    roh_text = re.sub(r'```(?:json)?\s*', '', roh_text, flags=re.DOTALL)
    roh_text = re.sub(r'```\s*', '\n', roh_text)

    # 2. Alle JSON-Objekte finden
    json_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', roh_text, re.DOTALL))

    for match in json_matches:
        try:
            data = json.loads(match.group(0))
            if "score" in data and isinstance(data.get("score"), (int, float)):
                return {
                    "score": data.get("score"),
                    "kommentar": data.get("kommentar", ""),
                    "tags": data.get("tags", []),
                }
        except json.JSONDecodeError:
            continue

    # 3. Fallback: Beste Score aus Text extrahieren
    score_matches = re.findall(r'score\s*[:\-=]\s*(\d+(?:\.\d+)?)', roh_text, re.IGNORECASE)
    if score_matches:
        best_score = max(float(s) for s in score_matches)
        return {
            "score": int(best_score),
            "kommentar": "Score aus mehreren Antworten extrahiert",
            "tags": []
        }

    # 4. DEBUG + Fail
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