import json
import base64
import requests
import re
from pathlib import Path
from PIL import Image, ImageEnhance
from PIL.ExifTags import TAGS


# === KONFIGURATION ============================================================

DIR_SOURCE   = r"D:\daten\NextCloud\share\Lichtblick\Venedig"
DIR_INPUT    = Path(DIR_SOURCE) / "Auswahl"
DIR_OUTPUT   = Path(DIR_SOURCE) / "Angepasst"
OLLAMA_URL    = "http://eva:11434/api/generate"
#MODELL        = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"   # oder z.B. "llava"
MODELL        = "qwen3-vl:4b"  # oder z.B. "llava"
#MODELL        = "hf.co/mradermacher/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M"  # Modell tut nicht -> Internal Server Error
MAX_ROTATION = 10
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE   = (0.8, 1.2)
SAT_RANGE        = (0.5, 1.5)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "crop": {
            "type": "object",
            "properties": {
                "left":   {"type": "integer", "minimum": 0},
                "top":    {"type": "integer", "minimum": 0},
                "right":  {"type": "integer", "minimum": 0},
                "bottom": {"type": "integer", "minimum": 0}
            },
            "required": ["left", "top", "right", "bottom"],
            "additionalProperties": False
        },
        "rotation": {
            "type": "number",
            "minimum": -10,
            "maximum": 10
        },
        "comment": {
            "type": "string"
        },
        "color": {
            "type": "object",
            "properties": {
                "recommended": {
                    "type": "boolean"
                },
                "brightness": {
                    "type": "number",
                    "minimum": 0.7,
                    "maximum": 1.3
                },
                "contrast": {
                    "type": "number",
                    "minimum": 0.8,
                    "maximum": 1.2
                },
                "saturation": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 1.5
                }
            },
            "required": ["brightness", "contrast", "saturation", "recommended"],
            "additionalProperties": False
        }
    },
    "required": ["crop", "rotation", "color", "comment"],
    "additionalProperties": False
}


PROMPT = (
    "Du bist ein Profi-Fotograf. Analysiere dieses Bild und schlage einen optimalen Ausschnitt, "
    "eine sinnvolle Rotation und Basis-Farbkorrekturen vor. \n\n"
    "Bei der Auswahl des Bildausschnitt müssen ästetische Regeln wie die Drittelregel oder Mittige Ausrichtung, "
    "Störfaktoren entfernen, Führungslinien nutzen. Es soll immer Originale Bildverhältnis "
    "(Bildhöhe / Bildbreite =  Bildhöhe2 / Bildbreite2) beibehalten bleiben\n"
    f"ANTWORTE IMMER ALS REINES JSON-OBJEKT nach folgendem Muster: {json.dumps(JSON_SCHEMA, indent=2)}"
    "'crop' enthält die Pixel-Koordinaten für den optimalen Ausschnitt des Motivs im Bild. "
    "Die obere linke Ecke des ORIGINALBILDES hat die Koordinate (0,0)."
    "Die x-Achse (Pixel) verläuft NACH RECHTS, die y-Achse NACH UNTEN.\n"
    "'left' ist die x-Koordinate der linken Kante des Ausschnitts.\n"
    "'top' ist die y-Koordinate der oberen Kante des Ausschnitts.\n"
    "'right' ist die x-Koordinate der rechten Kante des Ausschnitts.\n"
    "'bottom' ist die y-Koordinate der unteren Kante des Ausschnitts.\n"
    "Es muss immer gelten: 0 <= left < right <= Bildbreite und 0 <= top < bottom <= Bildhöhe und right - left > 800 und bottom - top > 800.\n"
    "'rotation' ist die Drehung in Grad (Gleitkommazahl).\n"
    "'comment' Beschreibung in Textform was mit dem neuen Bildausschnitt verbessert wird\n"
    #"'comment' Beschreibung womit die Farbkorrekturen begründet werden. \n"
    "Positive Werte drehen im Uhrzeigersinn, negative gegen den Uhrzeigersinn. \n"
    "'color' enthält die Farbkorrektur-Faktoren als Gleitkommazahlen (1.0 = unverändert):\n"
    "'recommended': Du empfiehlst eine Farbanpassung (ja = True, nein = False)\n"
    "'brightness': Helligkeit (0.7 = dunkler, 1.3 = heller)\n"
    "'contrast': Kontrast (0.8 = flacher, 1.2 = stärker)\n"
    "'saturation': Farbsättigung (0.5 = fahler, 1.5 = kräftiger)\n\n"
    "=== WICHTIG ===\n"
    "GIB NUR EIN JSON-OBJEKT ZURÜCK!\n"
    "KEINE Markdown-Codeblocks (```)\n"
    "ES DÜRFEN MAXIMAL DIE HÄLFTE DES BILDES WEGGESCHNITTEN WERDEN\n"
)

def bild_zu_base64(pfad: Path) -> str:
    """Liest ein Bild von der Platte und gibt eine Base64-kodierte Zeichenkette zurück."""
    with open(pfad, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def clamp(value, low, high):
    return max(low, min(high, value))

def parse_adjustments(data):
    return {
        "crop": {
            "left": int(data["crop"]["left"]),
            "top": int(data["crop"]["top"]),
            "right": int(data["crop"]["right"]),
            "bottom": int(data["crop"]["bottom"])
        },
        "rotation": float(data["rotation"]),
        "comment": data["comment"],
        "color": {
            "recommended": data["color"]["recommended"],
            "brightness": float(data["color"]["brightness"]),
            "contrast": float(data["color"]["contrast"]),
            "saturation": float(data["color"]["saturation"])
        }
    }

def ask_ollama_for_adjustments(image_path: Path) -> dict:

    image_b64 = bild_zu_base64(image_path)

    payload = {
        "model": MODELL,
        "prompt": PROMPT,
        "images": [image_b64],
        "format": JSON_SCHEMA,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=800)
    response.raise_for_status()

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
        if "crop" in data:
            return parse_adjustments(data)
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
                if "crop" in data:
                    return parse_adjustments(data)
            except json.JSONDecodeError:
                pass

    # 3. Letzter Fallback
    score_match = re.search(r'"crop"\s*:\s*(\d+(?:\.\d+)?)', cleaned)
    if score_match:
        return {
            "crop": {
                "left": 0,
                "top": 0,
                "right": 0,
                "bottom": 0
            },
            "rotation": 0.0,
            "comment": "Fehler beim Parsen",
            "color": {
                "recommended": False,
                "brightness": 1.0,
                "contrast": 1.0,
                "saturation": 1.0
            }
        }

    # 4: DEBUG + Fail
    print(f"  DEBUG Roh-Antwort: {response.json()}")
    raise ValueError("Kein gültiges JSON/Score gefunden")

def apply_adjustments(
    image_path: Path,
    adjustments: dict,
    output_path: Path,
):
    img = Image.open(image_path)


    # Originalformat merken (für EXIF etc.)
    original_format = img.format

    # In RGB konvertieren NUR wenn nötig (für Bearbeitung)
    if img.mode != 'RGB':
        img = img.convert("RGB")

    w, h = img.size

    crop = adjustments.get("crop", {})
    left = int(crop.get("left", 0))
    top = int(crop.get("top", 0))
    right = int(crop.get("right", w))
    bottom = int(crop.get("bottom", h))

    # Begrenzen auf Bildgrenzen
    left = clamp(left, 0, w - 1)
    top = clamp(top, 0, h - 1)
    right = clamp(right, left + 1, w)
    bottom = clamp(bottom, top + 1, h)
    #img = img.crop((left, top, right, bottom))

    # Rotation
    rotation = float(adjustments.get("rotation", 0.0))
    rotation = clamp(rotation, -MAX_ROTATION, MAX_ROTATION)
    # Pillow rotiert gegen den Uhrzeigersinn, daher Vorzeichen drehen
    #img = img.rotate(-rotation, expand=True)

    # Farbkorrekturen
    color_adj = adjustments.get("color", {})
    recommended = color_adj.get("recommended", False)
    brightness = float(color_adj.get("brightness", 1.0))
    contrast = float(color_adj.get("contrast", 1.0))
    saturation = float(color_adj.get("saturation", 1.0))

    brightness = clamp(brightness, *BRIGHTNESS_RANGE)
    contrast = clamp(contrast, *CONTRAST_RANGE)
    saturation = clamp(saturation, *SAT_RANGE)

    if recommended:
        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        if saturation != 1.0:
            img = ImageEnhance.Color(img).enhance(saturation)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # EXIF-Daten übernehmen
    exif_raw = img.info.get('exif') if 'exif' in img.info else None

    if recommended:
        if original_format == 'JPEG' or output_path.suffix.lower() in ['.jpg', '.jpeg']:
            img.save(output_path, 'JPEG', quality=95)  # JPEG mit Qualität 95 und EXIF
        elif original_format == 'PNG' or output_path.suffix.lower() == '.png':
            img.save(output_path, 'PNG', compress_level=6)  # PNG-Kompression (0-9)
        elif original_format == 'TIFF' or output_path.suffix.lower() == '.tiff':
            img.save(output_path, 'TIFF', compression='none')  # Lossless
        else:
            img.save(output_path, quality=95)  # Fallback
    print(f"Gespeichert: {output_path}")

def process_image_batch(input_dir: Path, output_dir: Path):
    """Verarbeitet alle Bilder im Eingabeordner."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Eingabeordner nicht gefunden: {input_dir}")

    # Unterstützte Formate
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(ext))
        image_files.extend(input_dir.glob(ext.upper()))  # Auch Großbuchstaben

    if not image_files:
        print(f"❌ Keine Bilddateien im Ordner {input_dir} gefunden.")
        return

    print(f"📁 Gefundene Bilder: {len(image_files)}")

    # Ausgabeordner erstellen
    print(f"📁 Ausgabeordner: {output_dir}")

    processed = 0
    for image_path in image_files:
        try:
            print(f"\n🔄 Verarbeite: {image_path.name}")

            # Ollama-Anpassungen abrufen
            adjustments = ask_ollama_for_adjustments(image_path)
            print(f"   Anpassungen: Crop({adjustments['crop']}), Rot:{adjustments['rotation']:.1f}°")
            print(f"   Farben: R{adjustments['color']['recommended']} B{adjustments['color']['brightness']:.2f} C{adjustments['color']['contrast']:.2f} S{adjustments['color']['saturation']:.2f}")
            print(f"   Kommentar: {adjustments['comment']}")

            # Ausgabename erstellen
            stem = image_path.stem
            suffix = image_path.suffix
            output_path = output_dir / f"{stem}_adjusted{suffix}"

            # Anwenden und speichern
            apply_adjustments(image_path, adjustments, output_path)
            processed += 1

        except Exception as e:
            print(f"❌ Fehler bei {image_path.name}: {e}")
            continue

    print(f"\n✅ Fertig! {processed}/{len(image_files)} Bilder verarbeitet.")
    print(f"📁 Ergebnisse: {output_dir}")

def main():
    process_image_batch(DIR_INPUT, DIR_OUTPUT)


if __name__ == "__main__":
    main()