import json
import base64
import requests
import re
from pathlib import Path
from PIL import Image, ImageEnhance


# === KONFIGURATION ============================================================
BILD_INPUT    = Path(r"D:\daten\NextCloud\share\Lichtblick\Venedig\DSC03190.JPG")  # <-- anpassen
BILD_OUTPUT   = Path(r"D:\daten\NextCloud\share\Lichtblick\Venedig\DSC031902.JPG")
OLLAMA_URL    = "http://eva:11434/api/generate"
#MODELL        = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"   # oder z.B. "llava"
MODELL        = "qwen3-vl:4b"  # oder z.B. "llava"
#MODELL        = "hf.co/mradermacher/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M"  # Modell tut nicht -> Internal Server Error
MAX_ROTATION = 10
BRIGHTNESS_RANGE = (0.3, 3.0)
CONTRAST_RANGE   = (0.3, 3.0)
SAT_RANGE        = (0.3, 3.0)

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
                "brightness": {
                    "type": "number",
                    "minimum": 0.3,
                    "maximum": 3.0
                },
                "contrast": {
                    "type": "number",
                    "minimum": 0.3,
                    "maximum": 3.0
                },
                "saturation": {
                    "type": "number",
                    "minimum": 0.3,
                    "maximum": 3.0
                }
            },
            "required": ["brightness", "contrast", "saturation"],
            "additionalProperties": False
        }
    },
    "required": ["crop", "rotation", "color"],
    "additionalProperties": False
}


PROMPT = (
    "Du bist ein Profi-Fotograf. Analysiere dieses Bild und schlage einen optimalen Ausschnitt, "
    "eine sinnvolle Rotation und Basis-Farbkorrekturen vor. \n\n"
    "Bei der Auswahl des Bildausschnitt müssen ästetische Regeln wie die Drittelregel, Mittige Ausrichtung, "
    "Störfaktoren entfernen, Führungslinien nutzen. Es soll immer Originale Bildverhältnis beibehalten bleiben\n"
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
    "Positive Werte drehen im Uhrzeigersinn, negative gegen den Uhrzeigersinn. \n"
    "'color' enthält die Farbkorrektur-Faktoren als Gleitkommazahlen (1.0 = unverändert):\n"
    "'brightness': Helligkeit (0.3 = dunkler, 3.0 = heller)\n"
    "'contrast': Kontrast (0.3 = flacher, 3.0 = stärker)\n"
    "'saturation': Farbsättigung (0.3 = fahler, 3.0 = kräftiger)\n\n"
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
    """
    Wendet Crop, Rotation und Farbkorrekturen auf ein Bild an
    und speichert das Ergebnis.
    """
    img = Image.open(image_path).convert("RGB")
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
    brightness = float(color_adj.get("brightness", 1.0))
    contrast = float(color_adj.get("contrast", 1.0))
    saturation = float(color_adj.get("saturation", 1.0))

    brightness = clamp(brightness, *BRIGHTNESS_RANGE)
    contrast = clamp(contrast, *CONTRAST_RANGE)
    saturation = clamp(saturation, *SAT_RANGE)

    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=95)
    print(f"Gespeichert: {output_path}")

def main():
    if not BILD_INPUT.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {BILD_INPUT}")

    if BILD_OUTPUT.exists():
        raise FileNotFoundError(f"Bild gefunden und will nicht überschreiben: {BILD_OUTPUT}")

    adjustments = ask_ollama_for_adjustments(BILD_INPUT)
    print(json.dumps(adjustments, indent=2))

    try:
        apply_adjustments(BILD_INPUT, adjustments, BILD_OUTPUT)
    except Exception as e:
        print(f"Fehler bei der Bildbearbeitung: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()