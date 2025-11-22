import re
import unicodedata
from num2words import num2words
import roman

def process_numbers(text: str) -> str:
    """
    Converte:
    1. 'a.C.' / 'd.C.' -> avanti Cristo / dopo Cristo
    2. Numeri arabi -> lettere italiane (supporta 50.000)
    3. Numeri romani maiuscoli/minuscoli -> lettere italiane
    """
    # Espansione a.C. / d.C. (case-insensitive)
    text = re.sub(r'\b(d\.c\.)\b', 'dopo Cristo', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(a\.c\.)\b', 'avanti Cristo', text, flags=re.IGNORECASE)

    # Numeri arabi
    def arabic_to_words(match):
        numero_str = match.group()
        numero_int = int(numero_str.replace('.', ''))
        return num2words(numero_int, lang='it')

    text = re.sub(r'\b\d+(?:\.\d{3})*\b', arabic_to_words, text)

    # Numeri romani
    def roman_to_words(match):
        romano = match.group().upper()
        try:
            numero_int = roman.fromRoman(romano)
            return num2words(numero_int, lang='it')
        except roman.InvalidRomanNumeralError:
            return match.group()

    text = re.sub(r'\b[IVXLCDM]+\b', roman_to_words, text, flags=re.IGNORECASE)

    return text


def normalize_text(text: str) -> str:
    # NFKC normalizzazione
    text = unicodedata.normalize("NFKC", text)

    # 1. Rimuovi frasi tra parentesi quadre
    text = re.sub(r'\[.*?\]', '', text)
    # 2. Rimuovi frasi tra parentesi tonde
    text = re.sub(r'\(.*?\)', '', text)

    # 3. Sostituisci simboli/punteggiatura/marcatori con spazio
    text = ''.join(
        ch if not unicodedata.category(ch).startswith(('M', 'S', 'P')) else ' '
        for ch in text
    )

    # 4. Converte numeri in lettere (arabi, romani, date speciali)
    text = process_numbers(text)

    # 5. Minuscole
    text = text.lower()

    # 6. Sostituisci vocali accentate italiane
    replacements = {
        'à': 'a', 'è': 'e', 'é': 'e',
        'ì': 'i', 'ò': 'o', 'ù': 'u'
    }
    for accented, plain in replacements.items():
        text = text.replace(accented, plain)

    # 7. Mantieni solo lettere italiane e spazi
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 8. Spazio singolo
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_text_dummy(text: str) -> list:
    """
    Normalizza il testo rimuovendo punteggiatura, case e token speciali.
    """
    text = text.lower()
    text = re.sub(r"<eol>|<eob>", " ", text)         # Rimuove tag speciali
    text = re.sub(r"[^\w\s]", "", text)              # Rimuove punteggiatura
    text = re.sub(r"\s+", " ", text).strip()         # Rimuove spazi multipli
    return text