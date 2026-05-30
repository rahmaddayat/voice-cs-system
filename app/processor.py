import re

# =====================================================================
# Processing Layer: Normalisasi Transkrip, Deteksi Bahasa, Code-Switching
# =====================================================================

# Daftar kata umum bahasa Inggris untuk deteksi
ENGLISH_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "this", "that", "these", "those", "and", "but", "or", "nor",
    "for", "so", "yet", "because", "if", "then", "than", "as", "at",
    "by", "for", "from", "in", "into", "of", "on", "out", "to", "up",
    "with", "about", "after", "before", "during", "between", "through",
    "not", "no", "yes", "please", "thank", "thanks", "hello", "hi",
    "what", "how", "much", "many", "time", "day", "year", "good", "bad",
    "big", "small", "new", "old", "first", "last", "here", "there",
    "now", "just", "also", "very", "too", "only", "same", "more",
    "natural", "language", "processing", "model", "system", "data"
}

# Pola skrip Arab (Unicode range U+0600 – U+06FF)
ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF]+')

# Teks filler / noise STT yang sering muncul
FILLER_PATTERN = re.compile(
    r'\b(um+|uh+|eh+|ah+|hmm+|mmm+|err+|erm+)\b', re.IGNORECASE
)

# Simbol dan tanda baca berlebih yang perlu dibersihkan
NOISE_PATTERN = re.compile(r'[^\w\s\u0600-\u06FF.,!?\'\"()-]')


def normalize_transcript(text: str) -> str:
    """
    Normalisasi teks transkripsi STT:
    - Hapus filler words (um, uh, eh, dll.)
    - Hapus simbol/karakter yang tidak relevan
    - Normalkan spasi ganda
    - Pertahankan huruf kapital awal kalimat
    """
    text = FILLER_PATTERN.sub('', text)
    text = NOISE_PATTERN.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Pastikan huruf kapital di awal kalimat
    if text:
        text = text[0].upper() + text[1:]
    return text


def detect_language_tokens(text: str) -> dict:
    """
    Deteksi token bahasa pada level kata.
    Mengembalikan dict berisi:
    - 'id'  : daftar kata Bahasa Indonesia
    - 'en'  : daftar kata Bahasa Inggris
    - 'ar'  : daftar segmen Bahasa Arab
    - 'dominant' : bahasa dominan ('id', 'en', 'ar', 'mixed')
    """
    words = text.split()
    id_words, en_words, ar_words = [], [], []

    for word in words:
        clean = re.sub(r'[^a-zA-Z\u0600-\u06FF]', '', word).lower()
        if not clean:
            continue
        if ARABIC_PATTERN.search(clean):
            ar_words.append(word)
        elif clean in ENGLISH_WORDS:
            en_words.append(word)
        else:
            id_words.append(word)

    total = max(len(id_words) + len(en_words) + len(ar_words), 1)
    id_ratio  = len(id_words) / total
    en_ratio  = len(en_words) / total
    ar_ratio  = len(ar_words) / total

    # Tentukan bahasa dominan
    if ar_ratio > 0.4:
        dominant = 'ar'
    elif en_ratio > 0.5:
        dominant = 'en'
    elif id_ratio > 0.5:
        dominant = 'id'
    else:
        dominant = 'mixed'

    return {
        "id": id_words,
        "en": en_words,
        "ar": ar_words,
        "dominant": dominant,
        "ratios": {
            "id": round(id_ratio, 2),
            "en": round(en_ratio, 2),
            "ar": round(ar_ratio, 2),
        }
    }


def tag_code_switching(text: str) -> str:
    """
    Menandai (tagging) setiap kata dengan label bahasa masing-masing.
    Contoh output:
      "[ID:Halo] [ID:aku] [EN:want] [EN:to] [ID:pergi] [AR:مدرسة]"
    """
    words = text.split()
    tagged = []
    for word in words:
        clean = re.sub(r'[^a-zA-Z\u0600-\u06FF]', '', word).lower()
        if not clean:
            tagged.append(word)
            continue
        if ARABIC_PATTERN.search(clean):
            tagged.append(f"[AR:{word}]")
        elif clean in ENGLISH_WORDS:
            tagged.append(f"[EN:{word}]")
        else:
            tagged.append(f"[ID:{word}]")
    return ' '.join(tagged)


def build_augmented_prompt(original_text: str, mode: str) -> tuple[str, dict]:
    """
    Membangun prompt yang diperkaya untuk LLM beserta metadata processing layer.

    Args:
        original_text (str): Teks transkripsi asli dari STT.
        mode (str): Mode sistem LLM ('preserve', 'normalize', 'translate').

    Returns:
        tuple: (prompt_untuk_llm, metadata_dict)
    """
    # Langkah 1: Normalisasi transkrip
    normalized = normalize_transcript(original_text)

    # Langkah 2: Deteksi bahasa
    lang_info = detect_language_tokens(normalized)

    # Langkah 3: Penandaan code-switching
    tagged = tag_code_switching(normalized)

    # Langkah 4: Bangun prompt sesuai mode
    lang_names = {"id": "Indonesia", "en": "Inggris", "ar": "Arab", "mixed": "Campuran"}
    dominant_name = lang_names.get(lang_info['dominant'], 'Campuran')

    prompt_header = f"[INFO: Input mendeteksi bahasa dominan = {dominant_name}]\n"
    prompt_header += f"[Code-switching tags: {tagged}]\n\n"

    if mode == "normalize":
        prompt = f"{prompt_header}User (sudah dinormalisasi ke Bahasa Indonesia): {normalized}"
    elif mode == "translate":
        prompt = f"{prompt_header}User (minta diterjemahkan ke Bahasa Indonesia): {normalized}"
    else:  # 'preserve' (default)
        prompt = f"{prompt_header}User: {normalized}"

    metadata = {
        "original":   original_text,
        "normalized": normalized,
        "tagged":     tagged,
        "language":   lang_info,
        "mode":       mode,
    }

    return prompt, metadata
