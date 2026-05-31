import re
from spellchecker import SpellChecker

# =====================================================================
# PROCESSING LAYER: LLM-BASED NORMALIZATION PIPELINE
# =====================================================================

# Tetap pertahankan untuk deteksi bahasa (opsional)
english_checker = SpellChecker(language='en')
FILLER_PATTERN = re.compile(r'\b(um+|uh+|eh+|ah+|hmm+|mmm+|err+|erm+)\b', re.IGNORECASE)
NOISE_PATTERN = re.compile(r'[^\w\s.,!?\'\"()-]')

def is_arabic_transliteration(word: str) -> bool:
    clean = re.sub(r'[^a-zA-Z]', '', word).lower()
    arabic_lexicon = {
        "akhi", "ukhti", "umrah", "umroh", "haji", "hajj", "sholat", "uridu", 
        "qadim", "mubasharah", "rihlatan", "syukron", "jazakallah", "waalaikum",
        "assalam", "makkah", "madinah", "jeddah", "riyadh", "masjid", "haram"
    }
    if clean in arabic_lexicon: return True
    arabic_pattern = r'(al|kh|dh|th|dz|gh|ii|aa|uu)'
    return len(clean) > 3 and bool(re.search(arabic_pattern, clean))

def normalize_transcript(text: str) -> str:
    """Pembersihan dasar saja, sisanya diserahkan ke LLM."""
    text = FILLER_PATTERN.sub('', text)
    text = NOISE_PATTERN.sub(' ', text)
    return text.strip()

def detect_language_tokens(text: str) -> dict:
    words = text.split()
    id_cnt, en_cnt, ar_cnt = 0, 0, 0
    for word in words:
        clean = re.sub(r'[^a-zA-Z]', '', word).lower()
        if not clean: continue
        if len(english_checker.unknown([clean])) == 0: en_cnt += 1
        elif is_arabic_transliteration(clean): ar_cnt += 1
        else: id_cnt += 1
            
    total = max(id_cnt + en_cnt + ar_cnt, 1)
    if ar_cnt / total > 0.25: dominant = 'ar'
    elif en_cnt / total > 0.35: dominant = 'en'
    else: dominant = 'id'
    
    return {
        "dominant": dominant,
        "ratios": {"id": round(id_cnt/total, 2), "en": round(en_cnt/total, 2), "ar": round(ar_cnt/total, 2)}
    }

def build_augmented_prompt(original_text: str, mode: str) -> tuple[str, dict]:
    normalized_base = normalize_transcript(original_text)
    lang_info = detect_language_tokens(normalized_base)
    
    # Instruksi memaksa output JSON
    prompt = f"""[SYSTEM INSTRUCTION]
Anda adalah asisten virtual ahli. Berikan jawaban dalam format JSON murni tanpa teks tambahan.
Format JSON:
{{
  "perbaikan": "teks baku hasil perbaikan",
  "tagging": "teks dengan tag [EN:...], [AR:...], [ID:...]",
  "respon": "jawaban Anda untuk user"
}}

[INPUT USER]
{original_text}
"""
    metadata = {
        "language": lang_info, 
        "mode": mode
    }
    return prompt, metadata