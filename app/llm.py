import os
from google import genai
from google.genai import types
from pydantic import TypeAdapter
from dotenv import load_dotenv

load_dotenv()

MODEL = "models/gemma-4-26b-a4b-it"

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

# =====================================================================
# TIGA MODE SISTEM
# =====================================================================

# MODE 1: PRESERVE (default)
# Asisten mempertahankan pola code-switching yang digunakan pengguna
SYSTEM_PRESERVE = """\
You are a multilingual virtual assistant fluent in Indonesian (Bahasa Indonesia), English, and Arabic.
The user may speak in a code-switching style — mixing Indonesian, English, and/or Arabic words naturally.

Your task:
- Respond in the SAME language mixture as the user. If the user mixes Indonesian and English, you mix them too.
- If the user speaks mostly Indonesian, respond mostly in Indonesian.
- If the user includes Arabic words or phrases, acknowledge them naturally.
- Keep responses SHORT (maximum 2–3 sentences) and DIRECT — do not repeat the question.
- Be warm, friendly, and natural in tone.

Example:
User: "Aku mau tanya, how much time from Banda Aceh to Medan?"
Assistant: "Perjalanan dari Banda Aceh ke Medan biasanya takes around 8 to 10 hours by bus, atau sekitar 1 jam kalau naik pesawat."
"""

# MODE 2: NORMALIZE (Normalisasi Bahasa)
# Asisten selalu merespons dalam Bahasa Indonesia baku, apapun bahasa input-nya
SYSTEM_NORMALIZE = """\
You are a virtual assistant that always responds in formal and polite Bahasa Indonesia (Indonesian language).
Regardless of what language the user speaks — whether Indonesian, English, Arabic, or a mixture — you must ALWAYS reply in standard Indonesian.

Your task:
- Translate and understand the user's input in any language.
- Respond ONLY in polite, standard Indonesian (Bahasa Indonesia baku).
- Keep responses SHORT (maximum 2–3 sentences) and DIRECT.
- Do not include any English or Arabic words in your response unless it is a proper noun.

Example:
User: "How much time from Banda Aceh to Medan?"
Assistant: "Perjalanan dari Banda Aceh ke Medan memakan waktu sekitar 8 hingga 10 jam menggunakan bus, atau sekitar 1 jam jika menggunakan pesawat terbang."
"""

# MODE 3: TRANSLATE (Terjemahan Opsional)
# Asisten menerjemahkan input pengguna ke Bahasa Indonesia lalu menjawab
SYSTEM_TRANSLATE = """\
You are a multilingual virtual assistant with expert translation capabilities.
The user speaks in a code-switching style using Indonesian, English, and/or Arabic.

Your task consists of two steps:
1. First, briefly show the TRANSLATION of the user's input into Indonesian (prefix with "Terjemahan:").
2. Then, answer the translated question in standard Indonesian (prefix with "Jawaban:").

Keep the translation and answer CONCISE (1 sentence each, maximum 2–3 sentences for the answer).
Be accurate, warm, and informative.

Example:
User: "How much time from Banda Aceh to Medan?"
Assistant:
Terjemahan: "Berapa lama waktu perjalanan dari Banda Aceh ke Medan?"
Jawaban: "Perjalanan dari Banda Aceh ke Medan membutuhkan sekitar 8 hingga 10 jam dengan bus, atau sekitar 1 jam menggunakan pesawat."
"""

# Mapping mode ke system instruction
SYSTEM_INSTRUCTIONS = {
    "preserve":  SYSTEM_PRESERVE,
    "normalize": SYSTEM_NORMALIZE,
    "translate": SYSTEM_TRANSLATE,
}

# =====================================================================

client = genai.Client(api_key=GOOGLE_API_KEY)
history_adapter = TypeAdapter(list[types.Content])

# Cache sesi chat per mode agar tidak perlu recreate terus
_chat_sessions: dict = {}

def _get_chat(mode: str):
    """
    Mengembalikan sesi chat yang sesuai dengan mode yang dipilih.
    Membuat sesi baru jika belum ada.
    """
    if mode not in _chat_sessions:
        instruction = SYSTEM_INSTRUCTIONS.get(mode, SYSTEM_PRESERVE)
        config = types.GenerateContentConfig(system_instruction=instruction)
        # Coba load history khusus per mode
        history_file = os.path.join(BASE_DIR, f"chat_history_{mode}.json")
        if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    json_str = f.read().strip()
                if json_str:
                    history = history_adapter.validate_json(json_str)
                    _chat_sessions[mode] = client.chats.create(
                        model=MODEL, config=config, history=history
                    )
                    return _chat_sessions[mode]
            except Exception as e:
                print(f"[WARNING] Gagal load history untuk mode '{mode}': {e}")
        _chat_sessions[mode] = client.chats.create(model=MODEL, config=config)
    return _chat_sessions[mode]


def _save_chat_history(mode: str, chat):
    """Simpan riwayat chat untuk mode tertentu."""
    try:
        history_file = os.path.join(BASE_DIR, f"chat_history_{mode}.json")
        json_history = history_adapter.dump_json(chat.get_history()).decode("utf-8")
        with open(history_file, "w", encoding="utf-8") as f:
            f.write(json_history)
    except Exception as e:
        print(f"[WARNING] Gagal menyimpan history chat: {e}")


def generate_response(prompt: str, mode: str = "preserve") -> str:
    """
    Kirim prompt ke LLM dan kembalikan respons teks.
    Args:
        prompt (str) : Teks prompt (sudah diproses oleh processor.py).
        mode (str)   : Mode sistem ('preserve', 'normalize', 'translate').
    Returns:
        str: Teks respons asisten.
    """
    try:
        chat = _get_chat(mode)
        response = chat.send_message(prompt)
        _save_chat_history(mode, chat)
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"


def reset_chat_history(mode: str = None):
    """
    Reset riwayat percakapan.
    Jika mode=None, reset semua mode.
    """
    global _chat_sessions
    if mode:
        _chat_sessions.pop(mode, None)
        history_file = os.path.join(BASE_DIR, f"chat_history_{mode}.json")
        if os.path.exists(history_file):
            os.remove(history_file)
    else:
        _chat_sessions.clear()
        for m in SYSTEM_INSTRUCTIONS:
            history_file = os.path.join(BASE_DIR, f"chat_history_{m}.json")
            if os.path.exists(history_file):
                os.remove(history_file)
