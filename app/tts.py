import os
import re
import uuid
import tempfile
import subprocess
import numpy as np
from num2words import num2words
from g2p_id import G2P

# ====================================================================
# TRICK PATCH: Memaksa ONNX Runtime menerima int32/int64 di Windows
# ====================================================================
from onnxruntime import InferenceSession
old_run = InferenceSession.run
def new_run(self, output_names, input_feed, run_options=None):
    clean_feed = {}
    for k, v in input_feed.items():
        if isinstance(v, np.ndarray) and v.dtype == np.int32:
            clean_feed[k] = v.astype(np.int64)
        else:
            clean_feed[k] = v
    return old_run(self, output_names, clean_feed, run_options)
InferenceSession.run = new_run

# Inisialisasi pengubah grafem-ke-fonem Indonesia
g2p = G2P()
# ====================================================================

def normalize_numbers(text: str) -> str:
    """
    Konversi semua angka dalam teks ke bentuk kata Bahasa Indonesia
    agar Coqui TTS dapat membacanya dengan benar.
    Contoh: '2025' -> 'dua ribu dua puluh lima'
    """
    def replace_num(match):
        raw = match.group()
        try:
            cleaned = raw.replace(',', '.')
            if '.' in cleaned:
                return num2words(float(cleaned), lang='id')
            else:
                return num2words(int(raw), lang='id')
        except Exception:
            return raw
    return re.sub(r'\b\d+[.,]?\d*\b', replace_num, text)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path ke folder utilitas TTS sesuai template baru Anda
COQUI_DIR = os.path.join(BASE_DIR, "coqui_utils")

# TODO: Lengkapi jalur path ke file model TTS
COQUI_MODEL_PATH = os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth")
COQUI_CONFIG_PATH = os.path.join(COQUI_DIR, "config.json")
COQUI_SPEAKER = "wibowo"

def transcribe_text_to_speech(text: str) -> str:
    """
    Fungsi untuk mengonversi teks menjadi suara menggunakan TTS engine yang ditentukan.
    Args:
        text (str): Teks yang akan diubah menjadi suara.
    Returns:
        str: Path ke file audio hasil konversi.
    """
    # Mengembalikan nilai path hasil dari fungsi _tts_with_coqui
    path = _tts_with_coqui(text)
    return path

# === ENGINE 1: Coqui TTS ===
def _tts_with_coqui(text: str) -> str:
    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4()}.wav")

    print("Mengonversi angka ke kata...")
    text_normalized = normalize_numbers(text)
    print(f"Teks setelah normalisasi angka: {text_normalized}")

    print("Mengonversi teks menjadi format fonemik...")
    try:
        phoneme_text = g2p(text_normalized)
        print(f"Hasil fonem: {phoneme_text}")
    except Exception as e:
        print(f"[ERROR] Gagal mengonversi fonem: {e}")
        return "[ERROR] Failed to convert phoneme"

    # Definisikan jalan ke file speakers.pth di dalam folder coqui_utils sesuai modul Anda
    COQUI_SPEAKERS_PATH = os.path.join(COQUI_DIR, "speakers.pth")

    # Jalankan Coqui TTS dengan subprocess
    cmd = [
        "tts",
        "--text", phoneme_text,
        "--model_path", COQUI_MODEL_PATH,
        "--config_path", COQUI_CONFIG_PATH,
        "--speakers_file_path", COQUI_SPEAKERS_PATH,
        "--speaker_idx", COQUI_SPEAKER,
        "--out_path", output_path
    ]
    
    try:
        print("Menjalankan Coqui TTS untuk sintesis suara...")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] TTS subprocess failed: {e}")
        return "[ERROR] Failed to synthesize speech"

    return output_path

if __name__ == "__main__":
    test_text = "Halo Rahmad, tes pengujian sistem teks ke suara telah berhasil dilakukan."
    print(f"Memulai pengujian TTS dengan teks: '{test_text}'")
    
    hasil_audio = transcribe_text_to_speech(test_text)
    
    if "[ERROR]" not in hasil_audio:
        print(f"\n[SUKSES] TTS berjalan dengan baik!")
        print(f"File audio sementara Anda sukses dibuat di: {hasil_audio}")
    else:
        print(f"\n[GAGAL] TTS masih mengalami kendala teknis.")