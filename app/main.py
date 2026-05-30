import os
import sys
import base64

# Menambahkan parent directory ke sys.path agar modul 'app' dapat ditemukan
# ketika file dijalankan secara langsung sebagai `python app/main.py`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.stt import transcribe_speech_to_text
from app.llm import generate_response, reset_chat_history
from app.tts import transcribe_text_to_speech
from app.processor import build_augmented_prompt

app = FastAPI(title="Voice Chatbot API - UAS NLP Premium")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/voice-chat")
async def voice_chat(
    file: UploadFile = File(...),
    mode: str = Form(default="preserve")
):
    """
    Endpoint utama pipeline Voice Chatbot.
    Menerima file audio WAV + pilihan mode sistem.
    Mengembalikan JSON berisi transkripsi, metadata processing layer,
    teks respons LLM, dan audio respons dalam format Base64.
    """
    # Validasi mode
    valid_modes = {"preserve", "normalize", "translate"}
    if mode not in valid_modes:
        mode = "preserve"

    # 1. Baca bytes audio dari request
    file_bytes = await file.read()

    # 2. STT: Whisper transkripsi audio → teks
    raw_transcription = transcribe_speech_to_text(file_bytes, file_ext=".wav")
    print(f"\n[STT] {raw_transcription}")

    if raw_transcription.startswith("[ERROR]"):
        return JSONResponse(status_code=500, content={"error": raw_transcription})

    # 3. Processing Layer: normalisasi, deteksi bahasa, code-switching tagging
    augmented_prompt, metadata = build_augmented_prompt(raw_transcription, mode)
    print(f"[PROCESSOR] Mode={mode} | Dominan={metadata['language']['dominant']}")
    print(f"[PROCESSOR] Tagged: {metadata['tagged']}")
    print(f"[PROCESSOR] Normalized: {metadata['normalized']}")

    # 4. LLM: Kirim prompt yang sudah diproses ke Gemma/Gemini
    response_text = generate_response(augmented_prompt, mode=mode)
    print(f"[LLM] {response_text}")

    if response_text.startswith("[ERROR]"):
        return JSONResponse(status_code=500, content={"error": response_text})

    # 5. TTS: Konversi teks respons LLM → suara WAV
    output_audio_path = transcribe_text_to_speech(response_text)
    print(f"[TTS] Output: {output_audio_path}")

    if output_audio_path.startswith("[ERROR]"):
        return JSONResponse(status_code=500, content={"error": output_audio_path})

    # 6. Encode audio ke Base64 untuk dikembalikan dalam JSON
    try:
        with open(output_audio_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Gagal membaca audio: {e}"})

    return {
        # Data utama
        "transcription":      raw_transcription,
        "response":           response_text,
        "audio":              audio_base64,
        # Metadata processing layer
        "mode":               mode,
        "normalized":         metadata["normalized"],
        "tagged":             metadata["tagged"],
        "language_dominant":  metadata["language"]["dominant"],
        "language_ratios":    metadata["language"]["ratios"],
    }


@app.post("/reset-chat")
async def reset_chat(mode: str = Form(default=None)):
    """Reset riwayat percakapan untuk mode tertentu atau semua mode."""
    reset_chat_history(mode if mode else None)
    return {"status": "ok", "message": f"Riwayat chat '{mode or 'semua'}' berhasil direset."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
