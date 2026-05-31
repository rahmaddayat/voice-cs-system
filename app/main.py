import os
import sys
import base64
import json
import re
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.stt import transcribe_speech_to_text
from app.llm import generate_response, reset_chat_history
from app.tts import transcribe_text_to_speech
from app.processor import build_augmented_prompt

app = FastAPI(title="Voice Chatbot API - UAS NLP Premium")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/voice-chat")
async def voice_chat(
    file: UploadFile = File(...),
    mode: str = Form(default="preserve")
):
    # 1. Validasi Mode
    valid_modes = {"preserve", "normalize", "translate"}
    mode = mode if mode in valid_modes else "preserve"
    
    # 2. Proses STT (Transkripsi)
    file_bytes = await file.read()
    raw_transcription = transcribe_speech_to_text(file_bytes)
    
    # 3. Build Prompt & Metadata
    prompt, metadata = build_augmented_prompt(raw_transcription, mode)
    normalized_base = metadata.get("normalized", raw_transcription)
    
    # 4. LLM: Generate Response
    response_text = generate_response(prompt, mode=mode)
    
    # 5. Parsing Hasil LLM (Menggunakan JSON karena lebih aman)
    # Catatan: Pastikan di llm.py atau processor.py Anda memberikan instruksi format JSON
    try:
        clean_json = response_text.replace("```json", "").replace("```", "").strip()
        data_json = json.loads(clean_json)
        perbaikan_teks = data_json.get("perbaikan", normalized_base)
        tagged_teks = data_json.get("tagging", "Tagging tidak tersedia")
        respon_asisten = data_json.get("respon", response_text)
    except:
        # Fallback jika LLM tidak mengirim JSON
        perbaikan_teks = normalized_base
        tagged_teks = "N/A"
        respon_asisten = response_text

    # 6. TTS: Konversi teks ke audio
    output_audio_path = transcribe_text_to_speech(respon_asisten)
    if output_audio_path.startswith("[ERROR]"):
        return JSONResponse(status_code=500, content={"error": output_audio_path})
    
    # --- ARSIP EVALUASI (TAMBAHAN) ---
    # Menyimpan salinan audio ke folder evaluasi agar mudah didengar untuk penilaian manual
    eval_folder = "data/evaluation_output/audio_results"
    os.makedirs(eval_folder, exist_ok=True)
    eval_audio_name = f"{file.filename.replace('.wav', '')}_result.wav"
    save_path = os.path.join(eval_folder, eval_audio_name)
    shutil.copy(output_audio_path, save_path)
    # ---------------------------------
        
    # 7. Encode audio ke Base64
    try:
        with open(output_audio_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Gagal membaca audio: {e}"})
        
    # 8. Return JSON Final
    return {
        "transcription": raw_transcription,
        "response": respon_asisten,
        "audio": audio_base64,
        "mode": mode,
        "normalized": perbaikan_teks,
        "tagged": tagged_teks,
        "language_dominant": metadata["language"]["dominant"],
        "language_ratios": metadata["language"]["ratios"],
    }
@app.post("/reset-chat")
async def reset_chat(mode: str = Form(default=None)):
    """Reset riwayat percakapan untuk mode tertentu atau semua mode."""
    reset_chat_history(mode if mode else None)
    return {"status": "ok", "message": f"Riwayat chat '{mode or 'semua'}' berhasil direset."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)