import os
import time
import csv
import glob
import editdistance
import shutil
from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech  # Tambahan untuk evaluasi TTS
from app.processor import build_augmented_prompt

# Konfigurasi
AUDIO_DIR = "data/corpus/audio"
TRANSCRIPT_DIR = "data/corpus/transcripts"
EVAL_AUDIO_DIR = "data/evaluation_output/audio_results" # Folder untuk audio evaluasi
CSV_OUTPUT = "analisis_hasil.csv"

# Pastikan folder output ada
os.makedirs(EVAL_AUDIO_DIR, exist_ok=True)

def sanitize_for_csv(text):
    """Membersihkan teks dari karakter yang merusak format CSV"""
    return str(text).replace('\n', ' ').replace('\r', '').replace('"', '""')

def get_processed_files():
    processed = set()
    if os.path.exists(CSV_OUTPUT):
        with open(CSV_OUTPUT, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='|')
            next(reader, None)
            for row in reader:
                if row: processed.add(row[0])
    return processed

def calculate_wer(reference, hypothesis):
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    dist = editdistance.eval(ref_words, hyp_words)
    return float(dist) / max(len(ref_words), 1)

def calculate_cer(reference, hypothesis):
    ref = reference.lower()
    hyp = hypothesis.lower()
    dist = editdistance.eval(ref, hyp)
    return float(dist) / max(len(ref), 1)

def run_analysis():
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    processed_files = get_processed_files()
    
    print(f"Total file ditemukan: {len(audio_files)}.")
    
    with open(CSV_OUTPUT, "a", newline="", encoding="utf-8", buffering=1) as f:
        writer = csv.writer(f, delimiter='|')
        
        # Header baru termasuk path_audio_eval
        if os.path.exists(CSV_OUTPUT) and os.path.getsize(CSV_OUTPUT) == 0:
            writer.writerow([
                "filename_audio", "filename_transcript", "transkripsi", "ground_truth", 
                "wer", "cer", "hasil_llm", "status", "latency", "path_audio_eval",
                "correctness_1_5", "naturalness_1_5", "intelligibility_1_3"
            ])

        for file_path in audio_files:
            filename = os.path.basename(file_path)
            if filename in processed_files: continue
            
            start_time = time.time()
            
            # --- Logika Pencarian GT ---
            clean_name = filename.split('_', 1)[1] if '_' in filename else filename
            base = os.path.splitext(clean_name)[0]
            kandidat = [base]
            if "audio" in base and base.replace("audio", "").isdigit():
                num = base.replace("audio", "")
                kandidat.append(f"audio0{num}" if len(num)==1 else f"audio{num[1:]}")
            
            gt_path, gt_name = None, "N/A"
            for k in kandidat:
                path = os.path.join(TRANSCRIPT_DIR, f"{k}.txt")
                if os.path.exists(path):
                    gt_path, gt_name = path, f"{k}.txt"
                    break
            
            ground_truth = "N/A"
            if gt_path:
                with open(gt_path, "r", encoding="utf-8") as gt_f:
                    ground_truth = gt_f.read().strip()

            # --- Pemrosesan Pipeline ---
            try:
                with open(file_path, "rb") as af:
                    transcription = transcribe_speech_to_text(af.read())
                
                prompt, _ = build_augmented_prompt(transcription, mode="normalize")
                llm_response = generate_response(prompt, mode="normalize")
                
                # TTS & Penyimpanan Audio Evaluasi
                tts_temp_path = transcribe_text_to_speech(llm_response)
                final_audio_path = os.path.join(EVAL_AUDIO_DIR, filename)
                
                if not tts_temp_path.startswith("[ERROR]"):
                    shutil.copy(tts_temp_path, final_audio_path)
                else:
                    final_audio_path = "TTS_ERROR"
                
                wer = calculate_wer(ground_truth, transcription) if ground_truth != "N/A" else 1.0
                cer = calculate_cer(ground_truth, transcription) if ground_truth != "N/A" else 1.0
                latency = round(time.time() - start_time, 2)
                
                writer.writerow([
                    filename, gt_name, transcription, ground_truth, 
                    round(wer, 4), round(cer, 4), sanitize_for_csv(llm_response), 
                    "SUCCESS", latency, final_audio_path, "", "", ""
                ])
                print(f"Selesai: {filename} | WER: {round(wer, 2)}s")
            
            except Exception as e:
                latency = round(time.time() - start_time, 2)
                writer.writerow([filename, gt_name, "ERROR", ground_truth, 1.0, 1.0, str(e), "FAILED", latency, "N/A", "", "", ""])
                print(f"Gagal: {filename}")

    print("Pipeline analisis selesai.")

if __name__ == "__main__":
    run_analysis()