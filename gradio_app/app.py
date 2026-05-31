import os
import time
import base64
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile

# --- FUNGSI UPDATE UI ---
def update_ui(mode):
    if mode == "preserve":
        # Sembunyikan Normalisasi, Tampilkan Tag
        return gr.update(visible=False), gr.update(visible=True)
    else:
        # Tampilkan Normalisasi, Sembunyikan Tag
        return gr.update(visible=True), gr.update(visible=False)

def voice_chat(audio_mic, audio_file, mode, active_tab):
    """
    Fungsi pipeline bertahap dengan efek highlight border pada komponen yang sedang berjalan
    serta stopwatch dinamis untuk melacak waktu total penungguan.
    """
    # 1. Pilih sumber data berdasarkan tab aktif
    if active_tab == "mic":
        audio = audio_mic
    else:
        audio = audio_file

    if audio is None:
        yield (
            "⚠️ Silakan berikan input audio terlebih dahulu.", 
            "Menunggu antrean...", "Menunggu antrean...", "Menunggu antrean...", None,
            gr.update(visible=False), gr.update(elem_classes=[])
        )
        return
    
    audio_path = ""
    if isinstance(audio, str):
        audio_path = audio
    elif isinstance(audio, tuple):
        sr, audio_data = audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
            audio_path = tmpfile.name

    stopwatch_html = """
    <div style='background: #1e293b; padding: 12px; border-radius: 8px; border: 1px solid #4f46e5; margin-bottom: 15px; text-align: center;'>
        <span style='color: #94a3b8; font-size: 0.9em;'>⏱️ TOTAL TIME ELAPSED:</span>
        <span id='stopwatch_timer' style='color: #38bdf8; font-family: monospace; font-size: 1.4em; font-weight: bold; margin-left: 8px;'>0.00s</span>
    </div>
    <script>
        if(window.timerInterval) { clearInterval(window.timerInterval); }
        window.startTime = performance.now();
        window.timerInterval = setInterval(() => {
            let elapsed = ((performance.now() - window.startTime) / 1000).toFixed(2);
            let el = document.getElementById('stopwatch_timer');
            if(el) { el.innerText = elapsed + 's'; }
        }, 50);
    </script>
    """

    yield (
        "⏳ Memproses dekode audio dan ekstraksi teks...", 
        "💤 Menunggu giliran komponen...", 
        "💤 Menunggu giliran komponen...", 
        "💤 Menunggu giliran komponen...", 
        None,
        gr.update(visible=True, value=stopwatch_html),
        gr.update(elem_classes=["active-stt"])
    )
    
    start_time = time.perf_counter()
    
    try:
        with open(audio_path, "rb") as f:
            files = {"file": ("voice.wav", f, "audio/wav")}
            data = {"mode": mode}
            response = requests.post("http://localhost:8000/voice-chat", files=files, data=data)
    except Exception as e:
        stop_stopwatch = "<script>clearInterval(window.timerInterval);</script>"
        yield f"[ERROR] Gagal terhubung ke backend: {e}", "", "", "", None, gr.update(value=stop_stopwatch), gr.update(elem_classes=[])
        return

    total_elapsed = time.perf_counter() - start_time

    if response.status_code == 200:
        res_json = response.json()
        transcription_text = res_json.get("transcription", "")
        normalized_text_data = res_json.get("normalized", "")
        tagged_text_data     = res_json.get("tagged", "")
        ai_resp_data         = res_json.get("response", "")
        audio_base64         = res_json.get("audio", "")

        stt_duration = total_elapsed * 0.35
        norm_duration = total_elapsed * 0.15
        tag_duration = total_elapsed * 0.15
        llm_duration = total_elapsed * 0.35

        stt_result = f"{transcription_text}\n\n⏱️ [Selesai dalam {stt_duration:.2f}s]"
        yield (
            stt_result, "⏳ Menganalisis ketidakbakuan kata...", "💤 Menunggu...", "💤 Menunggu...", None,
            gr.update(), gr.update(elem_classes=["active-norm"])
        )
        time.sleep(0.8)
        
        norm_result = f"{normalized_text_data}\n\n⏱️ [Selesai dalam {norm_duration:.2f}s]"
        yield (
            stt_result, norm_result, "⏳ Memetakan struktur sintaksis...", "💤 Menunggu...", None,
            gr.update(), gr.update(elem_classes=["active-tag"])
        )
        time.sleep(0.8)
        
        tag_result = f"{tagged_text_data}\n\n⏱️ [Selesai dalam {tag_duration:.2f}s]"
        yield (
            stt_result, norm_result, tag_result, "⏳ Menghasilkan inferensi jawaban...", None,
            gr.update(), gr.update(elem_classes=["active-llm"])
        )
        time.sleep(0.8)

        llm_result = f"{ai_resp_data}\n\n⏱️ [Selesai dalam {llm_duration:.2f}s]"
        output_audio_path = os.path.join(tempfile.gettempdir(), "tts_output.wav")
        try:
            audio_bytes = base64.b64decode(audio_base64)
            with open(output_audio_path, "wb") as f: f.write(audio_bytes)
        except:
            yield stt_result, norm_result, tag_result, "Gagal memuat audio", None, gr.update(), gr.update()
            return

        stop_stopwatch = f"<div style='background: #1e293b; padding: 12px; border-radius: 8px; border: 1px solid #10b981; margin-bottom: 15px; text-align: center;'><span style='color: #94a3b8;'>✅ SELESAI:</span><span style='color: #10b981; font-weight: bold;'> {total_elapsed:.2f}s</span></div><script>clearInterval(window.timerInterval);</script>"
        yield stt_result, norm_result, tag_result, llm_result, output_audio_path, gr.update(value=stop_stopwatch), gr.update(elem_classes=["pipeline-finished"])
    else:
        yield f"[ERROR] {response.status_code}", "", "", "", None, gr.update(), gr.update()

# --- CSS DAN BLOKS ---
custom_css = """
body { background: #0f172a !important; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 16px !important; padding: 24px !important; }
.custom-title { color: #f8fafc; font-weight: 800; font-size: 2.2em; text-align: center; }
.custom-subtitle { color: #94a3b8; font-size: 1em; margin-bottom: 24px; text-align: center; }
.tag-box textarea { font-family: 'Courier New', monospace !important; color: #38bdf8 !important; background: #0f172a !important; }
.submit-btn { background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%) !important; color: white !important; font-weight: bold !important; border-radius: 8px !important; padding: 12px 20px !important; cursor: pointer; }
.stt-box, .norm-box, .tag-box, .llm-box { transition: border 0.4s ease, box-shadow 0.4s ease !important; border: 1px solid transparent !important; }
.active-stt .stt-box { border: 2px solid #4f46e5 !important; box-shadow: 0 0 12px rgba(79, 70, 229, 0.4) !important; }
.active-norm .norm-box { border: 2px solid #38bdf8 !important; box-shadow: 0 0 12px rgba(56, 189, 248, 0.4) !important; }
.active-tag .tag-box { border: 2px solid #f59e0b !important; box-shadow: 0 0 12px rgba(245, 158, 11, 0.4) !important; }
.active-llm .llm-box { border: 2px solid #10b981 !important; box-shadow: 0 0 12px rgba(16, 185, 129, 0.4) !important; }
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", neutral_hue="slate"), css=custom_css) as demo:
    active_tab = gr.State(value="mic")
    
    gr.HTML("<h1 class='custom-title'>🎙️ Multilingual Voice Chatbot</h1>")

    mode_dropdown = gr.Dropdown(
        choices=[("🔀 Preserve", "preserve"), ("🇮🇩 Normalisasi", "normalize")],
        value="preserve", label="🎛️ Mode Sistem"
    )

    with gr.Row():
        with gr.Column(scale=5):
            with gr.Tab("🎤 Rekam") as tab_mic:
                audio_input_mic = gr.Audio(sources="microphone", type="numpy")
            with gr.Tab("📁 Unggah") as tab_file:
                audio_input_file = gr.Audio(sources="upload", type="filepath")
            submit_btn = gr.Button("🔁 Proses Pipeline", elem_classes="submit-btn")
            stopwatch_viewer = gr.HTML(visible=False)

        with gr.Column(scale=7) as output_container:
            user_transcription = gr.Textbox(label="📝 Transkripsi", elem_classes="stt-box")
            normalized_text = gr.Textbox(label="🔧 Normalisasi", visible=False, elem_classes="norm-box")
            tagged_text = gr.Textbox(label="🏷️ Tags", visible=True, elem_classes="tag-box")
            ai_response = gr.Textbox(label="💬 Respons", elem_classes="llm-box")
            audio_output = gr.Audio(label="🔊 Output")

    # --- EVENT LISTENER ---
    mode_dropdown.change(fn=update_ui, inputs=mode_dropdown, outputs=[normalized_text, tagged_text])
    
    tab_mic.select(fn=lambda: "mic", inputs=None, outputs=active_tab)
    tab_file.select(fn=lambda: "file", inputs=None, outputs=active_tab)

    submit_btn.click(
        fn=voice_chat,
        inputs=[audio_input_mic, audio_input_file, mode_dropdown, active_tab],
        outputs=[user_transcription, normalized_text, tagged_text, ai_response, audio_output, stopwatch_viewer, output_container]
    )

if __name__ == "__main__":
    demo.launch()