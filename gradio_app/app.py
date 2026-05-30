import os
import uuid
import base64
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile

# ──────────────────────────────────────────────────────────────────────
# Label tampilan untuk masing-masing mode
# ──────────────────────────────────────────────────────────────────────
MODE_LABELS = {
    "preserve":  "🔀 Preserve Code-Switching",
    "normalize": "🇮🇩 Normalisasi Bahasa Indonesia",
    "translate": "🌐 Terjemahan ke Bahasa Indonesia",
}
MODE_DESCRIPTIONS = {
    "preserve":  "Asisten merespons dalam campuran bahasa yang sama seperti pengguna (code-switching dipertahankan).",
    "normalize": "Asisten selalu merespons dalam Bahasa Indonesia baku, apapun bahasa input-nya.",
    "translate": "Asisten menampilkan terjemahan input lalu menjawab sepenuhnya dalam Bahasa Indonesia.",
}
LANG_EMOJI = {"id": "🇮🇩", "en": "🇺🇸", "ar": "🇸🇦", "mixed": "🌐"}
LANG_LABEL = {"id": "Indonesia", "en": "Inggris", "ar": "Arab", "mixed": "Campuran"}


def voice_chat(audio, mode: str):
    if audio is None:
        return (
            "Silakan rekam suara Anda terlebih dahulu.", "", "", "",
            None, "—", "—"
        )

    sr, audio_data = audio

    # Simpan input audio dari numpy → file WAV temporer
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
        audio_path = tmpfile.name

    # Kirim ke backend FastAPI dengan mode yang dipilih
    try:
        with open(audio_path, "rb") as f:
            files = {"file": ("voice.wav", f, "audio/wav")}
            data  = {"mode": mode}
            response = requests.post(
                "http://localhost:8000/voice-chat",
                files=files,
                data=data
            )
    except Exception as e:
        return (
            "[ERROR] Gagal menghubungi backend.",
            f"Pastikan backend FastAPI sudah berjalan di port 8000.\nDetail: {e}",
            "", "", None, "—", "—"
        )

    if response.status_code == 200:
        res = response.json()
        if "error" in res:
            return res["error"], "Terjadi kesalahan.", "", "", None, "—", "—"

        # Parsing respons JSON yang kaya
        transcription    = res.get("transcription", "")
        response_text    = res.get("response", "")
        normalized       = res.get("normalized", "")
        tagged           = res.get("tagged", "")
        dominant         = res.get("language_dominant", "mixed")
        ratios           = res.get("language_ratios", {})
        audio_b64        = res.get("audio", "")

        # Format label bahasa dengan emoji dan persentase
        lang_label = LANG_EMOJI.get(dominant, "🌐") + " " + LANG_LABEL.get(dominant, "Campuran")
        ratio_text = (
            f"🇮🇩 ID: {int(ratios.get('id',0)*100)}% | "
            f"🇺🇸 EN: {int(ratios.get('en',0)*100)}% | "
            f"🇸🇦 AR: {int(ratios.get('ar',0)*100)}%"
        )

        # Simpan audio Base64 → file WAV temporer untuk diputar
        output_audio_path = os.path.join(
            tempfile.gettempdir(), f"tts_output_{uuid.uuid4()}.wav"
        )
        try:
            audio_bytes = base64.b64decode(audio_b64)
            with open(output_audio_path, "wb") as f:
                f.write(audio_bytes)
        except Exception as e:
            return transcription, f"Gagal decode audio: {e}", normalized, tagged, None, lang_label, ratio_text

        return transcription, response_text, normalized, tagged, output_audio_path, lang_label, ratio_text

    else:
        return (
            "Gagal melakukan transkripsi.",
            f"Server kode status: {response.status_code}",
            "", "", None, "—", "—"
        )


def reset_chat(mode: str):
    """Memanggil endpoint reset chat pada backend."""
    try:
        requests.post(
            "http://localhost:8000/reset-chat",
            data={"mode": mode}
        )
        return f"✅ Riwayat chat mode '{MODE_LABELS.get(mode, mode)}' berhasil direset."
    except Exception as e:
        return f"❌ Gagal reset: {e}"


# ──────────────────────────────────────────────────────────────────────
# CSS Premium — Glassmorphism Dark Theme
# ──────────────────────────────────────────────────────────────────────
custom_css = """
body {
    background: radial-gradient(ellipse at top, #0f1a35 0%, #060b18 100%) !important;
    min-height: 100vh;
}
.gradio-container {
    max-width: 1100px !important;
    margin: 36px auto !important;
    background: rgba(15, 23, 42, 0.75) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 28px !important;
    box-shadow: 0 30px 60px -10px rgba(0,0,0,0.6) !important;
    padding: 36px !important;
}
.premium-header {
    text-align: center;
    padding: 10px 0 24px 0;
}
.premium-title {
    background: linear-gradient(135deg, #c084fc 0%, #818cf8 45%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
    font-size: 2.6em;
    letter-spacing: -0.03em;
    margin: 0;
}
.premium-subtitle {
    color: #94a3b8;
    font-size: 1.05em;
    margin-top: 8px;
    font-weight: 400;
}
.mode-desc {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(129,140,248,0.25);
    border-radius: 12px;
    padding: 10px 16px;
    color: #a5b4fc;
    font-size: 0.92em;
    margin: 6px 0 14px 0;
}
.section-title {
    color: #e2e8f0;
    font-weight: 700;
    font-size: 1.05em;
    margin-bottom: 6px;
}
.tag-box {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 0.85em;
    color: #7dd3fc;
    background: rgba(14, 165, 233, 0.08);
    border: 1px solid rgba(14,165,233,0.2);
    border-radius: 10px;
    padding: 8px 12px;
}
.lang-badge {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 8px;
    color: #c4b5fd;
    font-weight: 600;
    font-size: 0.95em;
    padding: 4px 12px;
    margin-bottom: 4px;
}
.submit-btn-glow {
    background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.45) !important;
    transition: all 0.25s ease !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    border-radius: 14px !important;
}
.submit-btn-glow:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.65) !important;
}
.reset-btn {
    background: rgba(239,68,68,0.15) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 12px !important;
    font-size: 0.9em !important;
    transition: all 0.2s ease !important;
}
.reset-btn:hover {
    background: rgba(239,68,68,0.3) !important;
    border-color: rgba(239,68,68,0.6) !important;
}
.footer-bar {
    text-align: center;
    color: #475569;
    font-size: 0.87em;
    margin-top: 28px;
    padding-top: 18px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Outfit"),
)

# ──────────────────────────────────────────────────────────────────────
# Bangun UI
# ──────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=theme, css=custom_css) as demo:

    # Header
    gr.HTML("""
    <div class="premium-header">
        <h1 class="premium-title">🎙️ Multilingual Voice Chatbot</h1>
        <p class="premium-subtitle">
            End-to-end pipeline: Speech-to-Text → Processing Layer → LLM → Text-to-Speech<br>
            Mendukung percakapan <i>code-switching</i> Bahasa Indonesia · Inggris · Arab
        </p>
    </div>
    """)

    # ── Baris Atas: Kontrol Mode ──
    with gr.Row():
        mode_dropdown = gr.Dropdown(
            choices=[
                ("🔀 Preserve Code-Switching", "preserve"),
                ("🇮🇩 Normalisasi Bahasa Indonesia", "normalize"),
                ("🌐 Terjemahan ke Bahasa Indonesia", "translate"),
            ],
            value="preserve",
            label="🎛️ Mode Sistem",
            info="Pilih cara asisten merespons input code-switching Anda.",
            interactive=True,
        )
        mode_desc_box = gr.Markdown(
            value=f"<div class='mode-desc'>{MODE_DESCRIPTIONS['preserve']}</div>"
        )

    # Update deskripsi saat mode berubah
    def update_mode_desc(mode):
        return f"<div class='mode-desc'>{MODE_DESCRIPTIONS.get(mode, '')}</div>"
    mode_dropdown.change(fn=update_mode_desc, inputs=mode_dropdown, outputs=mode_desc_box)

    gr.HTML("<hr style='border-color:rgba(255,255,255,0.07);margin:8px 0 18px 0;'>")

    # ── Baris Tengah: Input & Output Utama ──
    with gr.Row(equal_height=False):

        # Kolom Kiri: Input suara
        with gr.Column(scale=5):
            gr.HTML("<div class='section-title'>🎤 Input Suara</div>")
            audio_input = gr.Audio(
                sources="microphone",
                type="numpy",
                label="Rekam suara Anda"
            )
            submit_btn = gr.Button("🔁 Kirim & Proses", elem_classes="submit-btn-glow")
            reset_btn  = gr.Button("🗑️ Reset Percakapan", elem_classes="reset-btn")
            reset_status = gr.Markdown(value="")

        # Kolom Kanan: Output pipeline
        with gr.Column(scale=6):
            gr.HTML("<div class='section-title'>🤖 Hasil Pipeline NLP</div>")

            with gr.Row():
                lang_badge  = gr.Markdown(value="<div class='lang-badge'>Bahasa: —</div>")
                ratio_label = gr.Markdown(value="")

            user_transcription = gr.Textbox(
                label="📝 Transkripsi Suara (Whisper STT)",
                placeholder="Teks hasil transkripsi akan muncul di sini...",
                interactive=False, lines=2
            )
            normalized_text = gr.Textbox(
                label="🔧 Normalisasi Transkrip (Processing Layer)",
                placeholder="Teks setelah normalisasi filler word & simbol...",
                interactive=False, lines=2
            )
            tagged_text = gr.Textbox(
                label="🏷️ Code-Switching Tags (Processing Layer)",
                placeholder="[ID:kata] [EN:word] [AR:كلمة] ...",
                interactive=False, lines=2,
                elem_classes="tag-box"
            )
            ai_response = gr.Textbox(
                label="💬 Respons Asisten (LLM)",
                placeholder="Jawaban dari asisten AI akan tampil di sini...",
                interactive=False, lines=3
            )
            audio_output = gr.Audio(
                type="filepath",
                label="🔊 Balasan Suara (Coqui TTS)",
                interactive=False
            )

    # ── Event Handlers ──
    submit_btn.click(
        fn=voice_chat,
        inputs=[audio_input, mode_dropdown],
        outputs=[
            user_transcription, ai_response,
            normalized_text, tagged_text,
            audio_output,
            lang_badge, ratio_label
        ]
    )

    reset_btn.click(
        fn=reset_chat,
        inputs=[mode_dropdown],
        outputs=[reset_status]
    )

    # Footer
    gr.HTML("""
    <div class="footer-bar">
        UAS Praktikum Pemrosesan Bahasa Alami &nbsp;|&nbsp;
        STT: whisper.cpp &nbsp;·&nbsp; LLM: Gemma 4 &nbsp;·&nbsp; TTS: Coqui Indonesian
    </div>
    """)

demo.launch()
