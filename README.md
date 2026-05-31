# Link Video Presentasi :
- https://drive.google.com/drive/folders/1IA_JmLD6lj9bTyKUMGjILpX-DwPEaQYe?usp=sharing

# Voice Speech-to-Speech Code-Switching System (ID-EN-AR)

Sistem *Speech-to-Speech* (S2S) berbasis web yang dirancang untuk menangani fenomena pencampuran bahasa (*code-switching*) yang mengintegrasikan tiga bahasa sekaligus: **Bahasa Indonesia (ID), Bahasa Inggris (EN), dan Bahasa Arab (AR)**. Proyek ini dibangun sebagai komponen utama dalam pemenuhan UAS Praktikum Pemrosesan Bahasa Alami (NLP).

Sistem ini mengintegrasikan komponen *Speech-to-Text* (STT) lokal, penalaran berbasis *Large Language Model* (LLM) melalui API, dan komponen *Text-to-Speech* (TTS) ke dalam satu kesatuan *pipeline* multimodal yang stabil.

---

## 🚀 Fitur Utama

* **Multimodal Pipeline (End-to-End):** Pemrosesan langsung dari input suara pengguna hingga menghasilkan respons suara balik secara sinkron melalui protokol HTTP.
* **Dua Mode Operasional Luaran:**
    * **Mode *Preserve*:** Mempertahankan preferensi dan gaya bahasa campuran asal penutur (ID-EN-AR) dalam teks respons.
    * **Mode *ID Normalisasi*:** Mengonversi struktur kalimat acak/campuran menjadi Bahasa Indonesia formal yang baku sebelum diproses oleh komponen suara.
* **Language Tagging & Preprocessing:** Pemetaan otomatis elemen bahasa per segmen kata (misal: `[AR:uridu]`, `[EN:arrange]`) guna meningkatkan kualitas pelafalan fonem.
* **Antarmuka Interaktif:** Aplikasi web front-end yang responsif dan intuitif menggunakan *Gradio Framework*.
* **Rate Limit & Resource Management:** Implementasi *asynchronous handling* dan otomatisasi pembersihan berkas audio sementara untuk efisiensi penyimpanan server.

---

## 🛠️ Arsitektur & Teknologi Stack

* **Sisi Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
* **Sisi Frontend:** [Gradio](https://gradio.app/)
* **Speech-to-Text (STT):** [OpenAI Whisper-Large](https://github.com/openai/whisper) (Inferensi Lokal)
* **Cerebral/LLM Core:** [Gemma-4-31B-It](https://ai.google.dev/gemma) (via API Cloud dengan skema proteksi token)
* **Text-to-Speech (TTS):** [Coqui TTS](https://github.com/coqui-ai/TTS)

---

## 📦 Panduan Instalasi dan Setup

### 1. Kloning Repositori
```bash
git clone [https://github.com/rahmaddayat/voice-cs-system.git](https://github.com/rahmaddayat/voice-cs-system.git)
cd voice-cs-system