<div align="center">

# 🎙️ EchoMind — AI-Powered Video Assistant

**Ask anything about any YouTube video. Get instant, intelligent answers.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?logo=chainlink&logoColor=white)](https://langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📌 What is EchoMind?

EchoMind is a full-stack AI application that lets you **have a conversation with any YouTube video**. Paste a URL, and within moments you can ask the system questions about the video's content — from key takeaways and summaries to specific factual queries.

Under the hood it combines:
- **Speech-to-text** transcription via `faster-whisper`
- **Retrieval-Augmented Generation (RAG)** with `ChromaDB` + `HuggingFace` embeddings
- **LLM-powered answering** via Mistral AI through LangChain
- A clean **HTML/JS frontend** served by nginx, all wrapped in **Docker Compose**

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎬 **YouTube Ingestion** | Paste any YouTube URL — `yt-dlp` downloads and extracts the audio automatically |
| 🗣️ **Fast Transcription** | `faster-whisper` (CTranslate2 backend) transcribes audio accurately at high speed |
| 🧠 **RAG Q&A** | LangChain pipeline embeds the transcript into ChromaDB and answers questions with Mistral AI |
| 💬 **Session Management** | Each conversation session is persisted in `sessions.json` for continuity |
| 🐳 **One-command Deploy** | Full Docker Compose setup — backend + nginx frontend in two containers |
| 🔁 **Health Checks** | Backend exposes a `/health` endpoint; frontend waits for it before starting |
| 💾 **Model Caching** | HuggingFace and Whisper models are cached in named Docker volumes — downloaded once |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser / Client                      │
│                      http://localhost                        │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              nginx  (dockerfile.frontend)  :80               │
│         Serves static HTML/JS and proxies /api/* →          │
└────────────────────────────┬────────────────────────────────┘
                             │ reverse proxy
                             ▼
┌─────────────────────────────────────────────────────────────┐
│           FastAPI  (dockerfile.backend)  :8000               │
│                                                              │
│  ┌──────────────┐   ┌────────────────┐   ┌───────────────┐  │
│  │  yt-dlp      │   │ faster-whisper │   │  LangChain    │  │
│  │  Download    │──▶│  Transcribe    │──▶│  RAG Pipeline │  │
│  │  audio       │   │  audio → text  │   │  (Mistral AI) │  │
│  └──────────────┘   └────────────────┘   └──────┬────────┘  │
│                                                  │           │
│                                          ┌───────▼────────┐  │
│                                          │   ChromaDB     │  │
│                                          │  Vector Store  │  │
│                                          └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Ai-Powered-Video-Assistant/
├── Audio_Preprocessing/        # Audio extraction & chunking utilities
├── backend/                    # FastAPI application
│   ├── app.py                  # Main FastAPI app, routes, session logic
│   ├── audio_processor.py      # Whisper transcription wrapper
│   ├── llm_pipeline.py         # LangChain RAG chain (Mistral + ChromaDB)
│   └── main.py                 # Background task handlers
├── frontend/                   # Static HTML/CSS/JS UI
├── .dockerignore
├── .gitignore
├── docker-compose.yml          # Orchestrates backend + frontend services
├── dockerfile.backend          # Multi-stage Python 3.11 image (ffmpeg, git, curl)
├── dockerfile.frontend         # nginx image serving frontend
├── entrypoint.sh               # Container startup script
├── nginx.conf                  # nginx proxy configuration
└── requirements.txt            # All Python dependencies
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI + Uvicorn |
| **Audio Download** | yt-dlp |
| **Audio Conversion** | pydub + ffmpeg |
| **Transcription** | faster-whisper (CTranslate2) |
| **Embeddings** | sentence-transformers (HuggingFace) |
| **Vector Store** | ChromaDB |
| **LLM** | Mistral AI (via langchain-mistralai) |
| **Orchestration** | LangChain (LCEL — pipe `\|` syntax) |
| **Frontend** | HTML + CSS + Vanilla JS |
| **Reverse Proxy** | nginx |
| **Containerisation** | Docker + Docker Compose |

---

## 🚀 Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/) installed
- A **Mistral AI API key** — get one at [console.mistral.ai](https://console.mistral.ai/)

### 1. Clone the repository

```bash
git clone https://github.com/DeepanshuSharma1607/Ai-Powered-Video-Assistant.git
cd Ai-Powered-Video-Assistant
```

### 2. Create a `.env` file

```bash
cp .env.example .env   # or create it manually
```

Fill in the required values:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
WHISPER_MODEL=base          # tiny | base | small | medium | large-v3
```

### 3. Build and run

```bash
docker compose up --build
```

> First run downloads the Whisper and HuggingFace embedding models — this takes a few minutes. Subsequent runs use the cached volumes and start in seconds.

### 4. Open the app

```
http://localhost
```

---

## 🏃 Running Locally (Without Docker)

> Requires Python 3.11+, ffmpeg, and git installed.

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set your environment variables
export MISTRAL_API_KEY=your_key_here
export WHISPER_MODEL=base

# Start the FastAPI server
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Then open `frontend/index.html` in your browser (or serve it with any static file server).

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MISTRAL_API_KEY` | ✅ Yes | — | Mistral AI API key |
| `WHISPER_MODEL` | ❌ No | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check (used by Docker Compose) |
| `POST` | `/process` | Submit a YouTube URL to transcribe & index |
| `POST` | `/ask` | Ask a question about the processed video |
| `GET` | `/sessions` | List active sessions |

> Full interactive docs available at `http://localhost:8000/docs` (Swagger UI) when the backend is running.

---

## 🔬 How It Works

```
User pastes YouTube URL
        │
        ▼
  yt-dlp downloads audio
        │
        ▼
  pydub converts to WAV/MP3
        │
        ▼
  faster-whisper transcribes → plain text transcript
        │
        ▼
  LangChain splits text into chunks
        │
        ▼
  HuggingFace embeddings encode chunks → vectors
        │
        ▼
  ChromaDB stores vectors (persisted to disk)
        │
   User asks a question
        │
        ▼
  Query → embedded → similarity search in ChromaDB
        │
        ▼
  Top-k relevant chunks + question → Mistral AI prompt
        │
        ▼
  Mistral AI returns a grounded answer ✅
```

---

## 💡 Tips & Gotchas

- **Model warm-up:** The backend takes 30–60 seconds on first boot while models load. The health check handles this — the frontend only starts after the backend is ready.
- **Disk space:** Large Whisper models (`large-v3`) require ~3 GB. Use `base` or `small` on resource-constrained machines.
- **AWS Free Tier:** The `t2.micro` (1 GB RAM) can handle one transcription at a time. Add a swap file for stability. `t3.small` or better is recommended for multi-user use.
- **Video length:** Very long videos (>1 hour) may hit memory limits on low-RAM servers. Consider using `tiny` model or processing in chunks.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "Add: your feature"
git push origin feature/your-feature-name
# Open a Pull Request 🎉
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Built with by [Deepanshu Sharma](https://github.com/DeepanshuSharma1607)

⭐ If you found this useful, give it a star!

</div>
