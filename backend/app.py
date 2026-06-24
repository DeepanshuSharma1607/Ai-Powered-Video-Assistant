import os
import re
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from Audio_Preprocessing.main import run_pipeline, user_ids, save_session
from Audio_Preprocessing.vector_store import ask_questions, get_embeddings
from Audio_Preprocessing.transcriber import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_INDEX = os.path.join(BASE_DIR, "frontend", "index.html")

ALLOWED_AV_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".txt", ".md"
}
TRANSCRIPT_EXTENSIONS = {".txt", ".md"}


ALLOWED_ORIGINS = ["http://localhost:5500", "http://127.0.0.1:5500"]


def _safe_token(value: str, max_len: int = 64) -> str:
    """Strip path separators and anything non-alphanumeric so user-supplied
    strings can't be used for path traversal when building filesystem paths."""
    value = os.path.basename(value or "")
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return value[:max_len] or "file"


class Youtube_Ext(BaseModel):
    url: str
    user_id: str
    append: bool = False


class Questions(BaseModel):
    user_id: str
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up models...")
    load_model()
    get_embeddings()
    print("Models ready.")
    yield
    print("Shutting down.")


app = FastAPI(title="Audio Transcriber API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """serve the single-page frontend"""
    return FileResponse(FRONTEND_INDEX)


@app.get("/health")
async def health():
    return {"status": "ok", "message": "The app is working fine"}


@app.post("/process_text_audio")
async def process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    append_previous: bool = Form(False),
    user_id: str = Form("default"),
):
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in ALLOWED_AV_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    if ext in TRANSCRIPT_EXTENSIONS:
        raw = await file.read()
        text = raw.decode("utf-8", errors="ignore")
        user = run_pipeline(
            source=None,
            transcript=text,
            user_id=user_id,
            append=append_previous,
            language="english",
        )
        return {"summary": user["summary"]} 

    safe_user = _safe_token(user_id)
    safe_filename = _safe_token(file.filename)
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"{safe_user}_{uuid.uuid4().hex}_{safe_filename}",
    )
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        user = run_pipeline(
            source=tmp_path,
            transcript=None,
            user_id=user_id,
            append=append_previous,
            language="english",
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"summary": user["summary"]}


@app.post("/youtube_process")
def youtube_process(data: Youtube_Ext):
    user = run_pipeline(
        source=data.url,
        transcript=None,
        user_id=data.user_id,
        append=data.append,
        language="english",
    )
    return {"summary": user["summary"]}


@app.post("/question")
async def ask_question(data: Questions):
    try:
        answer = ask_questions(data.user_id, data.question)
        return {"question": data.question, "answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset")
async def reset_session(user: str):
    existed = user_ids.pop(user, None) is not None
    save_session()
    return {"message": f"Session for '{user}' reset." if existed else f"No session found for '{user}'."}