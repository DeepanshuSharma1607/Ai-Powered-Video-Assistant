from fastapi import FastAPI, UploadFile, File, HTTPException, Form , Body
from pydantic import BaseModel
from Audio_Preprocessing.main import run_pipeline
from Audio_Preprocessing.vector_store import build_rag_chain, ask_questions
from Audio_Preprocessing.transcriber import transcribe_all
from Audio_Preprocessing.llm_pipeline import split_transcript
from backend.database_ import load_transcript , save_transcript , cleanup_temp_files , UPLOAD_DIR , TRANSCRIPT_DIR , TRANSCRIPT_FILE
import os
import shutil

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
async def health():
    return {"status": "working..."}


ALLOWED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm"
}

rag_chain_store = None

@app.post("/process")
async def process_meeting(
    file: UploadFile = File(...),
    language: str = Form("english"),
    append_previous : bool = Form(False)
):
    global rag_chain_store
    try:
        ext = os.path.splitext(file.filename)[1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}"
            )

        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        result = run_pipeline(
            source=file_path,
            transcript=None,
            language=language
        )

        save_transcript(
            result["transcript"],append_previous
        )

        full_transcript = load_transcript()

        result = run_pipeline(
            source=None,
            transcript=full_transcript,
            language=language
        )

        rag_chain_store = result["rag_chain"]

        cleanup_temp_files()

        return {
            "title": result["title"],
            "summary": result["summary"],
            "action_items": result["action_items"],
            "decisions": result["decisions"],
            "questions": result["questions"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/transcript")
async def process_transcript(
    file: UploadFile = File(...),
    language: str = Form("english"),
    append_previous : bool = Form(False)):

    global rag_chain_store
    try:
        ext = os.path.splitext(file.filename)[1].lower()

        if ext not in [".txt",".md"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}"
            )
        
        transcript_text = (
            await file.read()
        ).decode("utf-8")

        save_transcript(
            transcript_text,
            append_previous
        )

        full_transcript = load_transcript()

        result = run_pipeline(
            source=None,
            transcript=full_transcript,
            language=language
        )       
        rag_chain_store = result["rag_chain"]

        cleanup_temp_files()

        return {
            "title": result["title"],
            "summary": result["summary"],
            "action_items": result["action_items"],
            "decisions": result["decisions"],
            "questions": result["questions"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/question")
async def ask_question(data: QuestionRequest):
    
    try:
        if rag_chain_store is None:
            raise HTTPException(
                status_code=400,
                detail="Please process a meeting first."
            )

        answer = ask_questions(
            rag_chain_store,
            data.question
        )

        return {
            "question": data.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@app.delete("/reset")
async def reset_session():
    global rag_chain_store

    rag_chain_store = None

    if os.path.exists(TRANSCRIPT_FILE):
        os.remove(TRANSCRIPT_FILE)

    return {
        "message": "All previous transcripts removed."
    }