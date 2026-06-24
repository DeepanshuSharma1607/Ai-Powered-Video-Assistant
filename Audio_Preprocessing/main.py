from concurrent.futures import ThreadPoolExecutor
from .vector_store import build_rag_chain, get_embeddings
from .transcriber import transcribe_all, load_model
from .audio_processor import process_input
from .llm_pipeline import summarize_title
from dotenv import load_dotenv
from typing import Optional
import threading
import shutil
import json
import os

load_dotenv()

SESSION_FILES = "sessions.json"
_session_lock = threading.Lock()  # guards user_ids dict + sessions.json writes

def save_session():
    to_save={
        k:{x : v[x] for x in ('user_id' , "summary" , "transcript")}
        for k , v in user_ids.items()
    }
    with open(SESSION_FILES , "w") as f:
        json.dump(to_save,f)

def load_session():
    if os.path.exists(SESSION_FILES):
        with open(SESSION_FILES) as f:
            return json.load(f)
    return {}

user_ids = load_session()

def warm_up():
    """
    pre-load heavy models in the main thread before user input.
    """

    print("warming up models...")
    load_model()
    get_embeddings()
    print("Models ready ...")


def preprocessing(
    source: Optional[str] = None,
    transcript: Optional[str] = None,
    language : str = "english"
):
    if transcript is None:
        if source is None:
            raise ValueError(
                "Either source or transcript must be provided."
            )

        chunks = process_input(source)
        session_dir = os.path.dirname(chunks[0]) if chunks else None

        try:
            transcript = transcribe_all(
                chunks,
                language=language
            )
        finally:
            # transcribe_all already removes the chunk files; this removes
            # the per-session download folder itself (and anything else
            # left behind, e.g. a failed/partial download).
            if session_dir and os.path.isdir(session_dir):
                shutil.rmtree(session_dir, ignore_errors=True)

    return transcript


def run_pipeline(
    source: Optional[str] = None,
    transcript: Optional[str] = None,
    user_id : str=None,
    append = False,
    language: str = "english") -> dict:

    if user_id in user_ids and not append:
        return user_ids[user_id]

    if append:
        if user_id not in user_ids:
            raise ValueError(
                f"Session '{user_id}' does not exist."
            )

        # new_transcript = ONLY the freshly transcribed audio/text.
        # This is what gets embedded into the vector store — never the
        # combined history, otherwise old content gets re-embedded and
        # duplicated every time someone appends.
        new_transcript = preprocessing(
            source,
            transcript,
            language
        )

        old_transcript = user_ids[user_id]["transcript"]
        full_transcript = old_transcript + "\n\n" + new_transcript

    else:
        new_transcript = preprocessing(
            source,
            transcript,
            language
        )
        full_transcript = new_transcript

    print(
        f"Raw Transcription (first 300 chars):\n{new_transcript[:300]}"
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Summary always needs the FULL history so the meeting recap stays complete.
        title_summary_future = executor.submit(summarize_title, full_transcript)
        # Vector store only ever gets the NEW chunk — old chunks are already in Chroma.
        rag_future = executor.submit(build_rag_chain, new_transcript, user_id, append)

    title_summary = title_summary_future.result()
    rag_chain = rag_future.result()

    with _session_lock:
        user_ids[user_id]={
            "user_id" : user_id,
            "summary": title_summary,
            "transcript": full_transcript,
        }
        save_session()
    return user_ids[user_id]

if __name__=="__main__":
    pass

'''
1hr and 10min video took 8min with 1 question


'''