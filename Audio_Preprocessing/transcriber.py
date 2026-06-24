import os
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
import threading
from dotenv import load_dotenv
load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")

_model = None
_whisper_lock = threading.Lock()

def load_model():
    global _model
    if _model is None:
        with _whisper_lock:
            if _model is None:
                print(f"Loading Whisper model: {WHISPER_MODEL}...")

                _model = WhisperModel(
                    WHISPER_MODEL,
                    device="cpu",
                    compute_type="int8"
                )

        print("Whisper model loaded.")

    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()

    segments, _ = model.transcribe(
        chunk_path,
        language = "en",
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500}
    )

    return " ".join(
        segment.text for segment in segments
    )

def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    return transcribe_chunk_whisper(chunk_path)

def transcribe_all(chunks: list, language: str = "english") -> str:
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            results = list(executor.map(lambda c: transcribe_chunk(c, language), chunks))

        print("Transcription complete.")
        return " ".join(results).strip()
    finally:
        for c in chunks:
            if os.path.exists(c): os.remove(c)

if __name__ == "__main__":
    pass


'''
for 49 min normmal audio it took 2 min 50sec
for yt video it takes https://youtu.be/7go0C30ia-M?si=x5CamPBn6l7RFB8w 2 min 50sec

'''