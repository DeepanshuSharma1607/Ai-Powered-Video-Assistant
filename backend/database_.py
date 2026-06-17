import os
import shutil
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

TRANSCRIPT_DIR = "storage/transcripts"
TRANSCRIPT_FILE = os.path.join(
    TRANSCRIPT_DIR,
    "master.txt"
)
os.makedirs(TRANSCRIPT_DIR,exist_ok=True)

rag_chain_store = None

def cleanup_temp_files():
    folders = [
        "uploads",
        "audio",
        "temp",
        "downloads"
    ]

    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)


def save_transcript(transcript : str, append_previous : bool = False):
    
    if append_previous and os.path.exists(TRANSCRIPT_FILE):

        with open(TRANSCRIPT_FILE,"a",encoding = "utf-8") as f:

            f.write("\n\n")
            f.write(transcript)
    else:

        with open(TRANSCRIPT_FILE,"w",encoding = "utf-8") as f:
            f.write(transcript)


def load_transcript():
    if not os.path.exists(TRANSCRIPT_FILE):
        return ""
    
    with open(TRANSCRIPT_FILE, "r",encoding = "utf-8") as f:
        return f.read()