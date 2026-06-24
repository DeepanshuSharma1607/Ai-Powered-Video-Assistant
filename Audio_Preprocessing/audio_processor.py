import os
import yt_dlp
import time
import uuid
from pydub import AudioSegment

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url: str, session_dir) -> str:
    output_path = os.path.join(session_dir, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename

def convert_to_wav(input_path: str , out_dir : str) -> str:
    filename = os.path.splitext(os.path.basename(input_path))[0] + ".wav"
    output_path = os.path.join(out_dir, filename)

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")

    return output_path

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    # The full-length wav has now been split into chunks; the original
    # is no longer needed and was previously left on disk forever.
    if os.path.exists(wav_path):
        os.remove(wav_path)

    return chunks

def process_input(source: str , session_id : str = None) -> list:
    session_id = session_id or uuid.uuid4().hex
    session_dir = os.path.join(DOWNLOAD_DIR, session_id)
    os.makedirs(session_dir , exist_ok = True)
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source , session_dir)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source , session_dir)
    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks

if __name__ == "__main__":
    start = time.time()
    chunks = process_input(r"C:\Users\Deepanshu sharma\Desktop\Audio_Transcriber\Audio_Preprocessing\nnn.mp3")
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(chunks)

'''
so for yt video of 49 min it took 22 sec
and same video in ordinary audio it took 18 sec
'''


