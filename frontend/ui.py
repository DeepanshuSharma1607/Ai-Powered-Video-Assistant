import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="MeetingMind",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0D1117; color: #C9D1D9; }

    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #21262D;
    }

    .card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        height: 100%;
    }
    .card-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #58A6FF;
        margin-bottom: 0.5rem;
    }
    .card-body {
        color: #C9D1D9;
        font-size: 0.88rem;
        line-height: 1.65;
        white-space: pre-wrap;
    }

    .meeting-banner {
        background: linear-gradient(90deg, #1F6FEB18, transparent);
        border-left: 3px solid #58A6FF;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1.1rem;
        margin-bottom: 1.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        color: #E6EDF3;
    }

    .badge-ready   { color:#3FB950; font-size:0.8rem; }
    .badge-waiting { color:#D29922; font-size:0.8rem; }

    .chat-bubble-user {
        background:#1F6FEB1A;
        border:1px solid #1F6FEB33;
        border-radius:8px;
        padding:0.55rem 0.9rem;
        margin:0.35rem 0;
        font-size:0.88rem;
        color:#C9D1D9;
    }
    .chat-bubble-bot {
        background:#161B22;
        border:1px solid #21262D;
        border-radius:8px;
        padding:0.55rem 0.9rem;
        margin:0.35rem 0;
        font-size:0.88rem;
        color:#C9D1D9;
    }
    .chat-role { font-size:0.7rem; font-weight:700; letter-spacing:0.08em;
                 text-transform:uppercase; margin-bottom:3px; }
    .role-you  { color:#58A6FF; }
    .role-bot  { color:#3FB950; }

    /* tighten Streamlit default padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="stFileUploader"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────
for k, v in {"results": None, "chat": [], "ready": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── helpers ───────────────────────────────────────────────────────────────────
def api(endpoint, *, method="POST", files=None, data=None, json=None, timeout=360):
    url = f"{API_BASE}{endpoint}"
    try:
        fn = {"POST": requests.post, "DELETE": requests.delete}[method]
        r = fn(url, files=files, data=data, json=json, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the backend. Is `uvicorn app:app` running?"
    except requests.exceptions.HTTPError as e:
        try:    msg = e.response.json().get("detail", str(e))
        except: msg = str(e)
        return None, msg
    except Exception as e:
        return None, str(e)


def show_results(r: dict):
    if r.get("title"):
        st.markdown(f'<div class="meeting-banner">📌 {r["title"]}</div>',
                    unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    def card(col, icon, label, content):
        col.markdown(
            f'<div class="card">'
            f'<div class="card-label">{icon} {label}</div>'
            f'<div class="card-body">{content or "—"}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    card(col1, "📋", "Summary",      r.get("summary", ""))
    card(col2, "🔑", "Key Decisions", r.get("decisions", ""))
    card(col1, "✅", "Action Items",  r.get("action_items", ""))
    card(col2, "❓", "Open Questions", r.get("questions", ""))


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    language = st.selectbox("Transcription language",
                            ["english", "hinglish"],
                            help="Hinglish routes through Sarvam AI")
    append   = st.toggle("Append to previous session", False,
                         help="Merge with already-loaded transcript")

    st.divider()
    badge = ('<span class="badge-ready">● Session active</span>'
             if st.session_state.ready else
             '<span class="badge-waiting">● No meeting loaded</span>')
    st.markdown(f"**Status** &nbsp; {badge}", unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Reset session", use_container_width=True):
        _, err = api("/reset", method="DELETE", timeout=10)
        if err:
            st.error(err)
        else:
            st.session_state.update(results=None, chat=[], ready=False)
            st.success("Session cleared.")
            st.rerun()

    st.markdown("<br><small style='color:#484F58'>Backend: localhost:8000</small>",
                unsafe_allow_html=True)


# ── main ──────────────────────────────────────────────────────────────────────
st.markdown("## 🎙️ MeetingMind")
st.markdown("<p style='color:#8B949E;margin-top:-0.4rem'>"
            "Upload a recording or a text transcript and get structured insights instantly."
            "</p>", unsafe_allow_html=True)

st.divider()

tab_av, tab_txt = st.tabs(["🎵  Audio / Video", "📄  Transcript (txt / md)"])

# ── audio tab ─────────────────────────────────────────────────────────────────
with tab_av:
    st.markdown("<br>", unsafe_allow_html=True)
    audio_file = st.file_uploader(
        "Drop your recording here",
        type=["mp3","wav","m4a","aac","ogg","flac","mp4","mov","mkv","avi","webm"],
        key="audio_up"
    )
    if audio_file:
        st.markdown(f"<small style='color:#3FB950'>✓ {audio_file.name} "
                    f"· {audio_file.size // 1024} KB</small>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ Process Recording", type="primary",
                 disabled=not audio_file, key="btn_audio"):
        with st.spinner("Transcribing and analysing — may take a few minutes for large files…"):
            res, err = api(
                "/process",
                files={"file": (audio_file.name, audio_file.getvalue(), audio_file.type)},
                data={"language": language, "append_previous": str(append).lower()}
            )
        if err:
            st.error(err)
        else:
            st.session_state.update(results=res, ready=True, chat=[])
            st.rerun()

# ── transcript tab ────────────────────────────────────────────────────────────
with tab_txt:
    st.markdown("<br>", unsafe_allow_html=True)
    txt_file = st.file_uploader(
        "Drop your transcript file here",
        type=["txt", "md"],
        key="txt_up"
    )
    if txt_file:
        with st.expander("Preview", expanded=False):
            st.text(txt_file.getvalue().decode("utf-8", errors="replace")[:1200] + "…")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ Process Transcript", type="primary",
                 disabled=not txt_file, key="btn_txt"):
        with st.spinner("Analysing transcript…"):
            res, err = api(
                "/transcript",
                files={"file": (txt_file.name, txt_file.getvalue(), "text/plain")},
                data={"language": language, "append_previous": str(append).lower()}
            )
        if err:
            st.error(err)
        else:
            st.session_state.update(results=res, ready=True, chat=[])
            st.rerun()


# ── results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    st.divider()
    st.markdown("### Analysis")
    show_results(st.session_state.results)


# ── chat ──────────────────────────────────────────────────────────────────────
if st.session_state.ready:
    st.divider()
    st.markdown("### 💬 Ask about the meeting")

    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-bubble-user">'
                f'<div class="chat-role role-you">You</div>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-bubble-bot">'
                f'<div class="chat-role role-bot">Assistant</div>{msg["content"]}</div>',
                unsafe_allow_html=True
            )

    question = st.chat_input("Ask anything about this meeting…")
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.spinner("Thinking…"):
            res, err = api("/question", json={"question": question})
        if err:
            st.error(err)
        else:
            st.session_state.chat.append(
                {"role": "assistant", "content": res.get("answer", "No answer.")}
            )
        st.rerun()