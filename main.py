from dotenv import load_dotenv
from audio_preprocessor import process_input
from transcriber import transcribe_all
from llm_pipeline import generate_title, summarize, extract_action_items, extract_key_decisions, extract_questions
from vector_store import build_rag_chain, ask_questions
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

load_dotenv()

def run_pipeline(
    source: Optional[str] = None,
    transcript: Optional[str] = None,
    language: str = "english"
) -> dict:

    if transcript is None:

        if source is None:
            raise ValueError(
                "Either source or transcript must be provided."
            )

        chunks = process_input(source)

        transcript = transcribe_all(
            chunks,
            language=language
        )

    print(
        f"Raw Transcription (first 300 chars):\n{transcript[:300]}"
    )

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_title = executor.submit(generate_title, transcript)
        future_summary = executor.submit(summarize, transcript)
        future_actions = executor.submit(extract_action_items, transcript)
        future_decisions = executor.submit(extract_key_decisions, transcript)
        future_questions = executor.submit(extract_questions, transcript)

        title = future_title.result()
        summary = future_summary.result()
        action_items = future_actions.result()
        decisions = future_decisions.result()
        questions = future_questions.result()

    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "decisions": decisions,
        "questions": questions,
        "rag_chain": rag_chain
    }

if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['decisions']}")
    print(f"\n❓ Open Questions:\n{result['questions']}")
    print("=" * 60)

    print("\nChat with your meeting (type 'exit' to quit)\n")
    rag_chain = result['rag_chain']

    while True:
        questions = input("You: ").strip()
        if questions.lower() in ['exit', 'quit', 'q']:
            print("Goodbye...")
            break
        if not questions:
            continue
        answer = ask_questions(rag_chain, questions)
        print(f"\nAssistant: {answer}\n")