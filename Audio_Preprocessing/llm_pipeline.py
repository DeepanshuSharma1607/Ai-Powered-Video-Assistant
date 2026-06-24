import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
import threading
from dotenv import load_dotenv
load_dotenv()                 

_llm = None
_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
_llm_lock = threading.Lock() 

def get_llm():
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                print("Loading Mistral llm ...")
                _llm = ChatMistralAI(model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.5)
    return _llm

def build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}")])
    return {"text": RunnablePassthrough()} | prompt | get_llm() | StrOutputParser()

def summarize_title(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely"),
        ("human", "{text}")
    ])

    map_chain = map_prompt | llm | StrOutputParser()
    chunks = _splitter.split_text(transcript)

    chunk_summaries = map_chain.batch(chunks , config = {"max_concurrency" : 5})

    summarization =  build_chain(
        "You are an expert meeting summarizer. Combine these partial summaries "
        "into a final professional meeting summary in bullet points"
    ).invoke("\n\n".join(chunk_summaries))

    title = build_chain("""Based on the meeting transcript,
                        generate a short professional
                        meeting title(max 8 words).
                        Only return the title, nothing else""").invoke(summarization)

    return title + "\n\n" + summarization

if __name__ == "__main__":
    pass

'''
it took 3min and 50 sec 

'''