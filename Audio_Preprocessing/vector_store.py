from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_pipeline import get_llm
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import threading

CHROMA_DIR = "vector_database"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert Meeting Assistant.

Your primary responsibility is to answer questions using ONLY the meeting transcript context.

Rules:

1. First determine whether the answer exists in the provided transcript context.

2. If the answer exists:

   * Answer only from the transcript.
   * Be concise and accurate.
   * Mention speaker names if available.

3. If the answer does NOT exist in the transcript:

   * Start your response with:

     "⚠️ This information was not found in the meeting transcript."

   * Then provide a brief general answer from your own knowledge.

   * Limit this answer to 50-100 words.

   * Clearly indicate that it is external knowledge and not derived from the meeting.

4. Never pretend that transcript information exists when it does not.

5. If retrieval confidence appears low, treat the question as out-of-context.

Transcript Context:
{context}
""",
    ),
    ("human", "{question}"),
])
_rag_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
_chain_cache : dict[str, object] = {}
_chain_lock = threading.Lock()
_vs_cache : dict[str , object ] = {}

_embeddings = None
_embed_lock = threading.Lock()

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _embed_lock:
            if _embeddings is None:
                print("[RAG] Loading embedding model...")
                _embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs = {"batch_size":32}
                )
    return _embeddings

def format_docs(docs : list[Document]) ->str:
    return "\n\n".join([doc.page_content for doc in docs])

def _make_chain(vector_store : Chroma):
    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k":4}
    )

    return (
        {
            "context":retriever | RunnableLambda(format_docs), # sawwal se related chunks dhundho
            "question":RunnablePassthrough(),
        } | _RAG_PROMPT | get_llm() | StrOutputParser()
    )


def build_vector_store(transcript: str , user : str = None) -> Chroma:
    print("Building vector store...")
    print("user : ",user)

    chunks = _rag_splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i , "source":"transcript"})
        for i, chunk in enumerate(chunks)
    ]

    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=user,
        persist_directory=CHROMA_DIR
    )

def load_vector_store(user : str) -> Chroma:
    if user in _vs_cache:
        return _vs_cache[user]
    vs = Chroma(
        collection_name=user,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR
    )

    _vs_cache[user] = vs
    return vs


def get_retriever(user : str = None):
    return load_vector_store(user).as_retriever(search_type="mmr", search_kwargs={"k": 4})

def add_to_vectorstore(new_transcript : str , user : str) ->Chroma :
    """
    Embed ONLY the newly appended meeting and add it to the
    session's existing collection — no re-embedding old content.
    """
    chunks = _rag_splitter.split_text(new_transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i, "source": "transcript"})
        for i, chunk in enumerate(chunks)
    ]

    vs = load_vector_store(user)
    vs.add_documents(docs)

    return vs

def build_rag_chain(transcript: str , user : str = None , append : bool = False):
    if append:
        vs = add_to_vectorstore(transcript  , user)
        chain = _make_chain(vs)
        _chain_cache[user] = chain
        return chain
    
    if user in _chain_cache:
        return _chain_cache[user]
    
    vs = build_vector_store(transcript , user)
    
    chain = _make_chain(vs)
    _chain_cache[user] = chain
    return chain

def load_rag_chain(user:str=None):
    if user in _chain_cache:
        return _chain_cache[user]
    
    vs = load_vector_store(user)
    chain = _make_chain(vs)
    _chain_cache[user] = chain
    return chain

def ask_questions(user :str, question: str) -> str:
    chain = load_rag_chain(user)
    return chain.invoke(question)

if __name__ == "__main__":
    pass


'''
3 min 10sec for creating everything
and fro asking answer it took 12sec

'''