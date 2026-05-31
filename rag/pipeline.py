from uuid import uuid4
from langchain_groq import ChatGroq
from rag.vectorstore import get_embeddings, get_vector_store
from rag.chunking import chunk_documents
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

def process_data(document: list, session_id: str):
    print("Initializing components...")
    llm = get_llm()
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings, session_id)

    try:
        vector_store.reset_collection()
    except Exception:
        pass

    print("Chunking documents...")
    all_docs = chunk_documents(document, embeddings)  # ← new

    print(f"Adding {len(all_docs)} chunks to vector DB...")
    uuids = [str(uuid4()) for _ in all_docs]
    vector_store.add_documents(all_docs, ids=uuids)

    print(f"✅ Stored {len(all_docs)} chunks")
    return llm, vector_store, all_docs
