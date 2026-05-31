from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.config import COLLECTION_NAME, VECTORSTORE_DIR, EMBEDDING_MODEL

#embeddings created separately
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_vector_store(embeddings, session_id):  # receives embeddings and session_id as parameters
    return Chroma(collection_name = f"{COLLECTION_NAME}_{session_id}",
                  persist_directory = str(VECTORSTORE_DIR),
                  embedding_function=embeddings)