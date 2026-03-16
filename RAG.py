from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from uuid import uuid4

load_dotenv()

# Constants
CHUNK_SIZE = 1000
COLLECTION_NAME = "real_estate_collection"
VECTORSTORE_DIR = Path(__file__).parent / "resources" / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful real-estate research assistant.

Use ONLY the information from the article below to answer the question.
If the answer is not present, say you don't know.

Article
=======
{context}

Question: {question}

Answer:
"""
)


def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)


def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )


def process_urls(urls):
    """
    Scrape web pages, split into chunks, store in Chroma DB.
    Returns the initialized (llm, vector_store) tuple for session storage.
    """
    print("Initializing components...")
    llm = get_llm()
    vector_store = get_vector_store()

    try:
        vector_store.reset_collection()
    except Exception:
        pass

    print("Loading data...")
    loader = WebBaseLoader(
        web_paths=urls,
        header_template={"User-Agent": "Mozilla/5.0"}
    )
    data = loader.load()

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100,
    )
    docs = text_splitter.split_documents(data)

    print("Adding docs to vector DB...")
    uuids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=uuids)

    print(f"Stored {len(docs)} chunks")
    return llm, vector_store


def generate_answer(query, llm, vector_store):
    """
    Generate answer using RetrievalQA chain with custom prompt.
    Returns answer string and comma-separated source URLs.
    """
    if vector_store is None:
        raise RuntimeError("Vector database is not initialized.")
    if llm is None:
        raise RuntimeError("LLM is not initialized.")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )

    result = chain.invoke({"query": query})
    answer = result["result"]

    # Extract unique source URLs from returned documents
    source_docs = result.get("source_documents", [])
    sources = ", ".join(
        set(
            doc.metadata.get("source", "")
            for doc in source_docs
            if doc.metadata.get("source")
        )
    )

    return answer, sources


if __name__ == "__main__":
    pass