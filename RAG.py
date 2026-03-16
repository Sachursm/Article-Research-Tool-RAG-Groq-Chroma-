from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from uuid import uuid4
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_core.prompts import PromptTemplate

load_dotenv()

CHUNK_SIZE = 1000
COLLECTION_NAME = "real_estate_collection"
VECTORSTORE_DIR = Path(__file__).parent / "resources" / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["summaries", "question"],
    template="""
You are a helpful real-estate research assistant.

Use ONLY the information from the article below to answer the question.
If the answer is not present, say you don't know.

Article
=======
{summaries}

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
    loader = WebBaseLoader(web_paths=urls, header_template={"User-Agent": "Mozilla/5.0"})
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
    return llm, vector_store  # ✅ Return instead of storing in globals


def generate_answer(query, llm, vector_store):  # ✅ Accept as parameters
    """Generate answer using the provided llm and vector_store."""
    if vector_store is None:
        raise RuntimeError("Vector database is not initialized.")
    if llm is None:
        raise RuntimeError("LLM is not initialized.")

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )

    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")
    return result["answer"], sources