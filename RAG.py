from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from uuid import uuid4
from langchain_classic.chains import RetrievalQAWithSourcesChain




load_dotenv()

# constants
CHUNK_SIZE = 1000
COLLECTION_NAME = "real_estate_collection"
VECTORSTORE_DIR = Path(__file__).parent / "resources" / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

llm = None
vector_store = None


def initialize_components():

    """
    Initialize and cache the LLM and vector store.

    This function ensures that:
    - Groq LLM is created only once
    - Chroma vector database is created/loaded only once
    - Local embedding model is attached to the vector store

    Uses global variables so multiple calls reuse the same objects.
    """

    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )


def process_urls(urls):
    """
    Scrape web pages, split them into chunks, and store in Chroma DB.

    Steps:
    1. Ensure LLM + vector DB initialized
    2. Reset collection (optional fresh ingestion)
    3. Load web content from URLs
    4. Split text into semantic chunks
    5. Store chunks in vector database with unique IDs

    :param urls: list[str] - web URLs to ingest
    """

    print("Initialize components...")
    initialize_components()

    # reset safely
    try:
        vector_store.reset_collection()
    except Exception:
        pass

    print("Load data...")
    loader = WebBaseLoader(
        web_paths=urls,
        header_template={
            "User-Agent": "Mozilla/5.0"
        }
    )
    data = loader.load()

    print("Split text...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100,
    )
    docs = text_splitter.split_documents(data)

    print("Add docs to vector db...")
    uuids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=uuids)

    print(f"Stored {len(docs)} chunks")

def generate_answer(query):
    """
    Create RetrievalQA chain using Groq LLM and Chroma retriever.
    """
    if not vector_store:
        raise RuntimeError('vector database is not initialized')
     
    chain = RetrievalQAWithSourcesChain.from_llm(llm = llm,
             retriever = vector_store.as_retriever())

    result = chain.invoke({'question': query}, return_only_outputs = True)
    sources = result.get("sources", "")
    return result['answer'], sources


if __name__ == "__main__":
    pass
    # urls = [
    #     "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html"
    # ]
    # process_urls(urls)
    
    # # results = vector_store.similarity_search(
    # # "How do Federal Reserve interest rates affect mortgages?",
    # # k=3
    # # )

    # # for doc in results:
    # #     print(doc.page_content[:300])
    # #     print("-----")
    
    # answer, sources = generate_answer("How do Federal Reserve interest rates affect mortgages? ")
    # print(f"Answer: {answer}")
    # print(f"Sources: {sources}")