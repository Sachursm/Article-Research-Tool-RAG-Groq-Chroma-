from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_experimental.text_splitter import SemanticChunker
import re

def extract_content_types(doc: Document):
    content = doc.page_content
    source  = doc.metadata.get("source", "")
    code_docs = []   
    table_docs = []

    # Step 1 — code blocks
    code_pattern = r"```[\s\S]*?```"
    code_blocks = re.findall(code_pattern, content)
    clean_text = re.sub(code_pattern, "[CODE BLOCK]", content)

    for block in code_blocks:
        code_docs.append(Document(          # ← 4 spaces
            page_content=block,
            metadata={"source": source, "type": "code"}
        ))

    # Step 2 — tables
    table_pattern = r"(\|.+\|\n)+"
    table_blocks = re.findall(table_pattern, content)
    clean_text = re.sub(table_pattern, "[TABLE]", clean_text)

    for table in table_blocks:
        table_docs.append(Document(         # ← 4 spaces
            page_content=table,
            metadata={"source": source, "type": "table"}
        ))

    # Step 3 — remaining text
    text_doc = Document(
        page_content=clean_text,
        metadata={"source": source, "type": "text"}
    )

    return code_docs, table_docs, text_doc





def chunk_documents(documents: list, embeddings) -> list:
    result = []

    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=500,
        chunk_overlap=50
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=500,
        chunk_overlap=100
    )

    for doc in documents:
        # Step 1 — separate content types
        code_docs, table_docs, text_doc = extract_content_types(doc)

        # Step 2 — text → SemanticChunker
        try:
            text_chunks = semantic_splitter.split_documents([text_doc])
            if not text_chunks:
                raise ValueError("Empty")
        except Exception:
            text_chunks = fallback_splitter.split_documents([text_doc])
        result.extend(text_chunks)

        # Step 3 — code → language aware splitter
        if code_docs:
            code_chunks = code_splitter.split_documents(code_docs)
            result.extend(code_chunks)

        # Step 4 — tables → no splitting, add directly
        result.extend(table_docs)

    return result
