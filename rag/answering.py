# rag/answering.py
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from rag.config import CUSTOM_PROMPT, SUMMARY_PROMPT, SUMMARY_KEYWORDS


def generate_summary(docs, llm):
    context = "\n\n".join(
        doc.page_content
        for doc in docs[:15]
    )
    formatted_prompt = SUMMARY_PROMPT.format(context=context)
    response = llm.invoke(formatted_prompt)
    return response.content
    
def generate_answer(query, llm, vector_store, docs):
    """
    Generate answer or summary from article.
    """

    if vector_store is None:
        raise RuntimeError("Vector database is not initialized.")

    if llm is None:
        raise RuntimeError("LLM is not initialized.")

    # Detect summary requests
    is_summary = any(
        word in query.lower()
        for word in SUMMARY_KEYWORDS
    )

    # SUMMARY FLOW
    if is_summary:

        summary = generate_summary(
            docs,
            llm
        )

        return summary, "Summary generated from full article"

    # NORMAL RAG QA FLOW
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20}
        ),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )

    result = chain.invoke({"query": query})

    answer = result["result"]

    source_docs = result.get("source_documents", [])

    sources = ", ".join(
        set(
            doc.metadata.get("source", "")
            for doc in source_docs
            if doc.metadata.get("source")
        )
    )

    return answer, sources


# 3 new functions for multi-document comparison
def get_retriever_for_source(vector_store, source_name: str):
    # your code here

def generate_per_source_answer(query, llm, vector_store, all_sources: list):
    # your code here

def compare_sources(query, llm, vector_store, all_sources: list):
    # your code here