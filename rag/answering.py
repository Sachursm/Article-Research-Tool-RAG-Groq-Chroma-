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
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
            "filter": {"source": source_name}
        }
    )

def generate_per_source_answer(query, llm, vector_store, all_sources: list):
    answers = {}
    for source in all_sources:
        # Step 1 — get retriever for this source only
        retriever = get_retriever_for_source(vector_store, source)

        # Step 2 — create chain with that retriever
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": CUSTOM_PROMPT}
        )

        # Step 3 — invoke chain and store answer
        result = chain.invoke({"query": query})
        answers[source] = result["result"]

    return answers  # dict: {source_name: answer}

def compare_sources(query, llm, vector_store, all_sources: list):
    
    # Step 1 — get answer from each source
    per_source = generate_per_source_answer(
        query, llm, vector_store, all_sources
    )

    # Step 2 — build comparison context
    comparison_context = ""
    for i, (source, answer) in enumerate(per_source.items(), 1):
        comparison_context += f"Source {i} ({source}):\n{answer}\n\n"

    # Step 3 — build comparison prompt
    comparison_prompt = f"""
You are a research assistant comparing multiple sources.
Below are answers from different sources on the same question.

{comparison_context}

Question: {query}

Compare these sources and structure your response as:

**Points they agree on:**
[list common points]

**Points they disagree on:**
[list differences]

**Unique insights:**
[what each source says that others don't]
"""

    # Step 4 — invoke LLM with comparison prompt
    response = llm.invoke(comparison_prompt)
    return response.content