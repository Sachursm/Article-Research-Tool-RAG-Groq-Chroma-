from pathlib import Path
from langchain_core.prompts import PromptTemplate


# Constants
CHUNK_SIZE = 500
COLLECTION_NAME = "article_research_collection"
VECTORSTORE_DIR = Path(__file__).parent.parent/ "resources" / "vectorstore"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

#custom prompt for answer generation
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful research assistant analyzing content from articles and videos.
Use ONLY the information from the context below to answer.
If the answer is not present, say you don't know.

Structure your response exactly like this:

**Answer**
[Direct one line answer]

**Explanation**
[Detailed explanation from the context]

**Evidence**
[Exact quote from the context supporting the answer]

Context:
=======
{context}

Question: {question}
"""
)

#specialized prompt for summarization tasks
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
You are a helpful article research assistant.
Look carefully through ALL the context provided.
If the question asks for a full form or abbreviation, look carefully for the expanded term in the context.
Provide a clear and structured summary
of the article below.

Include:
1. Main topic
2. Key points
3. Important insights
4. Final takeaway

Article:
{context}

Summary:
"""
)

SUMMARY_KEYWORDS = [
    "summary",
    "summarize",
    "overview",
    "main points",
    "key points",
]
