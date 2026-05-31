from rag.loaders import scrape_urls, extract_pdf, extract_txt
from rag.pipeline import process_data
from rag.answering import generate_answer, generate_per_source_answer, compare_sources
from rag.config import VECTORSTORE_DIR