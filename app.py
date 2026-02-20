import streamlit as st
from RAG import process_urls, generate_answer

try:
    from RAG import process_urls, generate_answer
except Exception as e:
    st.error(f"RAG import failed: {e}")
    st.stop()
    
# =========================
# Session state
# =========================
if "urls_processed" not in st.session_state:
    st.session_state.urls_processed = False

if "url1" not in st.session_state:
    st.session_state.url1 = ""
if "url2" not in st.session_state:
    st.session_state.url2 = ""
if "url3" not in st.session_state:
    st.session_state.url3 = ""


if "logs" not in st.session_state:
    st.session_state.logs = []

# =========================
# UI Layout
# =========================
st.set_page_config(page_title="RealEstate Research Tool", layout="wide")

st.title("🏢 RealEstate Research Tool")

col1, col2 = st.columns([1, 2])

# =========================
# LEFT PANEL — URL input
# =========================
with col1:
    st.subheader("News Article URLs")

    url1 = st.text_input("URL 1", key="url1")
    url2 = st.text_input("URL 2", key="url2")
    url3 = st.text_input("URL 3", key="url3")


    urls = [u for u in [url1, url2, url3] if u.strip()]

    if st.button("Process URLs"):
        if not urls:
            st.error("URL is not given")
        else:
            st.session_state.logs.clear()

            # live progress messages
            with st.spinner("Processing URLs..."):
                st.session_state.logs.append("Initialize components...")
                st.session_state.logs.append("Load data...")
                st.session_state.logs.append("Split text...")
                st.session_state.logs.append("Add docs to vector db...")

                process_urls(urls)

            st.session_state.urls_processed = True
            st.success("URLs processed successfully")

    if st.button("Remove URLs"):

        st.session_state.urls_processed = False

        # delete widget states (safe reset)
        for k in ["url1", "url2", "url3", "question_box"]:
            if k in st.session_state:
                del st.session_state[k]

        st.rerun()




    # show logs
    if st.session_state.logs:
        st.subheader("Processing Logs")
        for log in st.session_state.logs:
            st.write(log)

# =========================
# RIGHT PANEL — Q&A
# =========================
with col2:
    st.subheader("Ask Question")

    # Question input
    question = st.text_input("enter your question", key="question_box")

    if st.button("Ask"):

        # 1️⃣ URLs not processed
        if not st.session_state.get("urls_processed", False):
            st.error("Process the URL first")
            st.stop()

        # 2️⃣ No question
        if not question.strip():
            st.warning("Enter your question")
            st.stop()

        # 3️⃣ Valid → generate
        with st.spinner("Generating answer..."):
            answer, sources = generate_answer(question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        st.write(sources)

