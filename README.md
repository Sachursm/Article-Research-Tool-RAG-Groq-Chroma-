# 🏢 RealEstate Research Tool

> A Retrieval-Augmented Generation (RAG) web application that ingests real-estate news articles from URLs and enables grounded question-answering using Groq's Llama-3.3-70B and a local Chroma vector database — all through a clean Streamlit interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama--3.3--70B-orange?style=flat-square)
![ChromaDB](https://img.shields.io/badge/Chroma-VectorDB-purple?style=flat-square)

---

## ✨ Features

- 📄 **URL Ingestion** — Load real-estate news articles directly from URLs
- 🔎 **Semantic Search** — Retrieve relevant context using Chroma vector database
- 🧠 **Grounded Answers** — Context-aware responses powered by Groq Llama-3.3-70B
- 📚 **Source Citations** — Every answer includes source references
- 🖥️ **Interactive UI** — Clean and responsive Streamlit interface
- 💾 **Persistent Vector Store** — Local Chroma DB persists across sessions
- 🚫 **No Paid Embedding APIs** — Uses local Sentence-Transformers MiniLM for embeddings

---

## 🧱 Architecture

```
URLs Provided by User
        │
        ▼
  Web Article Loader
        │
        ▼
  Recursive Text Splitter
        │
        ▼
  Sentence-Transformer Embeddings (MiniLM)
        │
        ▼
  Chroma Vector Database  ◄────────────────────┐
        │                                       │
        ▼                                       │
  User Question ──► Semantic Retriever ─────────┘
                          │
                          ▼
                   Prompt Template
                          │
                          ▼
               Groq LLM (Llama-3.3-70B)
                          │
                          ▼
              Answer + Source Citations
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq — Llama-3.3-70B |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Vector Database** | ChromaDB (local, persistent) |
| **RAG Framework** | LangChain |
| **UI** | Streamlit |
| **Language** | Python 3.10+ |

---

## 📂 Project Structure

```
RAG_based_Real_estate_webapplication/
│
├── app.py                  # Streamlit UI & session management
├── RAG.py                  # RAG pipeline, retriever & QA chain
├── resources/
│   └── vectorstore/        # Persistent Chroma DB (gitignored)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Sachursm/RAG_based_Real_estate_webapplication.git
cd RAG_based_Real_estate_webapplication
```

### 2. Create & Activate a Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your Groq API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free API key at [console.groq.com](https://console.groq.com)

### 5. Run the App

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 📖 How to Use

1. **Enter URLs** — Paste one or more real-estate news article URLs into the sidebar
2. **Process URLs** — Click the "Process URLs" button to ingest and embed the articles
3. **Ask Questions** — Type your question in the main input field
4. **Get Answers** — Receive grounded answers with source citations from the ingested articles

---

## 🧠 Example Questions

```
How do Federal Reserve policies affect mortgage rates?
Why did mortgage rates rise despite the Fed cutting rates?
What factors are currently driving housing affordability issues?
Which cities are seeing the highest real estate price growth?
What is the outlook for commercial real estate in 2025?
```

---

## 📌 Key Implementation Details

- **Recursive Text Chunking** — Splits articles intelligently to preserve semantic context
- **Custom RAG Prompt** — Engineered to minimize hallucination and enforce source grounding
- **Persistent Chroma Store** — Vector embeddings survive app restarts
- **Streamlit Session State** — Manages ingestion pipeline state cleanly across reruns
- **Local Embeddings** — MiniLM runs fully on-device, no embedding API costs

---

## 🔮 Roadmap

- [ ] Multi-article conversational memory
- [ ] PDF & document upload support
- [ ] Reranking layer for higher retrieval accuracy
- [ ] Deployment to Streamlit Cloud
- [ ] Support for additional LLM providers

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Sachu Retna SM**

[![GitHub](https://img.shields.io/badge/GitHub-Sachursm-181717?style=flat-square&logo=github)](https://github.com/Sachursm)

---

<p align="center">Built with ❤️ using LangChain, Groq, ChromaDB & Streamlit</p>
