# 🧠 DeepRAG — RAG Pipeline Built From Scratch

A production-grade Retrieval-Augmented Generation (RAG) pipeline built **entirely from scratch** — no LangChain, no shortcuts. Every component is implemented manually to deeply understand how RAG works under the hood.

---

## 🎯 Why From Scratch?

Most RAG tutorials use frameworks like LangChain which hide the complexity. This project builds every component manually so you understand **exactly** what is happening at each step — before jumping to any framework.

---

## 🚀 Features

- 📄 **PDF Extraction** — custom extractor using `pdfplumber`
- ✂️ **Text Chunking** — fixed character chunker with configurable overlap
- 🔢 **Embeddings** — manual embedding pipeline using `sentence-transformers`
- 🗄️ **Vector Search** — FAISS index (flat + IVF) built and managed manually
- 💬 **Generation** — Ollama + Mistral with custom prompt engineering
- 📊 **3-Level Evaluation System:**
  - Level 1: Exact match + keyword match
  - Level 2: Cosine similarity + ROUGE-1, ROUGE-2, ROUGE-L
  - Level 3: LLM-as-judge (faithfulness, relevance, completeness)

---

## 🏗️ Project Structure

```
DeepRAG/
├── docs/
│   └── your_file.pdf          ← put your PDF here
├── rag/
│   ├── __init__.py
│   ├── pdf_extractor.py       ← extracts text from PDF page by page
│   ├── chunker.py             ← splits text into overlapping chunks
│   ├── embedder.py            ← converts chunks to vectors
│   ├── vector_store.py        ← FAISS index (build, save, search)
│   ├── generator.py           ← prompt builder + Ollama generation
│   └── pipeline.py            ← orchestrates all steps with caching
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py           ← runs all 3 evaluation levels
│   ├── metrics.py             ← all scoring functions
│   └── test_cases.py          ← ground truth Q&A pairs
├── main.py                    ← interactive terminal Q&A
├── evaluate.py                ← evaluation entry point
├── app.py                     ← Streamlit web UI
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/DeepRAG.git
cd DeepRAG
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama
Download from: https://ollama.com
```bash
ollama pull mistral
```

---

## 🖥️ Usage

### Terminal Q&A
```bash
python main.py
python main.py --pdf docs/your_file.pdf
python main.py --no-demo
```

### Streamlit UI
```bash
streamlit run app.py
```

### Run Evaluation
```bash
python evaluate.py
python evaluate.py --rebuild-cache
```

---

## 🔍 How Each Component Works

### 1. PDF Extractor (`pdf_extractor.py`)
Opens the PDF with `pdfplumber` and extracts text from each page.
Returns plain text with page markers.

### 2. Chunker (`chunker.py`)
Splits extracted text into overlapping chunks.
Overlap ensures context is not lost at chunk boundaries.
```
chunk_size = 1000 characters
overlap    = 150 characters
```

### 3. Embedder (`embedder.py`)
Loads `all-MiniLM-L6-v2` from HuggingFace.
Converts each chunk into a 384-dimensional vector.
Also encodes queries for similarity search.

### 4. Vector Store (`vector_store.py`)
Builds a FAISS index from all chunk embeddings.
Supports both flat (exact) and IVF (approximate) search.
Saves and loads from pickle cache.

### 5. Generator (`generator.py`)
Builds a RAG prompt with retrieved chunks as context.
Sends to Mistral via Ollama for local inference.
Supports streaming output.

### 6. Pipeline (`pipeline.py`)
Orchestrates all steps.
Caches embeddings so they only build once.

---

## 📊 Evaluation System

| Level | Method | Metrics |
|-------|--------|---------|
| Level 1 | Manual | Exact match, Keyword match |
| Level 2 | Automated | Cosine similarity, ROUGE-1, ROUGE-2, ROUGE-L |
| Level 3 | LLM Judge | Faithfulness, Relevance, Completeness |

**Results:**
- Initial score: **59.8%** (chunk_size=500)
- After optimization: **68.6%** (chunk_size=1000, top_k=5)

**Score interpretation:**
- 🟢 85%+ Excellent
- 🟡 70%+ Good
- 🟠 50%+ Fair
- 🔴 <50% Needs improvement

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Extraction | pdfplumber |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (faiss-cpu) |
| Local LLM | Ollama + Mistral |
| Evaluation | Custom metrics + LLM judge |
| UI | Streamlit |
| Language | Python 3.10 |

---

## 📈 Key Learnings

**Chunk size matters** — 500 chars gave 59.8%, 1000 chars gave 68.6%

**Overlap is important** — without overlap, context is lost at boundaries

**Evaluation is hard** — exact match alone is not enough. You need semantic similarity + LLM judge for real quality measurement

**Build from scratch first** — understanding every component manually makes frameworks much easier to use later

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

**Muhammad Nadeem**
AI / ML Engineer · RAG · Generative AI · LLM Systems

⭐ If you found this useful, please give it a star!