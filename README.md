# 🧠 DeepRAG — Retrieval-Augmented Generation Pipeline

A production-grade RAG pipeline built from scratch. Ask questions about any PDF document using local or cloud LLMs.

---

## 🚀 Features

- 📄 **PDF Loading** — supports text-based and image-based PDFs (OCR fallback)
- ✂️ **Smart Chunking** — RecursiveCharacterTextSplitter preserves natural boundaries
- 🔢 **Embeddings** — HuggingFace `all-MiniLM-L6-v2` (384-dim vectors)
- 🗄️ **Vector Store** — Chroma DB (persistent, no re-embedding on restart)
- 🧠 **Conversation Memory** — remembers previous questions automatically
- 🤖 **Dual LLM Support** — Ollama (local/free) or OpenAI (cloud)
- 📊 **3-Level Evaluation** — manual, automated, and LLM-as-judge scoring

---

## 🏗️ Project Structure

```
DeepRAG/
├── docs/                    ← put your PDF files here
├── rag/
│   ├── __init__.py
│   ├── loader.py            ← PDF loading + OCR fallback
│   ├── splitter.py          ← smart text chunking
│   ├── embedder.py          ← HuggingFace embeddings
│   ├── vector_store.py      ← Chroma vector database
│   ├── llm.py               ← Ollama + OpenAI support
│   └── pipeline.py          ← full chain + conversation memory
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py         ← runs all 3 evaluation levels
│   ├── metrics.py           ← exact match, ROUGE, cosine, LLM judge
│   └── test_cases.py        ← ground truth Q&A pairs
├── main.py                  ← terminal Q&A interface
├── evaluate.py              ← evaluation entry point
├── requirements.txt
└── .env.example
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

### 4. Install Ollama (for local LLM)
Download from: https://ollama.com
```bash
ollama pull mistral
```

### 5. (Optional) OpenAI setup
```bash
cp .env.example .env
# Add your OPENAI_API_KEY inside .env
```

---

## 🖥️ Usage

### Terminal Q&A
```bash
# Using Ollama (local, free)
python main.py --pdf docs/your_file.pdf

# Using OpenAI (cloud)
python main.py --pdf docs/your_file.pdf --provider openai --model gpt-4o-mini

# Skip demo, go straight to interactive mode
python main.py --pdf docs/your_file.pdf --no-demo
```

### Interactive Commands
```
clear    → reset conversation memory
history  → show previous Q&A
exit     → quit
```

### Run Evaluation
```bash
python evaluate.py --pdf docs/your_file.pdf
python evaluate.py --pdf docs/your_file.pdf --rebuild-cache
```

---

## 📊 Evaluation System

| Level | Method | Metrics |
|-------|--------|---------|
| Level 1 | Manual | Exact match, Keyword match |
| Level 2 | Automated | Cosine similarity, ROUGE-1, ROUGE-2, ROUGE-L |
| Level 3 | LLM Judge | Faithfulness, Relevance, Completeness |

**Score interpretation:**
- 🟢 85%+ — Excellent
- 🟡 70%+ — Good
- 🟠 50%+ — Fair
- 🔴 <50% — Needs improvement

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain 1.2.x |
| Embeddings | HuggingFace sentence-transformers |
| Vector DB | Chroma DB |
| Local LLM | Ollama + Mistral |
| Cloud LLM | OpenAI GPT-4o |
| PDF (text) | pdfplumber |
| PDF (image) | pymupdf + pytesseract |
| Memory | ConversationBufferMemory |
| Chain | ConversationalRetrievalChain |

---

## 🧠 How It Works

```
User Question
      ↓
ConversationalRetrievalChain
      ↓
  [Memory] adds chat history
      ↓
  [Retriever] finds top-k chunks from Chroma
      ↓
  [LLM] generates answer from chunks + question
      ↓
  [Memory] saves Q&A for next turn
      ↓
Answer
```

---

## 📈 Learning Journey

This project was built in two phases:

**Phase 1 — From Scratch**
Built every component manually: custom PDF extractor, fixed chunker, FAISS vector index, manual prompt engineering, and a 3-level evaluation system. Initial score: **59.8%**

**Phase 2 — With LangChain**
Rebuilt using LangChain abstractions. Added conversation memory, smarter chunking, persistent Chroma DB. Score improved to: **68.6%**

Key lesson: **Build from scratch first. Then use frameworks.**
You'll understand every abstraction because you've already implemented it manually.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

**Muhammad Nadeem**
AI/ML Engineer · LangChain · RAG · Generative AI

⭐ If you found this useful, please give it a star!