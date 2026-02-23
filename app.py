"""
app.py
------
Streamlit UI for the RAG-Based Q&A system.

Run with:
    streamlit run app.py
"""

import os
import sys
import tempfile
import streamlit as st # pyright: ignore[reportMissingImports]

sys.path.insert(0, os.path.dirname(__file__))

from rag.pipeline import RAGPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
}

/* ── Hide Streamlit default chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 2rem 2rem; max-width: 100%; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e3a;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #7c6af7;
}

/* ── Title ── */
.rag-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    background: linear-gradient(135deg, #7c6af7 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    margin-bottom: 0;
    line-height: 1.1;
}
.rag-subtitle {
    font-size: 0.85rem;
    color: #475569;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.2rem;
    font-weight: 600;
}

/* ── Chat messages ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    margin-bottom: 1.5rem;
}

.msg-user {
    background: linear-gradient(135deg, #1e1b4b, #1a1035);
    border: 1px solid #312e81;
    border-radius: 16px 16px 4px 16px;
    padding: 1rem 1.2rem;
    margin-left: 15%;
    color: #c4b5fd;
    font-size: 0.95rem;
    line-height: 1.6;
}

.msg-assistant {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.2rem;
    margin-right: 15%;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.7;
}

.msg-label-user {
    font-size: 0.7rem;
    color: #7c6af7;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}

.msg-label-assistant {
    font-size: 0.7rem;
    color: #38bdf8;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Chunks panel ── */
.chunk-card {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-left: 3px solid #7c6af7;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #94a3b8;
    line-height: 1.6;
}

.chunk-score {
    font-size: 0.68rem;
    color: #7c6af7;
    font-weight: 600;
    margin-bottom: 0.4rem;
    letter-spacing: 0.05em;
}

/* ── Status badges ── */
.badge-ready {
    display: inline-block;
    background: #052e16;
    color: #4ade80;
    border: 1px solid #166534;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
}
.badge-waiting {
    display: inline-block;
    background: #1c1917;
    color: #78716c;
    border: 1px solid #292524;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Input box ── */
.stTextInput input {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #7c6af7 !important;
    box-shadow: 0 0 0 2px rgba(124, 106, 247, 0.15) !important;
}

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, #7c6af7, #6d28d9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124, 106, 247, 0.4) !important;
}

/* ── Selectbox / slider ── */
.stSelectbox > div > div,
.stSlider > div {
    background: #0f172a !important;
}

/* ── Divider ── */
hr { border-color: #1e293b !important; }

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 0.8rem;
    margin: 0.8rem 0;
}
.metric-card {
    flex: 1;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 0.7rem;
    text-align: center;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #7c6af7;
}
.metric-label {
    font-size: 0.65rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0f172a !important;
    border: 1px dashed #1e293b !important;
    border-radius: 12px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0f172a !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "pipeline"      not in st.session_state: st.session_state.pipeline      = None
if "pipeline_info" not in st.session_state: st.session_state.pipeline_info = {}
if "last_chunks"   not in st.session_state: st.session_state.last_chunks   = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 RAG Q&A")
    st.markdown("---")

    # ── PDF Upload ──
    st.markdown("#### 📄 Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload any PDF to ask questions about it",
    )

    # Also allow using default PDF
    use_default = st.checkbox(
        "Use default resume PDF",
        value=not bool(uploaded_file),
        disabled=bool(uploaded_file),
    )

    st.markdown("---")

    # ── Model settings ──
    st.markdown("#### ⚙️ Settings")

    ollama_model = st.selectbox(
        "LLM Model",
        ["mistral", "llama3", "llama3.2", "gemma3:4b", "llava"],
        index=0,
        help="Choose your Ollama model",
    )

    top_k = st.slider(
        "Retrieved chunks (top-k)",
        min_value=1, max_value=10, value=5,
        help="How many chunks to retrieve per query",
    )

    chunk_size = st.select_slider(
        "Chunk size",
        options=[500, 750, 1000, 1500, 2000],
        value=1000,
        help="Larger = more context per chunk",
    )

    st.markdown("---")

    # ── Build pipeline button ──
    build_btn = st.button("🚀 Load Pipeline", use_container_width=True)

    if build_btn:
        pdf_path = None

        if uploaded_file:
            # Save uploaded file to a temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name
        elif use_default:
            pdf_path = "docs/Nadeem_Updated_Resume_26.pdf"

        if pdf_path and os.path.exists(pdf_path):
            with st.spinner("Building pipeline..."):
                try:
                    # Use unique cache per pdf+settings combo
                    cache_name = f"rag_cache_{chunk_size}_{top_k}.pkl"
                    st.session_state.pipeline = RAGPipeline(
                        pdf_path=pdf_path,
                        chunk_size=chunk_size,
                        overlap=chunk_size // 7,
                        top_k=top_k,
                        ollama_model=ollama_model,
                        cache_file=cache_name,
                    )
                    st.session_state.pipeline_info = {
                        "model":      ollama_model,
                        "top_k":      top_k,
                        "chunk_size": chunk_size,
                        "pdf":        uploaded_file.name if uploaded_file else "Default Resume",
                    }
                    st.session_state.messages    = []
                    st.session_state.last_chunks = []
                    st.success("✓ Pipeline ready!")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("PDF not found. Please upload a file or check the default path.")

    st.markdown("---")

    # ── Pipeline status ──
    st.markdown("#### 📊 Status")
    if st.session_state.pipeline:
        info = st.session_state.pipeline_info
        st.markdown('<span class="badge-ready">● READY</span>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">{info.get('top_k', '-')}</div>
        <div class="metric-label">Top-K</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{info.get('chunk_size', '-')}</div>
        <div class="metric-label">Chunk</div>
    </div>
</div>
<div style="font-size:0.72rem; color:#475569; font-family:'JetBrains Mono',monospace; margin-top:0.5rem;">
    Model: {info.get('model', '-')}<br>
    PDF: {info.get('pdf', '-')}
</div>
""", unsafe_allow_html=True)

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages    = []
            st.session_state.last_chunks = []
            st.rerun()
    else:
        st.markdown('<span class="badge-waiting">○ NOT LOADED</span>', unsafe_allow_html=True)
        st.caption("Upload a PDF and click Load Pipeline")


# ── Main area ─────────────────────────────────────────────────────────────────
col_chat, col_chunks = st.columns([3, 1.2])

with col_chat:
    # Title
    st.markdown('<div class="rag-title">RAG Q&A System</div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-subtitle">Retrieval-Augmented Generation · Local LLM · PDF Intelligence</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat history ──
    chat_html = '<div class="chat-container">'

    if not st.session_state.messages:
        st.markdown("""
<div style="text-align:center; padding: 3rem 0; color: #334155;">
    <div style="font-size:3rem; margin-bottom:1rem;">🧠</div>
    <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:600; color:#475569;">
        Load a pipeline from the sidebar, then ask anything about your document.
    </div>
</div>
""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"""
<div class="msg-user">
    <div class="msg-label-user">▸ You</div>
    {msg['content']}
</div>"""
            else:
                chat_html += f"""
<div class="msg-assistant">
    <div class="msg-label-assistant">◈ Assistant</div>
    {msg['content']}
</div>"""

        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    # ── Input ──
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.pipeline:
        with st.form(key="chat_form", clear_on_submit=True):
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                user_input = st.text_input(
                    "Ask a question",
                    placeholder="What are this person's technical skills?",
                    label_visibility="collapsed",
                )
            with col_btn:
                submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and user_input.strip():
            question = user_input.strip()

            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})

            # Retrieve + generate
            with st.spinner("Thinking..."):
                try:
                    retrieval = st.session_state.pipeline.retrieve(question)
                    context_chunks = [chunk for _, _, chunk in retrieval]

                    # Store chunks for the sidebar panel
                    st.session_state.last_chunks = [
                        {"rank": i + 1, "similarity": sim, "text": chunk}
                        for i, (sim, _, chunk) in enumerate(retrieval)
                    ]

                    # Generate answer (non-streaming for UI)
                    answer = st.session_state.pipeline.generator.generate(
                        question, context_chunks, stream=False
                    )

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ Error: {e}"
                    })

            st.rerun()
    else:
        st.markdown("""
<div style="background:#0f172a; border:1px dashed #1e293b; border-radius:12px;
            padding:1rem; text-align:center; color:#334155; font-size:0.85rem;">
    ← Load a pipeline from the sidebar to start asking questions
</div>
""", unsafe_allow_html=True)


# ── Right panel — retrieved chunks ────────────────────────────────────────────
with col_chunks:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:0.72rem; color:#475569; font-weight:700;
            letter-spacing:0.12em; text-transform:uppercase;
            font-family:'JetBrains Mono',monospace; margin-bottom:0.8rem;">
    Retrieved Context
</div>
""", unsafe_allow_html=True)

    if st.session_state.last_chunks:
        for chunk_info in st.session_state.last_chunks:
            score_pct = chunk_info["similarity"] * 100
            preview   = chunk_info["text"].replace("\n", " ").strip()

            # Color the border based on relevance score
            color = "#4ade80" if score_pct > 60 else "#f59e0b" if score_pct > 40 else "#f87171"

            with st.expander(f"Chunk {chunk_info['rank']}  ·  {score_pct:.1f}%"):
                st.markdown(f"""
<div class="chunk-score" style="color:{color};">
    ▸ Relevance: {score_pct:.1f}%
</div>
<div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem;
            color:#64748b; line-height:1.7; white-space:pre-wrap;">
{preview[:400]}{'...' if len(preview) > 400 else ''}
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="font-size:0.78rem; color:#334155; font-family:'JetBrains Mono',monospace;
            line-height:1.8; padding:1rem; background:#0a0a0f;
            border:1px solid #1e293b; border-radius:8px;">
    Chunks retrieved for<br>each query will<br>appear here.
</div>
""", unsafe_allow_html=True)