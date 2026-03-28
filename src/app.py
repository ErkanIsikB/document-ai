import streamlit as st
import tempfile
import time
import os
import sys
from pathlib import Path

# Ensure src/ is importable and .env is loaded
sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import load_pdf, vectorize_pdf, ask_llm

# ── Rate-limit config ──────────────────────────────────────
RATE_LIMIT = 3          # max queries
RATE_WINDOW  = 60        # seconds

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Document AI · PDF Q&A",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Global page bg */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0ff;
    }

    /* Chat message bubbles */
    .stChatMessage {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        backdrop-filter: blur(12px);
    }

    /* Source chunk cards */
    .source-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(124,131,253,0.25);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        transition: border-color 0.2s;
    }
    .source-card:hover {
        border-color: rgba(124,131,253,0.6);
    }
    .source-card .meta {
        font-size: 0.75rem;
        color: #8888cc;
        margin-bottom: 6px;
    }
    .source-card .content {
        font-size: 0.88rem;
        color: #d0d0e8;
        line-height: 1.55;
    }

    /* Highlight matches */
    mark {
        background: rgba(124,131,253,0.35);
        color: #fff;
        padding: 1px 4px;
        border-radius: 3px;
    }

    /* Rate-limit banner */
    .rate-limit-warning {
        background: rgba(255, 87, 87, 0.15);
        border: 1px solid rgba(255, 87, 87, 0.4);
        border-radius: 10px;
        padding: 12px 18px;
        color: #ff9999;
        text-align: center;
        font-weight: 500;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(124,131,253,0.35) !important;
        border-radius: 12px !important;
        padding: 8px !important;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .status-ready {
        background: rgba(46,213,115,0.18);
        color: #6cff8c;
        border: 1px solid rgba(46,213,115,0.35);
    }
    .status-idle {
        background: rgba(255,255,255,0.06);
        color: #888;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ────────────────────────────
defaults = {
    "vectorstore": None,
    "chunks": None,
    "pdf_name": None,
    "messages": [],          # [{role, content, sources}]
    "query_timestamps": [],  # for rate limiting
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ─────────────────────────────────────────────────
def is_rate_limited() -> bool:
    """Check and enforce the per-session rate limit."""
    now = time.time()
    st.session_state.query_timestamps = [
        t for t in st.session_state.query_timestamps if now - t < RATE_WINDOW
    ]
    return len(st.session_state.query_timestamps) >= RATE_LIMIT


def remaining_queries() -> int:
    now = time.time()
    recent = [t for t in st.session_state.query_timestamps if now - t < RATE_WINDOW]
    return max(0, RATE_LIMIT - len(recent))


def highlight_query_in_text(text: str, query: str) -> str:
    """Bold/highlight words from the query that appear in the source text."""
    words = set(query.lower().split())
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "not", "it", "this", "that",
            "with", "from", "by", "as", "what", "how", "who", "which", "about"}
    keywords = [w for w in words if w not in stop and len(w) > 2]
    result = text
    for kw in keywords:
        import re
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        result = pattern.sub(lambda m: f"<mark>{m.group()}</mark>", result)
    return result


# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Document AI")
    st.caption("Upload a PDF and ask questions about its content.")
    st.divider()

    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    # Advanced settings in an expander
    with st.expander("⚙️ Chunking Settings"):
        chunk_size = st.slider("Chunk size", 500, 5000, 2000, step=100,
                               help="Characters per chunk. Larger = more context, fewer chunks.")
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 100, step=50,
                                  help="Overlap between consecutive chunks (10-15% of chunk size recommended).")
        use_unstructured = st.toggle("Use OCR loader (slower, better for scanned PDFs)", value=False)

    # Process PDF button
    if uploaded is not None:
        if uploaded.name != st.session_state.pdf_name:
            if st.button("🔄 Process PDF", use_container_width=True, type="primary"):
                with st.spinner("Loading & vectorizing PDF…"):
                    # Save uploaded file to temp path
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(uploaded.read())
                    tmp.flush()
                    tmp_path = tmp.name
                    tmp.close()

                    try:
                        docs = load_pdf(tmp_path, use_unstructured=use_unstructured)
                        vectorstore, chunks = vectorize_pdf(
                            docs,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        st.session_state.vectorstore = vectorstore
                        st.session_state.chunks = chunks
                        st.session_state.pdf_name = uploaded.name
                        st.session_state.messages = []
                        st.session_state.query_timestamps = []
                        st.success(f"✅ **{uploaded.name}** processed — {len(chunks)} chunks")
                    except Exception as e:
                        st.error(f"Failed to process PDF: {e}")
                    finally:
                        os.unlink(tmp_path)

    st.divider()

    # Status
    if st.session_state.pdf_name:
        st.markdown(f'<span class="status-badge status-ready">● Ready</span>&nbsp;&nbsp;'
                    f'<span style="color:#ccc;font-size:0.85rem">{st.session_state.pdf_name}</span>',
                    unsafe_allow_html=True)
        st.caption(f"{len(st.session_state.chunks)} chunks · {remaining_queries()} queries left this minute")
    else:
        st.markdown('<span class="status-badge status-idle">○ No PDF loaded</span>', unsafe_allow_html=True)


# ── Main area ───────────────────────────────────────────────
st.markdown("# 💬 PDF Q&A")

if not st.session_state.pdf_name:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    # Display chat history (display-only, not sent to LLM)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show sources for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                st.markdown("---")
                st.markdown("##### 📑 Source Chunks")
                for i, src in enumerate(msg["sources"], 1):
                    page = src.get("page", "N/A")
                    content = src["content"]
                    highlighted = highlight_query_in_text(content, msg.get("query", ""))
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="meta">Chunk {i} · Page {page}</div>'
                        f'<div class="content">{highlighted}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # Chat input
    if prompt := st.chat_input("Ask a question about your PDF…"):
        # Rate limit check
        if is_rate_limited():
            st.markdown(
                '<div class="rate-limit-warning">'
                f'⚠️ Rate limit reached — max {RATE_LIMIT} queries per {RATE_WINDOW}s. Please wait a moment.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            # Record timestamp
            st.session_state.query_timestamps.append(time.time())

            # Show user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get answer from LLM (solely from PDF, no chat history sent)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        result = ask_llm(
                            query=prompt,
                            vectorstore=st.session_state.vectorstore,
                        )
                        answer = result["result"]
                        source_docs = result.get("source_documents", [])

                        st.markdown(answer)

                        # Build source data
                        sources = []
                        if source_docs:
                            st.markdown("---")
                            st.markdown("##### 📑 Source Chunks")
                            for i, doc in enumerate(source_docs, 1):
                                page = doc.metadata.get("page", "N/A")
                                content = doc.page_content[:600]
                                highlighted = highlight_query_in_text(content, prompt)
                                sources.append({"page": page, "content": content})
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<div class="meta">Chunk {i} · Page {page}</div>'
                                    f'<div class="content">{highlighted}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "query": prompt,
                        })

                    except Exception as e:
                        st.error(f"Error: {e}")
