# 📄 Document AI

**LLM-Powered Document Q&A Assistant**

A retrieval-augmented generation (RAG) pipeline that chunks, embeds, and indexes PDF documents — then lets you ask natural-language questions through a Streamlit chat interface. Answers are grounded entirely in the uploaded PDF, with source chunks displayed as citations.

---

## How It Works

```
PDF Upload → Load & Parse → Chunk → Embed (OpenAI) → FAISS Index → Query → GPT-4o Answer + Sources
```

1. **PDF Loading** — `PyMuPDFLoader` (fast, default) or `UnstructuredPDFLoader` (OCR-capable, for scanned documents)
2. **Chunking** — `RecursiveCharacterTextSplitter` with configurable chunk size & overlap
3. **Embedding** — OpenAI `text-embedding-3-small` embeddings stored in a local FAISS vector index
4. **Retrieval QA** — Top-k similar chunks are retrieved and passed to `GPT-4o` via LangChain's `RetrievalQA` chain
5. **Citation Display** — Source chunks are shown below the answer with keyword highlighting

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS (local) |
| Framework | LangChain |
| PDF Parsing | PyMuPDF / Unstructured |
| UI | Streamlit |

---

## Setup

### Prerequisites

- Python 3.10+
- [OpenAI API key](https://platform.openai.com/api-keys)
- macOS: `brew install poppler tesseract` (needed to use the OCR loader)

### Installation

```bash
# Clone the repo
git clone https://github.com/ErkanIsikB/document-ai.git
cd document-ai

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

---

## Usage

### Streamlit UI (recommended)

```bash
streamlit run src/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser:

1. **Upload** a PDF via the sidebar
2. Adjust chunk size/overlap in **⚙️ Chunking Settings** if needed
3. Click **🔄 Process PDF**
4. **Ask questions** in the chat input — answers appear with source citations below

> Rate limit: 3 queries per minute per session.

### CLI

```bash
python src/main.py
```

Runs against `The_Mentalist_1x01_-_Pilot.pdf` by default (edit `__main__` block to change).

---

## Project Structure

```
document-ai/
├── .env                    # API keys (not tracked)
├── .streamlit/
│   └── config.toml         # Streamlit dark theme config
├── requirements.txt        # Python dependencies
├── src/
│   ├── main.py             # Core RAG pipeline (load, chunk, embed, query)
│   └── app.py              # Streamlit chat UI
└── README.md
```

---

## License

[MIT](LICENSE)
