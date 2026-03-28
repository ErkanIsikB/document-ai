# pip install langchain-community
# pip install "unstructured[pdf]"
# pip install pytesseract
# pip install langchain-openai
# pip install faiss-cpu

# brew install poppler
# brew install tesseract
# Türkçe dil desteği ile kurulum
# brew install tesseract-lang

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # faiss patlio m chiplerde :) boyle is mi olur. Bu satiri eklemezsen faiss patlar.
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader # default — fast
from langchain_community.document_loaders import UnstructuredPDFLoader # fallback — OCR capable
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI

import time
import json
import tempfile
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
api_key = os.getenv('OPENAI_API_KEY')
#print(api_key)
def load_pdf(file_path, use_unstructured=False):
    """Load a PDF and return Document objects. Uses PyMuPDFLoader by default, UnstructuredPDFLoader as fallback."""
    file_path = Path(file_path)
    if use_unstructured:
        loader = UnstructuredPDFLoader(str(file_path))
    else:
        try:
            loader = PyMuPDFLoader(str(file_path))
        except Exception:
            loader = UnstructuredPDFLoader(str(file_path))
    documents = loader.load()
    return documents

def vectorize_pdf(documents, chunk_size=2000, chunk_overlap=100, index_path="faiss_index"):
    """Split documents into chunks, embed them, and save a FAISS index. Returns (vectorstore, chunks)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    # will vectorize these chunks and store them in a vector database. so we can query them later.
    # much better than sending the whole text to the LLM. We will only send the relevant text chunks to the LLM. (database stores both vectors and text chunks)
    # by only sending relevant chunks, we save tokens and make the LLM work faster.
    # this is the base idea of RAG and vector databases.

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore, chunks


def query_pdf(query, index_path="faiss_index", k=2):
    """Run a similarity search against the FAISS index. Returns matching documents."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.similarity_search(query, k=k)


def ask_llm(query, index_path="faiss_index", k=3, vectorstore=None):
    """Ask the LLM a question grounded in the PDF content. Returns {result, source_documents}."""
    if vectorstore is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True
    )
    result = qa_chain.invoke({"query": query})
    return result


if __name__ == "__main__":
    # Example standalone usage
    start_time = time.time()
    file_path = Path(__file__).resolve().parent.parent / 'The_Mentalist_1x01_-_Pilot.pdf'
    docs = load_pdf(file_path, use_unstructured=True)
    vectorize_pdf(docs)
    result = ask_llm("What is told about the main character's ability in the pilot episode?")
    print(result["result"])
    print("\n--- SOURCE DOCUMENTS ---")
    for doc in result["source_documents"]:
        print(f"Page: {doc.metadata.get('page', 'N/A')} | Source: {doc.metadata.get('source')}")
    print(f"Time taken: {time.time() - start_time:.1f}s")