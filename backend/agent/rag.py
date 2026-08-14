"""
RAG pipeline over your clinical guideline corpus (backend/agent/rag_corpus/).

Unlike the LangChain tutorial you were testing, this does NOT use an LLM
tool-calling loop or a grade->rewrite->retry cycle. Retrieval here is
deterministic: build_retrieval_query() always builds a query from the case's
actual evidence, retrieve_context() always runs once, and the result is
plain text handed to your existing analysis/reasoning/report nodes. This
matches your AgentState design (structured fields, fixed node sequence)
rather than the tutorial's MessagesState/tool-calling design.

Build the vector store ONCE (build_vector_store()) -- it persists to disk,
so subsequent agent runs just load it rather than re-embedding every time.
"""

import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAG_FILES_DIR = os.path.join(os.path.dirname(__file__), "rag_files")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

_vector_store = None  # module-level cache so we don't reload from disk every call

from dotenv import load_dotenv
load_dotenv()


def build_vector_store():
    """Builds a persisted Chroma vector store using both .txt and .pdf files 
    found in the RAG_FILES_DIR."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )

    documents = []

    # 1. Load PDF files using PyPDFDirectoryLoader
    if os.path.exists(RAG_FILES_DIR):
        pdf_loader = PyPDFDirectoryLoader(RAG_FILES_DIR)
        loaded_pdfs = pdf_loader.load()
        if loaded_pdfs:
            # Standardize metadata source tag to filename only
            for doc in loaded_pdfs:
                doc.metadata["source"] = os.path.basename(doc.metadata.get("source", ""))
            documents.extend(loaded_pdfs)

    # 2. Load TXT files
    for filename in os.listdir(RAG_FILES_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(RAG_FILES_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            # Wrap as a Document object matching PyPDF output
            from langchain_core.documents import Document
            documents.append(Document(page_content=content, metadata={"source": filename}))

    if not documents:
        raise ValueError(
            f"No .pdf or .txt files found in {RAG_FILES_DIR} -- add your guideline corpus first."
        )

    # Split documents (PDFs + TXTs) into chunks
    doc_chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    sources = set(doc.metadata.get("source", "unknown") for doc in doc_chunks)
    print(
        f"Vector store built: {len(doc_chunks)} chunks from {len(sources)} file(s), "
        f"saved to {PERSIST_DIR}"
    )
    return store


def _load_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            f"No vector store found at {PERSIST_DIR}. Run build_vector_store() once first."
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    _vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    return _vector_store


def retrieve_context(query: str, k: int = 4) -> str:
    """Returns the top-k most relevant guideline chunks for the query, as a
    single formatted text block ready to drop into a prompt."""
    store = _load_vector_store()
    results = store.similarity_search(query, k=k)

    if not results:
        return "No relevant guideline content found."

    blocks = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[Guideline excerpt {i}, source: {source}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_retrieval_query(patient_summary: str, records_block: str) -> str:
    """Builds a search query from the case's evidence -- NOT from the model's
    prediction alone, so retrieval isn't just re-confirming the classifier
    (the same confirmation-bias issue flagged in your supervisor feedback)."""
    return f"{patient_summary}\n{records_block}"


if __name__ == "__main__":
    # Run once to build the store: `python3 rag.py` from backend/agent/
    build_vector_store()

    # Quick smoke test
    print("\n--- Test retrieval ---")
    print(retrieve_context("elevated CRP, high fever, low oxygen saturation, adult in hospital", k=3))