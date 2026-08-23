import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.retrievers import PubMedRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

load_dotenv()

RAG_FILES_DIR = os.path.join(os.path.dirname(__file__), "rag_files")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# The 3 conditions under test: local-only, PubMed-only, and hybrid (both).
RAG_MODE_SOURCES = {
    "local": ["local"],
    "pubmed": ["pubmed"],
    "hybrid": ["local", "pubmed"],
}

_vector_store = None  # Module-level cache for Chroma instance
_pubmed_query_llm = None  # Module-level cache for the query-condensation LLM
_rerank_judge_llm = None  # Module-level cache for the candidate-pool reranker

# How many extra candidates to over-fetch before reranking down to the
# requested k. A binary keep/drop filter (tried and reverted) hurt recall by
# discarding chunks that carried a needed fact but weren't "on-topic" for the
# whole query. This instead pulls a wider candidate pool and keeps the same
# final count as before -- so downstream never sees less coverage, only a
# better-ranked selection from more options.
_RERANK_OVERFETCH = 2


def build_vector_store() -> Chroma:
    """Builds a persisted Chroma vector store using .txt and .pdf files in RAG_FILES_DIR."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )

    documents = []

    # 1. Load PDF files
    if os.path.exists(RAG_FILES_DIR):
        pdf_loader = PyPDFDirectoryLoader(RAG_FILES_DIR)
        loaded_pdfs = pdf_loader.load()
        if loaded_pdfs:
            for doc in loaded_pdfs:
                doc.metadata["source"] = os.path.basename(doc.metadata.get("source", ""))
            documents.extend(loaded_pdfs)

    # 2. Load TXT files
    if os.path.exists(RAG_FILES_DIR):
        for filename in os.listdir(RAG_FILES_DIR):
            if filename.endswith(".txt"):
                filepath = os.path.join(RAG_FILES_DIR, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append(
                    Document(page_content=content, metadata={"source": filename})
                )

    if not documents:
        raise ValueError(
            f"No .pdf or .txt files found in {RAG_FILES_DIR} -- add your guideline corpus first."
        )

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


def _load_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            f"No vector store found at {PERSIST_DIR}. Run build_vector_store() once first."
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    _vector_store = Chroma(
        persist_directory=PERSIST_DIR, embedding_function=embeddings
    )
    return _vector_store


def retrieve_local_context(query: str, k: int = 3) -> List[Document]:
    """Retrieves context from local persistent guideline corpus."""
    try:
        store = _load_vector_store()
        return store.similarity_search(query, k=k)
    except FileNotFoundError:
        print("Warning: Local vector store not built. Skipping local retrieval.")
        return []


def _condense_pubmed_query(query: str) -> str:
    """
    PubMed's ESearch term-mapping performs poorly on verbose natural-language
    text (patient summaries, full questions) -- it needs keyword-dense
    queries. Condense via LLM rather than naive truncation, which was
    returning zero results for realistic queries.
    """
    global _pubmed_query_llm
    if _pubmed_query_llm is None:
        _pubmed_query_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = (
        "Extract 3-5 concise clinical/medical search keywords (MeSH-style terms) "
        "from the text below, suitable for a PubMed search. "
        "Spell out clinical abbreviations in full (e.g. 'SpO2' -> 'oxygen saturation', "
        "'CAP' -> 'community-acquired pneumonia') -- PubMed's automatic term mapping "
        "fails to match many shorthand abbreviations. "
        "Return ONLY the keywords separated by spaces -- no punctuation, no explanation.\n\n"
        f"Text:\n{query[:1500]}"
    )
    try:
        condensed = _pubmed_query_llm.invoke(prompt).content.strip()
        return condensed or query[:200]
    except Exception:
        return query.replace("\n", " ")[:200]


def _flatten_pubmed_title(value) -> str:
    """
    xmltodict parses titles containing inline tags (e.g. italicized species
    names) as nested dicts instead of plain strings -- flatten back to text.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [_flatten_pubmed_title(v) for k, v in value.items() if not k.startswith("@")]
        return " ".join(p for p in parts if p)
    if isinstance(value, list):
        return " ".join(_flatten_pubmed_title(v) for v in value)
    return str(value) if value else ""


# Primary research abstracts rarely restate definitional/threshold facts
# (e.g. what CURB-65 scores, what CRP cutoff means) the way review articles
# and practice guidelines do -- bias toward those publication types first.
_PUBMED_REVIEW_FILTER = " AND (review[pt] OR practice guideline[pt] OR guideline[pt])"


def retrieve_pubmed_context(query: str, k: int = 2) -> List[Document]:
    """Retrieves peer-reviewed medical abstracts directly from PubMed API."""
    try:
        retriever = PubMedRetriever(
            top_k_results=k,
            email=os.environ.get("NCBI_ENTREZ_EMAIL", "your_email@example.com"),
            api_key=os.environ.get("NCBI_API_KEY", ""),
        )
        api_query = _condense_pubmed_query(query)

        # Prefer reviews/guidelines; the filter can over-narrow niche topics
        # to zero hits, so fall back to the unfiltered query when that happens.
        docs = retriever.invoke(api_query + _PUBMED_REVIEW_FILTER)
        if not docs:
            docs = retriever.invoke(api_query)

        for doc in docs:
            doc.metadata["Title"] = _flatten_pubmed_title(doc.metadata.get("Title", ""))
        return docs
    except Exception as e:
        print(f"Warning: PubMed API retrieval failed: {e}")
        return []


class _RelevanceScore(BaseModel):
    score: float  # 0.0 (irrelevant) - 1.0 (directly relevant)


def _score_chunk_relevance(query: str, chunk_text: str) -> float:
    global _rerank_judge_llm
    if _rerank_judge_llm is None:
        _rerank_judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = (
        "You are scoring a retrieved chunk for a clinical RAG system.\n"
        f"Query: {query}\n\nRetrieved chunk:\n{chunk_text}\n\n"
        "Score, from 0.0 (irrelevant) to 1.0 (directly relevant), how useful "
        "this chunk is for answering the query."
    )
    try:
        score = _rerank_judge_llm.with_structured_output(_RelevanceScore).invoke(prompt).score
        return max(0.0, min(1.0, score))
    except Exception:
        return 1.0  # fail open -- never let a judge error silently drop content


def _rerank_and_trim(query: str, docs: List[Document], keep: int) -> List[Document]:
    """Keeps the top `keep` docs by graded relevance. A no-op when the
    candidate pool isn't larger than what's being kept."""
    if len(docs) <= keep:
        return docs
    scored = [(doc, _score_chunk_relevance(query, doc.page_content)) for doc in docs]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in scored[:keep]]


def retrieve_context(
    query: str,
    k: int = 4,
    k_local: Optional[int] = None,
    k_pubmed: int = 2,
    sources: Optional[List[str]] = None,
) -> str:
    """
    Returns top guideline and medical literature chunks for the query.

    If 'k' is passed directly (e.g. k=4), it sets k_local for backward compatibility.
    """
    if k_local is None:
        k_local = k

    if sources is None:
        sources = ["local", "pubmed"]

    blocks = []
    excerpt_counter = 1

    # 1. Fetch Local Guideline Corpus
    if "local" in sources:
        local_docs = retrieve_local_context(query, k=k_local + _RERANK_OVERFETCH)
        local_docs = _rerank_and_trim(query, local_docs, keep=k_local)
        for doc in local_docs:
            src = doc.metadata.get("source", "local_guideline")
            blocks.append(
                f"[Guideline excerpt {excerpt_counter}, source: {src}]\n{doc.page_content}"
            )
            excerpt_counter += 1

    # 2. Fetch Live PubMed Peer-Reviewed Literature
    if "pubmed" in sources:
        pubmed_docs = retrieve_pubmed_context(query, k=k_pubmed + _RERANK_OVERFETCH)
        pubmed_docs = _rerank_and_trim(query, pubmed_docs, keep=k_pubmed)
        for doc in pubmed_docs:
            title = doc.metadata.get("Title") or "PubMed Article"
            pmid = doc.metadata.get("uid", "Unknown PMID")
            blocks.append(
                f"[PubMed Literature excerpt {excerpt_counter}, PMID: {pmid}, Title: {title}]\n{doc.page_content}"
            )
            excerpt_counter += 1

    if not blocks:
        return "No relevant guideline or literature content found."

    return "\n\n".join(blocks)

def build_retrieval_query(patient_summary: str, records_block: str) -> str:
    """Builds a search query from patient evidence."""
    return f"{patient_summary}\n{records_block}"


if __name__ == "__main__":
    # 1. Build local store if missing
    if not os.path.exists(PERSIST_DIR):
        build_vector_store()

    # 2. Test all 3 retrieval conditions: local, pubmed, hybrid
    test_query = "adult community-acquired pneumonia elevated CRP high fever"

    for mode, sources in RAG_MODE_SOURCES.items():
        print(f"\n--- Testing {mode.upper()} Retrieval ({', '.join(sources)}) ---")
        results = retrieve_context(test_query, k_local=2, k_pubmed=2, sources=sources)
        print(results)