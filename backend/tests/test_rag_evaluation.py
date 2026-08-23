"""
Real RAG evaluation across the 3 retrieval conditions (local / pubmed / hybrid).

Unlike test_rag_modes.py (fast, mocked, checks wiring), this hits the real
local Chroma store, the live PubMed API, and an LLM judge -- it costs API
calls and takes real time. It's opt-in: run with

    RUN_RAG_EVAL=1 pytest backend/tests/test_rag_evaluation.py -s -v

to print the filled-out Retrieval & RAG Evaluation Results table. Skipped by
default so `pytest backend/tests/` stays fast/free/offline.

Metrics are a lightweight LLM-judge approximation of the RAGAS definitions
(no external ragas dependency -- ragas 0.4.3 currently fails to import due to
an unpatched upstream bug, see github.com/vibrantlabsai/ragas/issues/2745):
  - Context Precision: fraction of retrieved chunks an LLM judges relevant to the query
  - Context Recall: fraction of pre-written reference key points an LLM finds supported by the retrieved context
  - Context Relevancy: single LLM-graded 0.0-1.0 score of how relevant the full retrieved context is to the query
  - Mean Retrieval Latency: wall-clock seconds per retrieve_context() call, averaged over the eval set
"""

import os
import time

import pytest
from pydantic import BaseModel

from backend.agent.rag import RAG_MODE_SOURCES, retrieve_context

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_RAG_EVAL"),
    reason="Real RAG evaluation hits OpenAI + PubMed APIs; set RUN_RAG_EVAL=1 to run.",
)

# =====================================================================
# Eval set -- clinical queries grounded in the local BTS/NICE guideline
# corpus, each with hand-written reference key points for recall scoring.
# =====================================================================

EVAL_SET = [
    {
        "query": "What CRP threshold, alongside clinical signs, indicates severe community-acquired pneumonia requiring hospital admission?",
        "reference_key_points": [
            "CRP above 100 mg/L indicates a higher likelihood of severe pneumonia needing admission",
            "CRP is interpreted alongside clinical assessment (e.g. CURB-65), not in isolation",
        ],
    },
    {
        "query": "What SpO2 level should prompt supplemental oxygen and escalation of care in an adult with suspected pneumonia?",
        "reference_key_points": [
            "SpO2 below 92% on air is a threshold for supplemental oxygen and escalation",
            "Target saturation ranges differ for patients at risk of hypercapnic respiratory failure",
        ],
    },
    {
        "query": "What does the CURB-65 score assess in patients with community-acquired pneumonia?",
        "reference_key_points": [
            "CURB-65 scores confusion, urea, respiratory rate, blood pressure, and age over 65",
            "A higher CURB-65 score indicates greater severity and mortality risk, guiding admission decisions",
        ],
    },
    {
        "query": "When should intravenous antibiotics for community-acquired pneumonia be reviewed for switching to oral therapy?",
        "reference_key_points": [
            "IV antibiotics should be reviewed by 48 hours, considering a switch to oral to complete the course",
            "Antibiotic choice should be reviewed against microbiological culture results when available",
        ],
    },
    {
        "query": "What oxygen saturation range should be targeted in acutely ill adults, and how does that differ for patients at risk of hypercapnic respiratory failure?",
        "reference_key_points": [
            "Target saturation range is 94-98% for most acutely ill patients with a reliable oximetry reading",
            "Target saturation range is 88-92% for patients at risk of hypercapnic respiratory failure, such as those with COPD, morbid obesity, chest wall deformities, or neuromuscular disorders",
        ],
    },
    {
        "query": "When should a follow-up chest X-ray be arranged after a patient is discharged following treatment for pneumonia?",
        "reference_key_points": [
            "Follow-up chest X-rays are typically arranged 4 to 8 weeks after discharge from inpatient care",
            "The purpose is to check the pneumonia has resolved and to detect underlying conditions such as lung cancer",
        ],
    },
    {
        "query": "How does hospital-acquired pneumonia differ in definition from community-acquired pneumonia?",
        "reference_key_points": [
            "Hospital-acquired pneumonia develops 48 hours or more after hospital admission and was not incubating at the time of admission",
            "Community-acquired pneumonia is diagnosed from symptoms and signs of lower respiratory tract infection arising outside of that hospital-acquired timeframe",
        ],
    },
    {
        "query": "What is the CRB-65 score used for in primary care, and what factors does it include?",
        "reference_key_points": [
            "CRB65 is used in primary care to assess mortality risk in suspected community-acquired pneumonia without requiring a blood urea test",
            "CRB65 scores confusion, raised respiratory rate (30 or more breaths per minute), low blood pressure, and age 65 or over",
        ],
    },
]

CONDITION_LABELS = {
    "local": "Local Guidelines Only",
    "pubmed": "Live PubMed API Only",
    "hybrid": "Hybrid (Local + PubMed)",
}


# =====================================================================
# LLM-judge metric primitives (lazy client -- only built if the test
# actually runs, so collection never requires an API key)
# =====================================================================


class _Verdict(BaseModel):
    verdict: bool


class _Score(BaseModel):
    score: float  # 0.0-1.0


_judge_llm = None


def _get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        from langchain_openai import ChatOpenAI

        _judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _judge_llm


def _split_chunks(context: str) -> list:
    """retrieve_context() joins excerpts with a blank line -- split back out."""
    if context.startswith("No relevant"):
        return []
    return [c.strip() for c in context.split("\n\n") if c.strip()]


def _judge_chunk_relevant(query: str, chunk: str) -> bool:
    prompt = (
        "You are grading retrieval quality for a clinical RAG system.\n"
        f"Query: {query}\n\nRetrieved chunk:\n{chunk}\n\n"
        "Is this chunk relevant to answering the query? Answer strictly true or false."
    )
    return _get_judge_llm().with_structured_output(_Verdict).invoke(prompt).verdict


def _judge_claim_supported(claim: str, context: str) -> bool:
    prompt = (
        "You are grading retrieval quality for a clinical RAG system.\n"
        f"Reference claim: {claim}\n\nRetrieved context:\n{context}\n\n"
        "Is this claim supported by the retrieved context? Answer strictly true or false."
    )
    return _get_judge_llm().with_structured_output(_Verdict).invoke(prompt).verdict


def _judge_relevancy(query: str, context: str) -> float:
    prompt = (
        "You are grading retrieval quality for a clinical RAG system.\n"
        f"Query: {query}\n\nRetrieved context:\n{context}\n\n"
        "Score, from 0.0 (completely irrelevant) to 1.0 (fully relevant), how "
        "relevant the retrieved context as a whole is to answering the query."
    )
    score = _get_judge_llm().with_structured_output(_Score).invoke(prompt).score
    return max(0.0, min(1.0, score))


# =====================================================================
# Per-condition evaluation
# =====================================================================


def _evaluate_condition(mode: str) -> dict:
    sources = RAG_MODE_SOURCES[mode]
    precisions, recalls, relevancies, latencies = [], [], [], []

    for item in EVAL_SET:
        query = item["query"]

        start = time.perf_counter()
        context = retrieve_context(query, k_local=3, k_pubmed=2, sources=sources)
        latencies.append(time.perf_counter() - start)

        chunks = _split_chunks(context)
        if chunks:
            relevant_flags = [_judge_chunk_relevant(query, c) for c in chunks]
            precisions.append(sum(relevant_flags) / len(relevant_flags))
            relevancies.append(_judge_relevancy(query, context))
        else:
            precisions.append(0.0)
            relevancies.append(0.0)

        supported_flags = [
            _judge_claim_supported(claim, context) for claim in item["reference_key_points"]
        ]
        recalls.append(sum(supported_flags) / len(supported_flags))

    n = len(EVAL_SET)
    return {
        "context_precision": sum(precisions) / n,
        "context_recall": sum(recalls) / n,
        "context_relevancy": sum(relevancies) / n,
        "mean_latency_s": sum(latencies) / n,
    }


def test_rag_evaluation_table():
    """Fills out the Retrieval & RAG Evaluation Results table for all 3 conditions."""
    rows = [(label, _evaluate_condition(mode)) for mode, label in CONDITION_LABELS.items()]

    header = (
        f"{'Configuration':<28} {'Ctx Precision':>14} {'Ctx Recall':>12} "
        f"{'Ctx Relevancy':>14} {'Mean Latency (s)':>18}"
    )
    print("\n" + header)
    print("-" * len(header))
    for label, m in rows:
        print(
            f"{label:<28} {m['context_precision']:>14.2f} {m['context_recall']:>12.2f} "
            f"{m['context_relevancy']:>14.2f} {m['mean_latency_s']:>18.2f}"
        )

    # Sanity assertions -- catches total pipeline breakage, not quality regressions
    for _, m in rows:
        assert 0.0 <= m["context_precision"] <= 1.0
        assert 0.0 <= m["context_recall"] <= 1.0
        assert 0.0 <= m["context_relevancy"] <= 1.0
        assert m["mean_latency_s"] >= 0.0
