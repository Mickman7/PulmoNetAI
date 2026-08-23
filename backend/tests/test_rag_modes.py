from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from backend.agent.nodes import retrieval_node
from backend.agent.rag import RAG_MODE_SOURCES, retrieve_context

# =====================================================================
# rag.py: retrieve_context() source-mixing per condition
# =====================================================================


@pytest.fixture
def mock_local_docs():
    with patch("backend.agent.rag.retrieve_local_context") as mock_local:
        mock_local.return_value = [
            Document(page_content="Local guideline text.", metadata={"source": "NG250.pdf"})
        ]
        yield mock_local


@pytest.fixture
def mock_pubmed_docs():
    with patch("backend.agent.rag.retrieve_pubmed_context") as mock_pubmed:
        mock_pubmed.return_value = [
            Document(
                page_content="PubMed abstract text.",
                metadata={"Title": "CAP outcomes study", "uid": "12345678"},
            )
        ]
        yield mock_pubmed


@pytest.mark.parametrize(
    "mode,expects_local,expects_pubmed",
    [
        ("local", True, False),
        ("pubmed", False, True),
        ("hybrid", True, True),
    ],
)
def test_retrieve_context_per_condition(
    mock_local_docs, mock_pubmed_docs, mode, expects_local, expects_pubmed
):
    """Each of the 3 conditions must pull only from its intended source(s)."""
    result = retrieve_context("test query", sources=RAG_MODE_SOURCES[mode])

    assert mock_local_docs.called == expects_local
    assert mock_pubmed_docs.called == expects_pubmed
    assert ("Local guideline text." in result) == expects_local
    assert ("PubMed abstract text." in result) == expects_pubmed


# =====================================================================
# nodes.py: retrieval_node() mode selection
# =====================================================================


@pytest.fixture
def base_state():
    return {
        "patient_summary": "Test Patient | DOB: 2000-01-01 | Sex: Female",
        "records": [
            {
                "created_at": "2026-07-12 10:00",
                "probability": 0.5,
                "label": "Normal",
                "attention_focus": "Clear lung fields",
                "notes": "",
                "wbc": 7.0,
                "crp": 5.0,
                "vitals_summary": None,
            }
        ],
    }


@pytest.mark.parametrize(
    "mode,expected_sources",
    [
        ("local", ["local"]),
        ("pubmed", ["pubmed"]),
        ("hybrid", ["local", "pubmed"]),
    ],
)
def test_retrieval_node_passes_expected_sources(base_state, mode, expected_sources):
    with patch("backend.agent.nodes.retrieve_context") as mock_retrieve:
        mock_retrieve.return_value = "context block"
        state = {**base_state, "rag_mode": mode}
        result = retrieval_node(state)

        _, kwargs = mock_retrieve.call_args
        assert kwargs["sources"] == expected_sources
        assert result == {"guideline_context": "context block"}


def test_retrieval_node_defaults_to_hybrid_when_unset(base_state):
    """Missing rag_mode (e.g. older callers/tests) should not silently drop a source."""
    with patch("backend.agent.nodes.retrieve_context") as mock_retrieve:
        mock_retrieve.return_value = "context block"
        retrieval_node(base_state)

        _, kwargs = mock_retrieve.call_args
        assert kwargs["sources"] == ["local", "pubmed"]


def test_retrieval_node_rejects_unknown_mode(base_state):
    state = {**base_state, "rag_mode": "not_a_real_mode"}
    with pytest.raises(ValueError):
        retrieval_node(state)
