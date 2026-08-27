from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

# Use consistent absolute imports matching your project layout
from backend.agent.graph import build_agent
from backend.agent.state import AgentState, PredictionRecord

# =====================================================================
# Fixtures & Mocks
# =====================================================================

@pytest.fixture
def compiled_agent():
    """Returns a compiled instance of the LangGraph agent graph."""
    return build_agent()


@pytest.fixture
def mock_retriever():
    """Mocks the Chroma vector store search to prevent real DB/API calls during testing."""
    # Ensure this targets the exact module where retrieve_context is invoked by your nodes
    with patch("backend.agent.nodes.retrieve_context") as mock_retrieve:
        mock_retrieve.return_value = (
            "[Guideline excerpt 1, source: NG250.pdf]\n"
            "Elevated CRP (>100 mg/L) indicates severe inflammation. "
            "Diagnosis of pneumonia requires clinical signs and chest X-ray confirmation."
        )
        yield mock_retrieve


@pytest.fixture
def base_patient_summary():
    return "Neha Anand | DOB: 2005-09-21 | Sex: Female | Chronic conditions: PCOS | Allergies: None"


# =====================================================================
# Pipeline Scenario Tests
# =====================================================================

def test_scenario_a_cross_modal_override(compiled_agent, mock_retriever, base_patient_summary):
    """Scenario A: High blood markers (CRP 100), but normal X-ray & model prediction 'Normal'."""
    
    initial_state: AgentState = {
        "patient_summary": base_patient_summary,
        "records": [
            {
                "created_at": "2026-07-12 10:00",
                "probability": 0.38,
                "label": "Normal",
                "attention_focus": "Clear bilateral lung fields",
                "notes": "Patient presents with persistent cough.",
                "wbc": 10.0,
                "crp": 100.0,
                "vitals_summary": None,
            }
        ],
        "guideline_context": None,
        "analysis": None,
        "reasoning": None,
        "report": None,
    }

    mock_responses = [
        AIMessage(content="SUFFICIENT\nGuideline excerpt directly covers the CRP threshold at play."),
        AIMessage(content="The clear bilateral lung fields drive the 'Normal' prediction despite elevated CRP."),
        AIMessage(content="Valid cross-modal override. Elevated CRP indicates systemic inflammation, but clear X-ray rules out pneumonia [Source: NG250.pdf]."),
        AIMessage(content="RESPIRATORY CONSULTATION REPORT\n4. CLINICAL INTERPRETATION\nValid override [Source: NG250.pdf]."),
    ]

    with patch.object(ChatOpenAI, "invoke", side_effect=mock_responses) as mock_llm:
        final_state = compiled_agent.invoke(initial_state)

        mock_retriever.assert_called_once()
        assert "guideline_context" in final_state
        assert "NG250.pdf" in final_state["guideline_context"]
        assert final_state["retrieval_sufficient"] is True
        assert final_state["retrieval_attempts"] == 1

        assert mock_llm.call_count == 4
        assert "Normal" in final_state["analysis"]
        assert "cross-modal override" in final_state["reasoning"].lower()
        assert "RESPIRATORY CONSULTATION REPORT" in final_state["report"]


def test_scenario_b_missing_vitals_and_history(compiled_agent, mock_retriever, base_patient_summary):
    """Scenario B: Incomplete clinical data (no vitals provided). Verifies 'Not available/Not assessed' handling."""
    
    initial_state: AgentState = {
        "patient_summary": base_patient_summary,
        "records": [
            {
                "created_at": "2026-07-12 11:30",
                "probability": 0.42,
                "label": "Normal",
                "attention_focus": "No focal consolidation",
                "notes": "Mild fatigue.",
                "wbc": 6.5,
                "crp": 4.0,
                "vitals_summary": None,
            }
        ],
        "guideline_context": None,
        "analysis": None,
        "reasoning": None,
        "report": None,
    }

    mock_responses = [
        AIMessage(content="SUFFICIENT\nGuideline excerpt is on-topic for this case."),
        AIMessage(content="Imaging and lab markers align with normal classification. Vitals were not provided."),
        AIMessage(content="Prediction is clinically consistent. Absence of vitals is not treated as a red flag."),
        AIMessage(content="RESPIRATORY CONSULTATION REPORT\nVitals Log: Not available/Not assessed."),
    ]

    with patch.object(ChatOpenAI, "invoke", side_effect=mock_responses):
        final_state = compiled_agent.invoke(initial_state)

        assert final_state["analysis"] is not None
        assert "Not available/Not assessed" in final_state["report"]


def test_scenario_c_concordant_severe_pneumonia(compiled_agent, mock_retriever, base_patient_summary):
    """Scenario C: High WBC, high CRP, low SpO2, and Pneumonia prediction (all modalities agree)."""
    
    initial_state: AgentState = {
        "patient_summary": base_patient_summary,
        "records": [
            {
                "created_at": "2026-07-12 14:00",
                "probability": 0.94,
                "label": "Pneumonia",
                "attention_focus": "Right lower lobe consolidation",
                "notes": "High fever, dyspnoea.",
                "wbc": 18.5,
                "crp": 150.0,
                "vitals_summary": "HR: 115 bpm, RR: 28 breaths/min, SpO2: 89%",
            }
        ],
        "guideline_context": None,
        "analysis": None,
        "reasoning": None,
        "report": None,
    }

    mock_responses = [
        AIMessage(content="SUFFICIENT\nGuideline excerpt covers severe pneumonia thresholds."),
        AIMessage(content="Dense right lower lobe consolidation with high CRP and hypoxia strongly supports severe pneumonia."),
        AIMessage(content="All modalities (labs, X-ray, vitals) concordantly support pneumonia diagnosis per NG250 guidance."),
        AIMessage(content="RESPIRATORY CONSULTATION REPORT\n5. MANAGEMENT PLAN\nCare Setting Recommendation: Inpatient admission."),
    ]

    with patch.object(ChatOpenAI, "invoke", side_effect=mock_responses):
        final_state = compiled_agent.invoke(initial_state)

        assert "Pneumonia" in final_state["records"][0]["label"]
        assert "Inpatient" in final_state["report"]


def test_scenario_d_longitudinal_multi_record(compiled_agent, mock_retriever, base_patient_summary):
    """Scenario D: Two static records evaluated independently (Pneumonia -> Normal following treatment)."""
    
    record_1: PredictionRecord = {
        "created_at": "2026-07-01 09:00",
        "probability": 0.88,
        "label": "Pneumonia",
        "attention_focus": "Left mid-zone infiltrate",
        "notes": "Initial admission.",
        "wbc": 16.0,
        "crp": 120.0,
        "vitals_summary": "SpO2: 91%",
    }
    
    record_2: PredictionRecord = {
        "created_at": "2026-07-10 10:00",
        "probability": 0.15,
        "label": "Normal",
        "attention_focus": "Resolving infiltrate, clear margins",
        "notes": "Post-antibiotic review.",
        "wbc": 7.2,
        "crp": 8.5,
        "vitals_summary": "SpO2: 98%",
    }

    initial_state: AgentState = {
        "patient_summary": base_patient_summary,
        "records": [record_1, record_2],
        "guideline_context": None,
        "analysis": None,
        "reasoning": None,
        "report": None,
    }

    mock_responses = [
        AIMessage(content="SUFFICIENT\nGuideline excerpt applies to both encounters."),
        AIMessage(content="Record 1 shows acute pneumonia features. Record 2 demonstrates feature normalization across independent encounters."),
        AIMessage(content="Both records are internally consistent. The transition reflects feature variation between independent assessments."),
        AIMessage(content="RESPIRATORY CONSULTATION REPORT\nEvaluation of 2 encounters shows radiological and lab resolution."),
    ]

    with patch.object(ChatOpenAI, "invoke", side_effect=mock_responses):
        final_state = compiled_agent.invoke(initial_state)

        assert len(final_state["records"]) == 2
        assert final_state["records"][0]["label"] == "Pneumonia"
        assert final_state["records"][1]["label"] == "Normal"
        assert "2 encounters" in final_state["report"]


# =====================================================================
# Corrective-RAG loop: grade -> rewrite -> retry
# =====================================================================

def test_scenario_e_retrieval_grade_and_retry(compiled_agent, base_patient_summary):
    """An empty first-pass retrieval is graded insufficient (deterministically,
    no LLM call needed) and triggers a query rewrite + a second retrieval
    attempt before the pipeline proceeds."""

    initial_state: AgentState = {
        "patient_summary": base_patient_summary,
        "records": [
            {
                "created_at": "2026-07-12 10:00",
                "probability": 0.6,
                "label": "Pneumonia",
                "attention_focus": "Right basal opacity",
                "notes": "Cough and fever.",
                "wbc": 14.0,
                "crp": 90.0,
                "vitals_summary": None,
            }
        ],
        "guideline_context": None,
        "analysis": None,
        "reasoning": None,
        "report": None,
    }

    with patch("backend.agent.nodes.retrieve_context") as mock_retrieve:
        mock_retrieve.side_effect = [
            "No relevant guideline or literature content found.",
            "[Guideline excerpt 1, source: NG250.pdf]\nCRP >100 mg/L indicates severe inflammation.",
        ]

        mock_responses = [
            AIMessage(content="community-acquired pneumonia CRP severity threshold"),  # rewrite_query_node
            AIMessage(content="SUFFICIENT\nGuideline excerpt directly covers CRP severity thresholds."),  # grade #2
            AIMessage(content="Right basal opacity aligns with elevated CRP."),  # analysis
            AIMessage(content="Elevated CRP aligns with guideline threshold [Source: NG250.pdf]."),  # reasoning
            AIMessage(content="RESPIRATORY CONSULTATION REPORT\nFindings support pneumonia [Source: NG250.pdf]."),  # report
        ]

        with patch.object(ChatOpenAI, "invoke", side_effect=mock_responses) as mock_llm:
            final_state = compiled_agent.invoke(initial_state)

        assert mock_retrieve.call_count == 2
        assert mock_llm.call_count == 5  # rewrite, grade#2, analysis, reasoning, report -- grade#1 short-circuited
        assert final_state["retrieval_attempts"] == 2
        assert final_state["retrieval_sufficient"] is True
        assert "CRP severity threshold" in final_state["retrieval_query"]
        assert "NG250.pdf" in final_state["guideline_context"]


def test_scenario_f_retrieval_cap_forces_proceed(compiled_agent, base_patient_summary, monkeypatch):
    """If the grader keeps judging retrieval insufficient, the loop still
    terminates at MAX_RETRIEVAL_ATTEMPTS and proceeds with the last context
    instead of retrying forever."""

    monkeypatch.setattr("backend.agent.nodes.MAX_RETRIEVAL_ATTEMPTS", 2)

    initial_state: AgentState = {
        "patient_summary": base_patient_summary,
        "records": [
            {
                "created_at": "2026-07-12 10:00",
                "probability": 0.6,
                "label": "Pneumonia",
                "attention_focus": "Right basal opacity",
                "notes": "Cough and fever.",
                "wbc": 14.0,
                "crp": 90.0,
                "vitals_summary": None,
            }
        ],
        "guideline_context": None,
        "analysis": None,
        "reasoning": None,
        "report": None,
    }

    with patch("backend.agent.nodes.retrieve_context") as mock_retrieve:
        mock_retrieve.side_effect = [
            "[Guideline excerpt 1, source: unrelated.pdf]\nUnrelated topic, off-target for this case.",
            "[Guideline excerpt 1, source: unrelated.pdf]\nStill off-target after rewrite.",
        ]

        mock_responses = [
            AIMessage(content="INSUFFICIENT\nContext doesn't address this case."),  # grade #1
            AIMessage(content="community-acquired pneumonia CRP severity threshold"),  # rewrite_query_node
            AIMessage(content="INSUFFICIENT\nStill off-topic after rewrite."),  # grade #2 -- cap hit, forced through
            AIMessage(content="Right basal opacity aligns with elevated CRP."),  # analysis
            AIMessage(content="Best available synthesis despite thin guideline coverage."),  # reasoning
            AIMessage(content="RESPIRATORY CONSULTATION REPORT\nFindings support pneumonia."),  # report
        ]

        with patch.object(ChatOpenAI, "invoke", side_effect=mock_responses) as mock_llm:
            final_state = compiled_agent.invoke(initial_state)

        assert mock_retrieve.call_count == 2  # capped at MAX_RETRIEVAL_ATTEMPTS, not retried a 3rd time
        assert mock_llm.call_count == 6
        assert final_state["retrieval_attempts"] == 2
        assert final_state["retrieval_sufficient"] is False  # grader's real verdict is preserved even though we proceeded
        assert final_state["report"] is not None