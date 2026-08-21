from unittest.mock import patch
from datasets import load_dataset
import pytest
import spacy

from backend.agent.graph import build_agent
from backend.agent.state import AgentState

# Load the biomedical NLP pipeline globally
nlp = spacy.load("en_core_sci_sm")


def extract_clinical_entities(text: str) -> set[str]:
    """Extracts biomedical entities and normalises them to lower-case lemmas."""
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        # Normalise entity tokens to base form (lemmas)
        cleaned_entity = " ".join([token.lemma_.lower() for token in ent if not token.is_stop])
        if len(cleaned_entity) > 2:
            entities.add(cleaned_entity)
    return entities


def compute_entity_metrics(generated_text: str, reference_text: str) -> dict[str, float]:
    """Calculates Precision, Recall, and F1-score based on medical entity extraction."""
    gen_entities = extract_clinical_entities(generated_text)
    ref_entities = extract_clinical_entities(reference_text)

    if not ref_entities and not gen_entities:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not ref_entities or not gen_entities:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positives = len(gen_entities.intersection(ref_entities))
    
    precision = true_positives / len(gen_entities)
    recall = true_positives / len(ref_entities)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gen_count": len(gen_entities),
        "ref_count": len(ref_entities),
    }


@pytest.fixture
def compiled_agent():
    return build_agent()


def test_agent_pipeline_entity_evaluation_openi(compiled_agent):
    """Evaluates the agent pipeline against OpenI ground truth clinical entities."""
    dataset = load_dataset("ykumards/open-i", split="train")

    # Filter for valid OpenI records
    eval_samples = [
        s
        for s in dataset
        if s.get("impression")
        and len(s["impression"].strip()) > 10
        and s.get("findings")
    ][:10]

    precision_scores, recall_scores, f1_scores = [], [], []

    for sample in eval_samples:
        reference_report = sample["impression"].strip()
        findings = sample.get("findings", "No detailed findings provided.")

        # Dynamically pass sample findings into the initial state
        initial_state: AgentState = {
            "patient_summary": "Arthur Pendelton | DOB: 1958-03-14 | Sex: Male | History: COPD, Type 2 Diabetes",
            "records": [
                {
                    "created_at": "2026-08-14 12:00",
                    "probability": 0.85,
                    "label": "Pneumonia",
                    "attention_focus": findings,
                    "notes": f"Radiological findings: {findings}",
                    "wbc": 14.2,
                    "crp": 115.0,
                    "vitals_summary": "HR: 102 bpm, BP: 118/76 mmHg, Temp: 38.4°C, RR: 24 breaths/min, SpO2: 92%",
                }
            ],
            "guideline_context": None,
            "analysis": None,
            "reasoning": None,
            "report": None,
        }

        # Execute agent graph with mocked retrieval
        with patch("backend.agent.nodes.retrieve_context") as mock_retrieve:
            mock_retrieve.return_value = "NG250 guideline context excerpt..."
            final_state = compiled_agent.invoke(initial_state)

        generated_report = final_state["report"]

        # Extract impression line if structured section exists
        if "CLINICAL INTERPRETATION:" in generated_report:
            comparison_text = generated_report.split("CLINICAL INTERPRETATION:")[1].split("\n")[0].strip()
        else:
            comparison_text = generated_report

        metrics = compute_entity_metrics(comparison_text, reference_report)

        precision_scores.append(metrics["precision"])
        recall_scores.append(metrics["recall"])
        f1_scores.append(metrics["f1"])

    mean_precision = sum(precision_scores) / len(precision_scores)
    mean_recall = sum(recall_scores) / len(recall_scores)
    mean_f1 = sum(f1_scores) / len(f1_scores)

    print("\n==========================================")
    print("   OpenI Entity Metrics Evaluation Results ")
    print("==========================================")
    print(f"Mean Clinical Entity Precision: {mean_precision:.4f}")
    print(f"Mean Clinical Entity Recall:    {mean_recall:.4f}")
    print(f"Mean Clinical Entity F1 Score:  {mean_f1:.4f}")
    print("==========================================")

    # Minimum diagnostic entity alignment check
    assert mean_f1 > 0.20