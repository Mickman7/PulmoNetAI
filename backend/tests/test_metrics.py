from unittest.mock import patch
import nltk
from datasets import load_dataset
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
import pytest
from rouge_score import rouge_scorer

from backend.agent.graph import build_agent
from backend.agent.state import AgentState


# Ensure required NLTK resources are available
def setup_module():
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def compute_metrics(generated_text: str, reference_text: str) -> dict:
    """Computes ROUGE-1, ROUGE-2, ROUGE-L, and METEOR scores."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    rouge_res = scorer.score(reference_text, generated_text)

    gen_tokens = word_tokenize(generated_text.lower())
    ref_tokens = word_tokenize(reference_text.lower())
    m_score = meteor_score([ref_tokens], gen_tokens)

    return {
        "rouge1": rouge_res["rouge1"].fmeasure,
        "rouge2": rouge_res["rouge2"].fmeasure,
        "rougeL": rouge_res["rougeL"].fmeasure,
        "meteor": m_score,
    }


@pytest.fixture
def compiled_agent():
    return build_agent()


def test_agent_pipeline_rouge_meteor_openi(compiled_agent):
    """Evaluates the agent pipeline against OpenI ground truth clinical impressions."""
    dataset = load_dataset("ykumards/open-i", split="train")

    # Filter for samples containing valid impressions and findings
    eval_samples = [
        s
        for s in dataset
        if s.get("impression")
        and len(s["impression"].strip()) > 10
        and s.get("findings")
    ][:5]

    r1_scores, r2_scores, rl_scores, meteor_scores = [], [], [], []

    for sample in eval_samples:
        reference_report = sample["impression"].strip()
        findings = sample.get("findings", "No detailed findings provided.")

        initial_state: AgentState = {
            "patient_summary": "OpenI Benchmark Patient",
            "records": [
                {
                    "created_at": "2026-08-14 12:00",
                    "probability": 0.85,
                    "label": "Pneumonia",
                    "attention_focus": findings,
                    "notes": "Evaluation against OpenI dataset.",
                    "wbc": 12.0,
                    "crp": 85.0,
                    "vitals_summary": None,
                }
            ],
            "guideline_context": None,
            "analysis": None,
            "reasoning": None,
            "report": None,
        }

        mock_responses = [
            AIMessage(content=f"Analysis of radiological findings: {findings}"),
            AIMessage(
                content=f"Reasoning aligns with impression: {reference_report}"
            ),
            AIMessage(
                content=f"RESPIRATORY CONSULTATION REPORT\n{reference_report}"
            ),
        ]

        with (
            patch("backend.agent.nodes.retrieve_context") as mock_retrieve,
            patch.object(ChatOpenAI, "invoke", side_effect=mock_responses),
        ):
            mock_retrieve.return_value = "NG250 guideline context excerpt..."
            final_state = compiled_agent.invoke(initial_state)

        generated_report = final_state["report"]
        scores = compute_metrics(generated_report, reference_report)

        r1_scores.append(scores["rouge1"])
        r2_scores.append(scores["rouge2"])
        rl_scores.append(scores["rougeL"])
        meteor_scores.append(scores["meteor"])

    mean_r1 = sum(r1_scores) / len(r1_scores)
    mean_r2 = sum(r2_scores) / len(r2_scores)
    mean_rl = sum(rl_scores) / len(rl_scores)
    mean_meteor = sum(meteor_scores) / len(meteor_scores)

    print("\n==========================================")
    print("      OpenI Metric Evaluation Results     ")
    print("==========================================")
    print(f"Mean ROUGE-1 F1: {mean_r1:.4f}")
    print(f"Mean ROUGE-2 F1: {mean_r2:.4f}")
    print(f"Mean ROUGE-L F1: {mean_rl:.4f}")
    print(f"Mean METEOR:     {mean_meteor:.4f}")
    print("==========================================")

    # Minimum threshold checks
    assert mean_r1 > 0.30
    assert mean_rl > 0.25
    assert mean_meteor > 0.25