"""
Each node's prompt is built only from data already in AgentState (patient
history + clinician-selected prediction records) — never invented context.
This is the grounding mechanism that keeps reports traceable to real model output.
"""

from langchain_openai import ChatOpenAI
from .state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # low temp -> deterministic, less hallucination


def _render_records(records) -> str:
    """Turns the list of selected predictions into a readable block for the prompt."""
    lines = []
    for i, r in enumerate(records, start=1):
        lines.append(
            f"[{i}] {r['created_at']} — Prediction: {r['label']} ({r['probability']:.2f}). "
            f"Attention: {r['attention_focus']}. Notes: {r['notes']}. "
            f"WBC: {r.get('wbc')}, CRP: {r.get('crp')}."
        )
    return "\n".join(lines)


def analysis_node(state: AgentState) -> AgentState:
    records_block = _render_records(state["records"])
    patient_summary = state["patient_summary"]
    
    prompt = f"""You are assisting with interpreting a multimodal pneumonia classifier's output across
one or more selected clinical encounters for the same patient.

CRITICAL ARCHITECTURE RULES:
1. Every record contains a chest X-ray image (reflected in the 'Attention' field) alongside the lab data. There are NO records without imaging data.
2. The model evaluates each encounter as a completely independent, static point in time. It has no temporal memory, recurrence, or awareness of trends. Any change in prediction across records is driven by differences in the input features (such as the X-ray image tracking localized findings), not a calculated trend over time.

Patient history: {patient_summary}

Selected records (oldest to newest):
{records_block}

In 3-4 sentences, describe how the image attention focus and the clinical/lab data interact to drive these results. If multiple records are present, explain the changes based purely on differing feature inputs between independent encounters without implying the model tracks a temporal trend. Use only the information above."""
    
    state["analysis"] = llm.invoke(prompt).content
    return state


def reasoning_node(state: AgentState) -> AgentState:
    records_block = _render_records(state["records"])
    patient_summary = state["patient_summary"]
    
    prompt = f"""Check the following predictions for internal consistency against standard
lab reference ranges (WBC normal ~4.5-11.0 x10^9/L, CRP normal <10 mg/L).

CRITICAL CLINICAL RULES:
1. Elevated WBC and CRP indicate systemic inflammation but do not automatically guarantee localized pneumonia. 
2. If labs are highly elevated but the model predicts 'Normal', this is a clinically valid cross-modal override. It means the chest X-ray showed clear lung fields, which correctly overrode the non-specific blood markers. Do NOT flag this as a model error or contradiction.

Patient history: {patient_summary}

Selected records:
{records_block}

In 3-4 sentences, evaluate whether the combination of lab values and image attention focus supports the clinical validity of each independent prediction. Only flag a record as inconsistent if the model's prediction directly contradicts both the image attention behavior and the clinical context provided. Do not invent information."""
    
    state["reasoning"] = llm.invoke(prompt).content
    return state


def report_node(state: AgentState) -> AgentState:
    patient_summary = state["patient_summary"]
    analysis = state["analysis"]
    reasoning = state["reasoning"]
    
    prompt = f"""Write a short structured clinical AI report using ONLY the information below.

Patient history: {patient_summary}
Modality Analysis: {analysis}
Consistency Check: {reasoning}

Format exactly as:
Summary:
Modality Analysis:
Consistency Check:
Recommendation: (e.g. flag for radiologist review if inconsistent)"""
    
    state["report"] = llm.invoke(prompt).content
    return state