"""
Each node's prompt is built only from data already in AgentState (patient
history + clinician-selected prediction records) — never invented context.
This is the grounding mechanism that keeps reports traceable to real model output.
"""

from langchain_openai import ChatOpenAI
from .state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # low temp -> deterministic, less hallucination


def _render_records(records) -> str:
    """Turns the list of selected predictions into a readable block for the prompt.
    Vitals are optional per record (1D-CNN input may not always be present) --
    explicitly marked as unavailable rather than silently dropped, so the LLM
    never assumes normal/absent vitals mean anything clinically."""
    lines = []
    for i, r in enumerate(records, start=1):
        vitals_line = r.get("vitals_summary") or "Not provided for this record"
        lines.append(
            f"[{i}] {r['created_at']} — Prediction: {r['label']} ({r['probability']:.2f}). "
            f"Attention: {r['attention_focus']}. Notes: {r['notes']}. "
            f"WBC: {r.get('wbc')}, CRP: {r.get('crp')}. "
            f"Vitals: {vitals_line}."
        )
    return "\n".join(lines)


def analysis_node(state: AgentState) -> AgentState:
    records_block = _render_records(state["records"])
    patient_summary = state["patient_summary"]

    prompt = f"""You are assisting with interpreting a multimodal pneumonia classifier's output across
        one or more selected clinical encounters for the same patient.

        CRITICAL ARCHITECTURE RULES:
        1. Every record contains a chest X-ray image (reflected in the 'Attention' field) alongside the lab data. There are NO records without imaging data.
        2. Vitals (heart rate, respiratory rate, SpO2 trend) are an OPTIONAL third input. When present, treat them strictly as supporting/corroborating evidence for the image+lab-driven prediction — never as a primary driver of the result. When a record says vitals were "Not provided", do not speculate about what they might have shown.
        3. The model evaluates each encounter as a completely independent, static point in time. It has no temporal memory, recurrence, or awareness of trends. Any change in prediction across records is driven by differences in the input features (such as the X-ray image tracking localized findings), not a calculated trend over time.

        Patient history: {patient_summary}

        Selected records (oldest to newest):
        {records_block}

        In 3-4 sentences, describe how the image attention focus and the clinical/lab data interact to drive these results, noting where available vitals corroborate that picture. If multiple records are present, explain the changes based purely on differing feature inputs between independent encounters without implying the model tracks a temporal trend. Use only the information above."""

    state["analysis"] = llm.invoke(prompt).content
    return state


def reasoning_node(state: AgentState) -> AgentState:
    records_block = _render_records(state["records"])
    patient_summary = state["patient_summary"]

    prompt = f"""Check the following predictions for internal consistency against standard
        lab reference ranges (WBC normal ~4.5-11.0 x10^9/L, CRP normal <10 mg/L) and, where available,
        typical vitals ranges (resting HR ~60-100 bpm, RR ~12-20 breaths/min, SpO2 ~95-100%).

        CRITICAL CLINICAL RULES:
        1. Elevated WBC and CRP indicate systemic inflammation but do not automatically guarantee localized pneumonia.
        2. If labs are highly elevated but the model predicts 'Normal', this is a clinically valid cross-modal override. It means the chest X-ray showed clear lung fields, which correctly overrode the non-specific blood markers. Do NOT flag this as a model error or contradiction.
        3. Vitals are supporting evidence only. If vitals are available and align with the prediction (e.g. low SpO2 / elevated RR alongside a Pneumonia prediction), note this as corroboration. If vitals are available but appear to conflict with the prediction, note it as a soft observation worth clinical attention — NOT as grounds to override or contradict the image+lab-driven result, since vitals were never the deciding input. If vitals are marked "Not provided", do not treat their absence as a red flag or evidence of anything.

        Patient history: {patient_summary}

        Selected records:
        {records_block}

        In 3-4 sentences, evaluate whether the combination of lab values, image attention focus, and (where available) vitals supports the clinical validity of each independent prediction. Only flag a record as inconsistent if the model's prediction directly contradicts both the image attention behavior and the clinical context provided. Do not invent information."""

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