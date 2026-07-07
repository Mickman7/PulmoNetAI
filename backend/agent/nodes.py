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
    prompt = f"""You are assisting with interpreting a pneumonia classifier's output across
one or more selected clinical encounters for the same patient.

Patient history: {state['patient_summary']}

Selected records (oldest to newest):
{records_block}

In 3-4 sentences, describe whether the image or the clinical/lab data appears to be
driving these results, and note any trend across the records if more than one is given.
Use only the information above."""
    state["analysis"] = llm.invoke(prompt).content
    return state


def reasoning_node(state: AgentState) -> AgentState:
    records_block = _render_records(state["records"])
    prompt = f"""Check the following predictions for internal consistency against standard
lab reference ranges (WBC normal ~4.5-11.0 x10^9/L, CRP normal <10 mg/L).

Patient history: {state['patient_summary']}

Selected records:
{records_block}

In 3-4 sentences, note whether the lab values support or contradict each prediction,
and flag any records that appear inconsistent. Do not invent information not given above."""
    state["reasoning"] = llm.invoke(prompt).content
    return state


def report_node(state: AgentState) -> AgentState:
    prompt = f"""Write a short structured clinical AI report using ONLY the information below.

Patient history: {state['patient_summary']}
Modality Analysis: {state['analysis']}
Consistency Check: {state['reasoning']}

Format as:
Summary:
Modality Analysis:
Consistency Check:
Recommendation: (e.g. flag for radiologist review if inconsistent)"""
    state["report"] = llm.invoke(prompt).content
    return state