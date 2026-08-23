"""
Each node's prompt is built only from data already in AgentState (patient
history + clinician-selected prediction records) — never invented context.
This is the grounding mechanism that keeps reports traceable to real model output.
"""

import os

from langchain_openai import ChatOpenAI
from .state import AgentState
from .rag import RAG_MODE_SOURCES, build_retrieval_query, retrieve_context
from backend.models.gradcam import SwinGradCAM, overlay_heatmap

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # low temp -> deterministic, less hallucination

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "report_template.txt")



def _load_report_template() -> str:
    """Read fresh from disk each call (not cached at import) so the template
    can be edited without restarting the server."""
    with open(TEMPLATE_PATH, "r") as f:
        return f.read()


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


def retrieval_node(state: AgentState) -> dict:
    records_block = _render_records(state["records"])
    patient_summary = state["patient_summary"]

    rag_mode = state.get("rag_mode") or "hybrid"
    if rag_mode not in RAG_MODE_SOURCES:
        raise ValueError(
            f"Unknown rag_mode '{rag_mode}'. Expected one of {list(RAG_MODE_SOURCES)}."
        )
    sources = RAG_MODE_SOURCES[rag_mode]

    query = build_retrieval_query(patient_summary, records_block)
    context = retrieve_context(query, k=4, sources=sources)

    # Visual terminal indicator
    print("\n" + "=" * 60)
    print(" [RAG PIPELINE EXECUTED]")
    print(f" Mode: {rag_mode} (sources: {sources})")
    print(f" Query: {query[:100]}...")
    print(f" Context Retrieved: {len(context)} characters")
    print("=" * 60 + "\n")

    # Return ONLY the key updated in this node
    return {"guideline_context": context}


def analysis_node(state: AgentState) -> dict:
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

    analysis_res = llm.invoke(prompt).content
    
    # Return ONLY the key updated in this node
    return {"analysis": analysis_res}


def reasoning_node(state: AgentState) -> dict:
    records_block = _render_records(state["records"])
    patient_summary = state["patient_summary"]
    guidelines = state.get("guideline_context", "No guidelines provided.")

    prompt = """
        Act as a Senior Consultant Respiratory Physician conducting a formal diagnostic review.

        INPUT DATA:
        Patient Demographics & Features: {patient_data}
        Vision Classifier Output & Visual Attention: {analysis}
        Retrieved Guidance: {guideline_context}

        TASK:
        Deliver a concise clinical synthesis evaluating the prediction. 

        STRICT FORMATTING RULES:
        1. Do NOT use meta-language or planning text (e.g., "Let's assume", "Step 1", "To perform a clinical validation").
        2. State clinical facts directly.
        3. Quantify alignment by directly pairing patient findings with retrieved thresholds.

        EXPECTED STRUCTURE:

        CLINICAL IMPRESSION & MODEL VALIDATION:
        - State whether the model classification is valid, citing primary radiological and lab drivers.

        SPECIFIC GUIDELINE MAPPING:
        - Inflammatory Baseline: Pair patient CRP/WBC against guideline markers (e.g., NG250 CRP threshold > 100 mg/L vs patient CRP of 165 mg/L).
        - Oxygenation & Vitals: Pair patient SpO2/vitals against guideline admission boundaries (e.g., NG250 SpO2 threshold < 92% vs patient SpO2 of 89%).
        - Imaging Synthesis: Explain how the visual attention distribution aligns with expected radiological patterns.

        MANAGEMENT RATIONALE:
        - State the care setting escalation dictated by these threshold breaches.
        """

    reasoning_res = llm.invoke(prompt).content
    
    # Return ONLY the key updated in this node
    return {"reasoning": reasoning_res}



def report_node(state: AgentState) -> dict:
    patient_summary = state["patient_summary"]
    records_block = _render_records(state["records"])
    analysis = state["analysis"]
    reasoning = state["reasoning"]
    guidelines = state.get("guideline_context", "No guidelines provided.")
    template = _load_report_template()

    prompt = f"""You are a clinical reporting assistant. Generate a Respiratory Consultation
        Report using ONLY the information below.

        Formatting rules:
        - Follow the template's structure, section numbers, and field labels exactly -- do not add, remove, rename, or reorder sections.
        - Use British English spelling.
        - Use explicit, objective language in the active voice.
        - Do not add conversational text, preambles, or metadata outside the structured report and retrieved source information.
        - For any field the source information does not cover, write exactly: Not available/Not assessed. Never estimate or invent a value.
        - Explicitly reference retrieved guideline sources (e.g. [Source: NG250]) when justifying clinical recommendations.

        {template}

        --- Source information ---
        Patient history: {patient_summary}

        Selected records (oldest to newest):
        {records_block}

        Retrieved Guidelines:
        {guidelines}

        Modality Analysis: {analysis}
        Consistency Check: {reasoning}"""

    report_res = llm.invoke(prompt).content
    
    # Return ONLY the key updated in this node
    return {"report": report_res}


def image_analysis_node(state):
    """
    LangGraph node responsible for running vision inference and generating
    visual explainability heatmaps via Grad-CAM.
    """
    image_data = state["image_tensor"]
    text_data = state["text_tensor"]
    lab_data = state["lab_tensor"]
    
    # Initialize Grad-CAM on Swin backbone's final stage
    target_layer = model.swin_backbone.layers[-1]
    grad_cam = SwinGradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap, predicted_class = grad_cam.generate_heatmap(image_data, text_data, lab_data)
    
    # Render overlay image for frontend display / report generation
    original_img = state["raw_image_np"]
    overlaid_img, norm_heatmap = overlay_heatmap(heatmap, original_img)
    
    # Save results to agent state for downstream reasoning and report nodes
    state["prediction"] = predicted_class
    state["visual_explanation"] = {
        "heatmap": norm_heatmap,
        "overlay_image": overlaid_img,
        "explanation_summary": f"Grad-CAM highlighted high-activation regions driving class {predicted_class}."
    }
    
    return state