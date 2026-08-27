"""
Each node's prompt is built only from data already in AgentState (patient
history + clinician-selected prediction records) — never invented context.
This is the grounding mechanism that keeps reports traceable to real model output.
"""

import os
import re

from langchain_openai import ChatOpenAI
from .state import AgentState
from .rag import (
    RAG_MODE_SOURCES,
    build_retrieval_query,
    retrieve_context,
    grade_retrieval,
    rewrite_retrieval_query,
)
from backend.models.gradcam import SwinGradCAM, overlay_heatmap

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # low temp -> deterministic, less hallucination

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "report_template.txt")

CITATION_PATTERN = re.compile(r"\[Source:\s*[^\]]+\]", re.IGNORECASE)

# Initial attempt + up to 2 retries. The grade/rewrite loop (retrieval ->
# grade_retrieval -> rewrite_query -> retrieval -> ...) is capped by this so a
# corpus that just doesn't cover a topic can't loop forever -- once hit, the
# pipeline proceeds with whatever was last retrieved instead of blocking.
MAX_RETRIEVAL_ATTEMPTS = 3



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

    # A rewrite from a prior failed grade takes precedence over rebuilding
    # from scratch -- that's what makes this the "retrieve again" leg of the
    # grade/rewrite loop rather than always restarting from the same query.
    query = state.get("retrieval_query") or build_retrieval_query(patient_summary, records_block)
    context = retrieve_context(query, k=4, sources=sources)
    attempts = (state.get("retrieval_attempts") or 0) + 1

    # Visual terminal indicator
    print("\n" + "=" * 60)
    print(" [RAG PIPELINE EXECUTED]")
    print(f" Mode: {rag_mode} (sources: {sources})")
    print(f" Attempt: {attempts}/{MAX_RETRIEVAL_ATTEMPTS}")
    print(f" Query: {query[:100]}...")
    print(f" Context Retrieved: {len(context)} characters")
    print("=" * 60 + "\n")

    return {
        "guideline_context": context,
        "retrieval_query": query,
        "retrieval_attempts": attempts,
    }


def grade_retrieval_node(state: AgentState) -> dict:
    """
    Grade step of the corrective-RAG loop: judges whether the context just
    retrieved is actually usable evidence for this case. build_agent() routes
    on retrieval_sufficient -- True proceeds to analysis, False loops to
    rewrite_query_node (unless MAX_RETRIEVAL_ATTEMPTS has been hit).
    """
    sufficient, reasoning = grade_retrieval(state["retrieval_query"], state["guideline_context"])

    print(
        f"\n[RAG GRADE] attempt {state.get('retrieval_attempts')}: "
        f"{'SUFFICIENT' if sufficient else 'INSUFFICIENT'} -- {reasoning}\n"
    )

    return {
        "retrieval_sufficient": sufficient,
        "retrieval_grade_reasoning": reasoning,
    }


def rewrite_query_node(state: AgentState) -> dict:
    """Rewrite step: reformulates the query after a failed grade, then loops
    back to retrieval_node via build_agent()'s rewrite_query -> retrieval edge."""
    rewritten = rewrite_retrieval_query(state["retrieval_query"], state.get("retrieval_grade_reasoning", ""))

    print(f"\n[RAG REWRITE] '{state['retrieval_query'][:80]}' -> '{rewritten[:80]}'\n")

    return {"retrieval_query": rewritten}


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
    analysis = state["analysis"]
    guidelines = state.get("guideline_context", "No guidelines provided.")

    prompt = f"""
        Act as a Senior Consultant Respiratory Physician conducting a formal diagnostic review.

        INPUT DATA:
        Patient Demographics & Features: {patient_summary}

        Selected records (oldest to newest):
        {records_block}

        Vision Classifier Output & Visual Attention: {analysis}
        Retrieved Guidance: {guidelines}

        TASK:
        Deliver a concise clinical synthesis evaluating the prediction.

        STRICT FORMATTING RULES:
        1. Do NOT use meta-language or planning text (e.g., "Let's assume", "Step 1", "To perform a clinical validation").
        2. State clinical facts directly.
        3. Quantify alignment by directly pairing patient findings with retrieved thresholds.
        4. Each excerpt in Retrieved Guidance above is tagged with its own source (e.g. "source: NG250-summary.pdf" or "PMID: 12345678"). Every guideline threshold you cite MUST come from one of those excerpts and MUST carry an inline citation immediately after it, in the exact form [Source: <source or PMID from that excerpt>]. Never state a guideline threshold that isn't backed by a Retrieved Guidance excerpt -- if none of the excerpts cover a marker (e.g. no WBC threshold was retrieved), say so plainly instead of citing a general-knowledge figure with no source.

        EXPECTED STRUCTURE:

        CLINICAL IMPRESSION & MODEL VALIDATION:
        - State whether the model classification is valid, citing primary radiological and lab drivers.

        SPECIFIC GUIDELINE MAPPING:
        - Inflammatory Baseline: Pair the patient's actual CRP/WBC values (from the input data above) against relevant guideline markers found in the Retrieved Guidance, in the form "guideline threshold X [Source: ...] vs patient value Y". Only state a pairing when the patient's own value is present in the input data -- never invent or assume a value.
        - Oxygenation & Vitals: Pair the patient's actual SpO2/vitals against guideline admission boundaries, following the same rule. If vitals are marked "Not provided" for a record, explicitly state that oxygenation cannot be assessed against guideline thresholds for that record -- do not cite any SpO2 number, including any threshold or example value that appears in the Retrieved Guidance itself. Guideline figures are reference ranges from clinical literature, never this patient's own measurement.
        - Imaging Synthesis: Explain how the visual attention distribution aligns with expected radiological patterns.

        MANAGEMENT RATIONALE:
        - State the care setting escalation dictated by these threshold breaches, citing [Source: ...] for any guideline-driven escalation.
        """

    reasoning_res = llm.invoke(prompt).content
    
    # Return ONLY the key updated in this node
    return {"reasoning": reasoning_res}



def _has_citation(text: str) -> bool:
    return bool(CITATION_PATTERN.search(text))


def _ensure_citations(report_text: str, reasoning: str, guidelines: str) -> str:
    """
    Deterministic verification pass. The LLM's instruction to carry [Source: ...]
    citations forward from the Consistency Check into the report is followed
    inconsistently (~40% hit rate measured in backend/tests/evaluate_reports.py).
    Checking is a plain regex; only when it finds a real miss do we spend a
    corrective LLM call, rather than accepting the silent drop or re-running
    generation from scratch.
    """
    if not _has_citation(reasoning):
        return report_text  # nothing to carry forward -- nothing to check

    if _has_citation(report_text):
        return report_text  # already compliant

    fix_prompt = f"""The report below was generated from the Consistency Check below it, but
        dropped every [Source: ...] citation the Consistency Check backs its guideline-derived
        claims with.

        Revise the report to reinsert a [Source: ...] citation, copied verbatim from the
        Consistency Check or Retrieved Guidelines, next to each guideline-derived claim in
        sections 4 and 5. Change nothing else -- keep every other word, section, and field
        exactly as-is. Return the full corrected report.

        --- Report ---
        {report_text}

        --- Consistency Check (source of the missing citations) ---
        {reasoning}

        --- Retrieved Guidelines ---
        {guidelines}"""

    return llm.invoke(fix_prompt).content


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
        - The Consistency Check below already contains [Source: ...] citations for every guideline threshold it references. When section 4 or 5 restates or summarises a claim from the Consistency Check, you MUST keep that claim's [Source: ...] citation attached -- do not paraphrase a cited guideline comparison into prose that drops the citation. Do not invent a new citation that isn't already present in the Consistency Check or Retrieved Guidelines below.
        - NOTE the template below uses [square brackets] two different ways -- do not confuse them: the template's own [bracketed instructions] (e.g. "[Findings]", "[Summarise ...]") describe what to write and must NOT appear in your output; a [Source: ...] citation is the opposite -- it is literal output text you must actually write, copied verbatim from the Consistency Check or Retrieved Guidelines, every time you use a guideline-sourced claim.

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
    report_res = _ensure_citations(report_res, reasoning, guidelines)

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