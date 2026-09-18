import numpy as np


def build_patient_summary(patient) -> str:
    """Renders a patient's structured medical history into a short text block
    for the agent prompts. Only includes fields that are actually populated."""
    parts = [f"{patient.name}"]
    if patient.date_of_birth:
        parts.append(f"DOB: {patient.date_of_birth}")
    if patient.sex:
        parts.append(f"Sex: {patient.sex}")
    if patient.chronic_conditions:
        parts.append(f"Chronic conditions: {', '.join(patient.chronic_conditions)}")
    if patient.allergies:
        parts.append(f"Allergies: {', '.join(patient.allergies)}")
    if patient.current_medications:
        parts.append(f"Current medications: {', '.join(patient.current_medications)}")
    if patient.past_surgeries:
        parts.append(f"Past surgeries: {', '.join(patient.past_surgeries)}")
    if patient.smoking_status:
        parts.append(f"Smoking status: {patient.smoking_status}")
    if patient.family_history:
        parts.append(f"Family history: {patient.family_history}")
    return " | ".join(parts)


def format_vitals_summary(encounter) -> str | None:
    """Renders a ClinicalEncounter's vitals fields into a short text block for
    the agent prompts. Only includes fields that are actually populated."""
    if encounter is None:
        return None
    parts = []
    if encounter.heart_rate is not None:
        parts.append(f"HR {encounter.heart_rate} bpm")
    if encounter.blood_pressure:
        parts.append(f"BP {encounter.blood_pressure} mmHg")
    if encounter.respiratory_rate is not None:
        parts.append(f"RR {encounter.respiratory_rate}/min")
    if encounter.temperature is not None:
        parts.append(f"Temp {encounter.temperature}°C")
    if encounter.spo2 is not None:
        parts.append(f"SpO2 {encounter.spo2}%")
    return ", ".join(parts) if parts else None


def find_nearest_encounter(encounters, when):
    """Pick the ClinicalEncounter closest in time to `when`, for pairing a
    point-in-time vitals snapshot with the prediction made around the same
    visit. Returns None if there are no encounters to choose from."""
    if not encounters:
        return None
    return min(encounters, key=lambda e: abs((e.encounter_date - when).total_seconds()))


def summarize_spatial_focus(heatmap) -> str:
    """
    Turn a Grad-CAM heatmap into a short interpretable label describing
    whether the model's image focus was localized or diffuse.

    heatmap: normalised 2D array [H_patches, W_patches] from
    SwinGradCAM.generate_heatmap(), values in [0, 1] with max == 1
    (unless the CAM was entirely zero).

    This replaced an older version keyed off the fusion module's
    attn_weights -- that was valid when this model used cross-attention
    with shape [1, 1, num_patches] (one query attending over image
    patches), so "concentrated vs. spread" was a real per-patch
    measurement. The current architecture pools the image into a single
    token before a 4-token [image, text, labs, vitals] self-attention
    fusion, so attn_weights is now [batch, 4, 4] -- modality-to-modality,
    with no patch-level information left in it at all. The Grad-CAM
    heatmap is the only tensor that still carries real spatial
    information, so it's the only valid source for this claim now.
    """
    weights = np.asarray(heatmap)
    n_patches = weights.size
    if n_patches == 0 or weights.max() <= 0:
        return "not available (no activation detected)"

    # Peak-to-mean ratio, normalised by patch count so the score stays
    # comparable across different grid resolutions (e.g. 7x7 vs 8x8):
    # ranges from ~1/n_patches (activation spread uniformly across every
    # patch) up to 1.0 (a single hot patch, everything else exactly 0).
    # 0.15 corresponds roughly to activation concentrated in ~10% of the
    # image -- an empirical cutoff for "localized" vs. "broad", in the
    # same spirit as the old function's 0.05 cutoff.
    concentration = (weights.max() / (weights.mean() + 1e-6)) / n_patches
    if concentration > 0.15:
        return "concentrated on specific image regions (suggests a localized finding)"
    return "spread broadly across the image (suggests a diffuse or less certain finding)"