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


def summarize_attention(attn_weights) -> str:
    """Turn raw attention weights into a short interpretable label.
    attn_weights shape: [1, 1, num_patches]"""
    weights = attn_weights.squeeze().detach().cpu()
    max_weight = weights.max().item()
    if max_weight > 0.05:
        return "concentrated on specific image regions (suggests a localized finding)"
    return "spread broadly across the image (suggests a diffuse or less certain finding)"