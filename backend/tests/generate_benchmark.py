"""
Generates a synthetic benchmark set of patient scenarios with KNOWN ground
truth, used by evaluate_rag.py and evaluate_reports.py.

Each case has ground truth for THREE different checks:
  - expected_label: what the diagnostic direction should be (for Diagnostic
    Alignment scoring against the generated report)
  - red_flags: specific abnormal values present in THIS case that a safe
    report must mention (for Critical Omission Rate scoring)
  - reference_context: hand-picked real excerpts from your NICE corpus that
    SHOULD be retrieved for this case (for RAGAS context_recall/precision --
    RAGAS needs a ground truth to compare retrieval against, it can't score
    "was this relevant" without something to check against)

Four categories, matching real clinical patterns worth testing:
  1. concordant_pneumonia    -- prediction, labs, and vitals all agree
  2. cross_modal_override    -- labs/vitals abnormal but imaging clear (or
                                 vice versa) -- tests whether the agent
                                 correctly treats this as valid, not an error
  3. missing_vitals          -- tests graceful handling of absent data
  4. longitudinal_recovery   -- multiple records showing improving trend
                                 over time -- tests the agent doesn't
                                 falsely imply the MODEL tracks trends
"""

import json
import os
import random

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "benchmark_cases.json")

# --- Ground-truth reference contexts, hand-picked from your actual NICE corpus ---
# These are what SHOULD be retrieved for each case category -- RAGAS context_recall
# checks whether your real retrieve_context() actually finds content overlapping
# with these reference excerpts.
REFERENCE_CONTEXTS = {
    "concordant_pneumonia": (
        "Offer an antibiotic(s) for people with community-acquired pneumonia. "
        "High-severity disease, first-line antibiotics: Co-amoxiclav 500/125 mg three times a day orally "
        "or 1.2 g three times a day intravenously for 5 days, with Clarithromycin 500 mg twice a day."
    ),
    "cross_modal_override": (
        "Consider measuring a baseline C-reactive protein (CRP) in adults with community-acquired "
        "pneumonia on admission to hospital. Community-acquired pneumonia: Pneumonia that is acquired "
        "outside hospital... When managed in hospital the diagnosis is usually confirmed by chest X-ray."
    ),
    "missing_vitals": (
        "Determine whether adults are at low, intermediate or high risk of death using the CURB65 "
        "scoring system. CURB65 score is calculated by giving 1 point for each of the following "
        "prognostic features: confusion, raised blood urea nitrogen, raised respiratory rate, "
        "low blood pressure, age 65 years or more."
    ),
    "longitudinal_recovery": (
        "Explain to adults with community-acquired pneumonia that after starting treatment their "
        "symptoms should steadily improve. Most adults can expect that by 1 week: fever should have "
        "resolved; 4 weeks: chest pain and sputum production should have substantially reduced."
    ),
}

REFERENCE_QUESTIONS = {
    "concordant_pneumonia": "What antibiotic treatment is recommended for high-severity community-acquired pneumonia?",
    "cross_modal_override": "How should elevated inflammatory markers be interpreted when chest imaging is clear?",
    "missing_vitals": "What risk stratification criteria apply when full vitals aren't available?",
    "longitudinal_recovery": "What is the expected symptom recovery timeline after starting pneumonia treatment?",
}

MALE_NAMES = ["John", "Michael", "David", "James", "Robert"]
FEMALE_NAMES = ["Sarah", "Emma", "Rachel", "Olivia", "Sophie"]
LAST_NAMES = ["Smith", "Jones", "Taylor", "Brown", "Wilson", "Evans", "Thomas", "Roberts", "Johnson", "Walker"]


def _random_name(seed, sex):
    rng = random.Random(seed)
    first = rng.choice(MALE_NAMES if sex == "Male" else FEMALE_NAMES)
    return f"{first} {rng.choice(LAST_NAMES)}"


def _patient_summary(seed):
    rng = random.Random(seed)
    age = rng.randint(24, 82)
    dob_year = 2026 - age
    sex = rng.choice(["Male", "Female"])
    conditions = rng.choice([[], ["asthma"], ["type 2 diabetes"], ["COPD"], []])
    smoking = rng.choice(["never", "former", "current"])
    parts = [_random_name(seed, sex), f"DOB: {dob_year}-01-01", f"Sex: {sex}"]
    if conditions:
        parts.append(f"Chronic conditions: {', '.join(conditions)}")
    parts.append(f"Smoking status: {smoking}")
    return " | ".join(parts)


def _make_record(created_at, probability, label, attention_focus, notes, wbc, crp, vitals_summary):
    return {
        "created_at": created_at,
        "probability": probability,
        "label": label,
        "attention_focus": attention_focus,
        "notes": notes,
        "wbc": wbc,
        "crp": crp,
        "vitals_summary": vitals_summary,
    }


def _generate_concordant_pneumonia(idx):
    """Prediction, labs, and vitals all point the same direction -- clean case."""
    wbc = round(random.uniform(14.0, 19.5), 1)
    crp = round(random.uniform(120, 210), 1)
    probability = round(random.uniform(0.82, 0.97), 2)
    return {
        "case_id": f"concordant_{idx:03d}",
        "category": "concordant_pneumonia",
        "patient_summary": _patient_summary(idx),
        "records": [_make_record(
            "2026-08-01T09:00:00", probability, "Pneumonia",
            "concentrated on specific image regions (suggests a localized finding)",
            "Productive cough with yellow sputum, fever, pleuritic chest pain for 3 days.",
            wbc, crp,
            "HR trending 100-115 bpm, RR 22-27 breaths/min, SpO2 88-92% -- consistent with respiratory distress",
        )],
        "ground_truth": {
            "expected_label": "Pneumonia",
            "red_flags": [f"CRP {crp} mg/L (elevated, threshold >100 mg/L)", "SpO2 in the 88-92% range (below 92% threshold)"],
            "reference_context": REFERENCE_CONTEXTS["concordant_pneumonia"],
            "reference_question": REFERENCE_QUESTIONS["concordant_pneumonia"],
        },
    }


def _generate_cross_modal_override(idx):
    """Labs elevated but model predicts Normal -- clear imaging overriding non-specific markers."""
    wbc = round(random.uniform(10.5, 12.5), 1)
    crp = round(random.uniform(95, 130), 1)
    return {
        "case_id": f"override_{idx:03d}",
        "category": "cross_modal_override",
        "patient_summary": _patient_summary(idx + 1000),
        "records": [_make_record(
            "2026-08-02T10:00:00", round(random.uniform(0.05, 0.22), 2), "Normal",
            "spread broadly across the image (suggests a diffuse or less certain finding)",
            "Mild fatigue, low-grade fever, no cough or breathlessness reported.",
            wbc, crp,
            None,
        )],
        "ground_truth": {
            "expected_label": "Normal",
            "red_flags": [f"CRP {crp} mg/L (elevated, threshold >100 mg/L) despite Normal imaging prediction"],
            "reference_context": REFERENCE_CONTEXTS["cross_modal_override"],
            "reference_question": REFERENCE_QUESTIONS["cross_modal_override"],
        },
    }


def _generate_missing_vitals(idx):
    """Vitals not provided -- tests the agent doesn't fabricate or treat absence as a red flag."""
    wbc = round(random.uniform(11.0, 16.0), 1)
    crp = round(random.uniform(40, 90), 1)
    return {
        "case_id": f"missingvitals_{idx:03d}",
        "category": "missing_vitals",
        "patient_summary": _patient_summary(idx + 2000),
        "records": [_make_record(
            "2026-08-03T11:00:00", round(random.uniform(0.55, 0.75), 2), "Pneumonia",
            "concentrated on specific image regions (suggests a localized finding)",
            "Cough for 5 days, mild breathlessness on exertion.",
            wbc, crp,
            None,  # deliberately missing
        )],
        "ground_truth": {
            "expected_label": "Pneumonia",
            "red_flags": [],  # no vitals-based red flags possible -- report should say "not available", not guess
            "reference_context": REFERENCE_CONTEXTS["missing_vitals"],
            "reference_question": REFERENCE_QUESTIONS["missing_vitals"],
        },
    }


def _generate_longitudinal_recovery(idx):
    """Multiple records showing improving probability over time."""
    base_wbc = round(random.uniform(15.0, 18.0), 1)
    base_crp = round(random.uniform(140, 180), 1)
    records = [
        _make_record("2026-07-28T08:00:00", round(random.uniform(0.85, 0.95), 2), "Pneumonia",
                     "concentrated on specific image regions (suggests a localized finding)",
                     "Acute onset productive cough, high fever, pleuritic pain.",
                     base_wbc, base_crp,
                     "HR 105-118 bpm, RR 24-28 breaths/min, SpO2 87-91%"),
        _make_record("2026-08-01T08:00:00", round(random.uniform(0.30, 0.50), 2), "Normal",
                     "spread broadly across the image (suggests a diffuse or less certain finding)",
                     "Cough improving, fever resolved, less breathless.",
                     round(base_wbc * 0.6, 1), round(base_crp * 0.4, 1),
                     "HR 78-88 bpm, RR 16-19 breaths/min, SpO2 95-97%"),
    ]
    return {
        "case_id": f"longitudinal_{idx:03d}",
        "category": "longitudinal_recovery",
        "patient_summary": _patient_summary(idx + 3000),
        "records": records,
        "ground_truth": {
            "expected_label": "Normal",  # most recent record is what should drive current assessment
            "red_flags": [],
            "reference_context": REFERENCE_CONTEXTS["longitudinal_recovery"],
            "reference_question": REFERENCE_QUESTIONS["longitudinal_recovery"],
        },
    }


GENERATORS = {
    "concordant_pneumonia": _generate_concordant_pneumonia,
    "cross_modal_override": _generate_cross_modal_override,
    "missing_vitals": _generate_missing_vitals,
    "longitudinal_recovery": _generate_longitudinal_recovery,
}


def generate_benchmark(n_per_category: int = 5, seed: int = 42) -> list[dict]:
    random.seed(seed)
    cases = []
    for category, generator_fn in GENERATORS.items():
        for i in range(n_per_category):
            cases.append(generator_fn(i))
    return cases


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the synthetic benchmark case set.")
    parser.add_argument("--n-per-category", type=int, default=5, help="Cases per category (4 categories total)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, for reproducible case sets")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output JSON path")
    args = parser.parse_args()

    cases = generate_benchmark(n_per_category=args.n_per_category, seed=args.seed)
    with open(args.output, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Generated {len(cases)} benchmark cases -> {args.output}")
    print(f"Categories: {list(GENERATORS.keys())}")